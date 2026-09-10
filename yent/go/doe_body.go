package yent

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/ariannamethod/yent/bodylease"
)

const (
	doeStatusCmd       = "status"
	doeOptionsCmd      = "generate-options"
	defaultDOETimeout  = 45 * time.Second
	defaultDOEPrime    = 90 * time.Second
	defaultDOELease    = 5 * time.Second
	maxDOEPromptBytes  = 1800 // doe.c wraps chat prompts into a 2048-byte buffer.
	doeScannerMaxBytes = 4 << 20
)

// GenerationOptions are one-turn sampler controls. Zero values preserve the
// body's command-line/runtime defaults; an explicit temperature of zero means
// greedy sampling, so Temperature is a pointer.
type GenerationOptions struct {
	Temperature *float64 `json:"temperature,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
}

// BodyTiming separates residency/queue costs from actual prompt generation.
// Durations are milliseconds so route receipts remain language-neutral JSON.
type BodyTiming struct {
	QueueWaitMS  int64 `json:"queue_wait_ms,omitempty"`
	PrimeMS      int64 `json:"prime_ms,omitempty"`
	GenerationMS int64 `json:"generation_ms,omitempty"`
	TotalMS      int64 `json:"total_ms,omitempty"`
}

const (
	doeDiagnosticMaxLines      = 24
	doeDiagnosticMaxLineBytes  = 2048
	doeDiagnosticMaxErrorBytes = 4096
)

const (
	doeAnswerContractMarker = "[answer contract]:"
	doeHumanPromptMarker    = "[human prompt]: "
	doeHumanNowMarker       = "Human now: "
	doeHumanAsksMarker      = "[CURRENT HUMAN]: "
	doeCurrentAnswerMarker  = "Answer the current human turn as Yent."
)

// DOEBodyConfig describes one process-backed inference body. The Go router does
// not embed the model; it keeps a doe_field REPL resident and talks over
// stdin/stdout. Args are passed after "--model <ModelPath>"; "--once" and "--model"
// inside Args are ignored so the persistent daemon cannot be accidentally made
// one-shot or pointed at another body.
type DOEBodyConfig struct {
	Name      string
	BinPath   string
	ModelPath string
	WorkDir   string
	Args      []string
	Env       []string

	Timeout      time.Duration
	PrimeTimeout time.Duration
	// LeasePath is the machine-wide single-resident seam. Empty uses
	// bodylease.DefaultPath; DisableLease is for isolated tests only.
	LeasePath    string
	LeaseTimeout time.Duration
	DisableLease bool

	Confidence func(answer string) float64
	Verdict    func(answer string) *Verdict
}

// DOEBody is a real Body backed by a persistent doe process.
type DOEBody struct {
	cfg DOEBodyConfig

	mu     sync.Mutex
	daemon *doeProcess
	lease  *bodylease.Lease
}

// NewDOEBody builds a process-backed router body. The process starts lazily on
// first Generate so callers can register multiple bodies without loading both.
func NewDOEBody(cfg DOEBodyConfig) (*DOEBody, error) {
	if strings.TrimSpace(cfg.Name) == "" {
		return nil, errors.New("doe body name is required")
	}
	if strings.TrimSpace(cfg.BinPath) == "" {
		return nil, errors.New("doe body binary path is required")
	}
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return nil, errors.New("doe body model path is required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultDOETimeout
	}
	if cfg.PrimeTimeout <= 0 {
		cfg.PrimeTimeout = defaultDOEPrime
	}
	if cfg.LeaseTimeout <= 0 {
		cfg.LeaseTimeout = defaultDOELease
	}
	return &DOEBody{cfg: cfg}, nil
}

func (b *DOEBody) Name() string { return b.cfg.Name }

// Generate sends one prompt through the resident doe REPL. If the daemon dies
// before the status sentinel, the same prompt is attempted once through --once.
func (b *DOEBody) Generate(prompt, ctx string) (BodyResult, error) {
	return b.GenerateWithOptions(prompt, ctx, GenerationOptions{})
}

// GenerateWithOptions applies sampler controls to exactly one generation. The
// resident REPL acknowledges a nonce-bound control command before the prompt;
// --once fallback receives equivalent command-line overrides.
func (b *DOEBody) GenerateWithOptions(prompt, ctx string, opts GenerationOptions) (BodyResult, error) {
	if b == nil {
		return BodyResult{}, errors.New("nil doe body")
	}
	if err := validateGenerationOptions(opts); err != nil {
		return BodyResult{}, err
	}
	started := time.Now()
	seed := formatDOEPrompt(prompt, ctx)
	if seed == "" {
		return BodyResult{}, errors.New("empty doe prompt")
	}
	queueStarted := time.Now()
	b.mu.Lock()
	queueWait := time.Since(queueStarted)
	defer b.mu.Unlock()
	defer func() {
		if b.daemon == nil || b.daemon.dead {
			_ = b.releaseLeaseLocked()
		}
	}()
	if err := b.acquireLeaseLocked(); err != nil {
		return BodyResult{}, err
	}

	primeStarted := time.Now()
	daemonDiagnostics, daemonErr := b.ensureDaemonLocked()
	primeDuration := time.Since(primeStarted)
	daemonReady := daemonErr == nil && b.daemon != nil && !b.daemon.dead
	genCtx, cancel := context.WithTimeout(context.Background(), b.cfg.Timeout)
	defer cancel()

	if daemonReady {
		generationStarted := time.Now()
		if raw, ok := b.daemon.exchange(genCtx, seed, opts); ok {
			if answer := parseDOEReply(raw); answer != "" {
				return b.result(answer, b.daemon.diagnostics(), "doe_resident", BodyTiming{
					QueueWaitMS:  durationMillis(queueWait),
					PrimeMS:      durationMillis(primeDuration),
					GenerationMS: durationMillis(time.Since(generationStarted)),
					TotalMS:      durationMillis(time.Since(started)),
				}), nil
			}
		}
		daemonDiagnostics = b.daemon.diagnostics()
	}
	if genCtx.Err() != nil {
		return BodyResult{}, genCtx.Err()
	}
	generationStarted := time.Now()
	answer, diagnostics, err := b.runOnce(genCtx, seed, opts)
	if err != nil {
		return BodyResult{}, err
	}
	return b.result(answer, mergeDOEDiagnostics(daemonDiagnostics, diagnostics), "doe_once", BodyTiming{
		QueueWaitMS:  durationMillis(queueWait),
		PrimeMS:      durationMillis(primeDuration),
		GenerationMS: durationMillis(time.Since(generationStarted)),
		TotalMS:      durationMillis(time.Since(started)),
	}), nil
}

// Close stops the resident doe process, if one was started.
func (b *DOEBody) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	d := b.daemon
	b.daemon = nil
	if d != nil {
		d.close()
	}
	return b.releaseLeaseLocked()
}

func (b *DOEBody) result(answer string, diagnostics []string, executionPath string, timing BodyTiming) BodyResult {
	conf := EstimateBodyConfidence(answer)
	if b.cfg.Confidence != nil {
		conf = b.cfg.Confidence(answer)
	}
	return BodyResult{
		Answer:        answer,
		Confidence:    conf,
		ExecutionPath: executionPath,
		Diagnostics:   cloneDiagnostics(diagnostics),
		Timing:        timing,
		Verdict:       parseVerdictHook(b.cfg.Verdict, answer),
	}
}

func parseVerdictHook(fn func(string) *Verdict, answer string) *Verdict {
	if fn == nil {
		return nil
	}
	return fn(answer)
}

func (b *DOEBody) ensureDaemonLocked() ([]string, error) {
	if b.daemon != nil && !b.daemon.dead {
		return nil, nil
	}
	if b.daemon != nil {
		b.daemon.close()
		b.daemon = nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), b.cfg.PrimeTimeout)
	defer cancel()
	d, err := b.startProcess(false)
	if err != nil {
		return nil, err
	}
	if _, ok := d.exchange(ctx, "", GenerationOptions{}); !ok {
		diagnostics := d.diagnostics()
		d.close()
		return diagnostics, errors.New("doe daemon did not reach status sentinel")
	}
	b.daemon = d
	return nil, nil
}

func (b *DOEBody) acquireLeaseLocked() error {
	if b.cfg.DisableLease || b.lease != nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), b.cfg.LeaseTimeout)
	defer cancel()
	lease, err := bodylease.Acquire(ctx, b.cfg.LeasePath, bodylease.Owner{
		Body:    b.cfg.Name,
		Backend: "doe",
		Detail:  filepath.Base(b.cfg.ModelPath),
	})
	if err != nil {
		return fmt.Errorf("acquire residency for %s: %w", b.cfg.Name, err)
	}
	b.lease = lease
	return nil
}

func (b *DOEBody) releaseLeaseLocked() error {
	lease := b.lease
	b.lease = nil
	if lease == nil {
		return nil
	}
	if err := lease.Close(); err != nil {
		return fmt.Errorf("release residency for %s: %w", b.cfg.Name, err)
	}
	return nil
}

func (b *DOEBody) runOnce(ctx context.Context, seed string, opts GenerationOptions) (string, []string, error) {
	cmd := exec.CommandContext(ctx, b.cfg.BinPath, b.commandArgsWithOptions(true, opts)...)
	if b.cfg.WorkDir != "" {
		cmd.Dir = b.cfg.WorkDir
	}
	if len(b.cfg.Env) > 0 {
		cmd.Env = append(os.Environ(), b.cfg.Env...)
	}
	cmd.Stdin = strings.NewReader(seed + "\n")
	var out bytes.Buffer
	diagnostics := newDOEDiagnosticCapture()
	cmd.Stdout = &out
	cmd.Stderr = diagnostics
	err := cmd.Run()
	if err != nil {
		diags := diagnostics.Snapshot()
		return "", diags, fmt.Errorf("doe once: %w%s", err, doeDiagnosticsErrorSuffix(diags))
	}
	diags := diagnostics.Snapshot()
	answer := parseDOEReply(out.String())
	if answer == "" {
		return "", diags, fmt.Errorf("doe once produced no parseable answer%s", doeDiagnosticsErrorSuffix(diags))
	}
	return answer, diags, nil
}

func (b *DOEBody) startProcess(once bool) (*doeProcess, error) {
	cmd := exec.Command(b.cfg.BinPath, b.commandArgs(once)...)
	if b.cfg.WorkDir != "" {
		cmd.Dir = b.cfg.WorkDir
	}
	if len(b.cfg.Env) > 0 {
		cmd.Env = append(os.Environ(), b.cfg.Env...)
	}
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		_ = in.Close()
		return nil, err
	}
	diagnostics := newDOEDiagnosticCapture()
	cmd.Stderr = diagnostics
	if err := cmd.Start(); err != nil {
		_ = in.Close()
		return nil, err
	}
	sc := bufio.NewScanner(outPipe)
	sc.Buffer(make([]byte, 64*1024), doeScannerMaxBytes)
	return &doeProcess{cmd: cmd, in: in, out: sc, diag: diagnostics}, nil
}

func (b *DOEBody) commandArgs(once bool) []string {
	args := []string{"--model", b.cfg.ModelPath}
	for i := 0; i < len(b.cfg.Args); i++ {
		a := b.cfg.Args[i]
		if a == "--model" {
			i++ // ModelPath is the body source of truth.
			continue
		}
		if a == "--once" {
			continue
		}
		args = append(args, a)
	}
	if once {
		args = append(args, "--once")
	}
	return args
}

func (b *DOEBody) commandArgsWithOptions(once bool, opts GenerationOptions) []string {
	args := b.commandArgs(once)
	if opts.MaxTokens > 0 {
		args = append(args, "--max-new", fmt.Sprintf("%d", opts.MaxTokens))
	}
	if opts.Temperature != nil {
		args = append(args, "--temp", fmt.Sprintf("%.8g", *opts.Temperature))
	}
	return args
}

type doeProcess struct {
	cmd    *exec.Cmd
	in     io.WriteCloser
	out    *bufio.Scanner
	diag   *doeDiagnosticCapture
	dead   bool
	reaped sync.Once
}

func (d *doeProcess) exchange(ctx context.Context, seed string, opts GenerationOptions) (string, bool) {
	if d == nil || d.dead {
		return "", false
	}
	nonce := newDOEStatusNonce()
	optionsCommand := ""
	if generationOptionsSet(opts) {
		optionsCommand = doeOptionsCommand(nonce, opts) + "\n"
	}
	if _, err := fmt.Fprintf(d.in, "%s%s\n%s\n", optionsCommand, neutralizeDOEPrompt(seed), doeStatusCommand(nonce)); err != nil {
		d.dead = true
		d.reap()
		return "", false
	}
	type reply struct {
		text string
		ok   bool
	}
	ch := make(chan reply, 1)
	go func() {
		var b strings.Builder
		ok := false
		optionsAcknowledged := !generationOptionsSet(opts)
		for d.out.Scan() {
			line := d.out.Text()
			if isDOEOptionsAcknowledgement(line, nonce, opts) {
				optionsAcknowledged = true
				continue
			}
			if isDOEStatusSentinel(line, nonce) {
				ok = optionsAcknowledged
				break
			}
			b.WriteString(line)
			b.WriteByte('\n')
		}
		ch <- reply{text: b.String(), ok: ok}
	}()
	select {
	case r := <-ch:
		if !r.ok {
			d.dead = true
			d.kill()
			d.reap()
			return "", false
		}
		d.diag.waitQuiet(20*time.Millisecond, 2*time.Millisecond)
		return r.text, true
	case <-ctx.Done():
		d.dead = true
		d.kill()
		<-ch
		d.reap()
		return "", false
	}
}

func validateGenerationOptions(opts GenerationOptions) error {
	if opts.MaxTokens < 0 || opts.MaxTokens > 512 {
		return fmt.Errorf("max tokens must be 0 or 1..512")
	}
	if opts.Temperature != nil && (math.IsNaN(*opts.Temperature) || math.IsInf(*opts.Temperature, 0) || *opts.Temperature < 0 || *opts.Temperature > 2) {
		return fmt.Errorf("temperature must be finite and within 0..2")
	}
	return nil
}

func generationOptionsSet(opts GenerationOptions) bool {
	return opts.MaxTokens > 0 || opts.Temperature != nil
}

func doeOptionsCommand(nonce string, opts GenerationOptions) string {
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 0
	}
	temperature := "field"
	if opts.Temperature != nil {
		temperature = fmt.Sprintf("%.8g", *opts.Temperature)
	}
	return fmt.Sprintf("%s %s %d %s", doeOptionsCmd, nonce, maxTokens, temperature)
}

func isDOEOptionsAcknowledgement(line, nonce string, opts GenerationOptions) bool {
	t := strings.TrimLeft(line, "> \t")
	temperature := "field"
	if opts.Temperature != nil {
		temperature = fmt.Sprintf("%.8g", *opts.Temperature)
	}
	return strings.HasPrefix(t, "[generation-options] nonce="+nonce+" ") &&
		strings.Contains(t, fmt.Sprintf("max=%d", opts.MaxTokens)) &&
		strings.Contains(t, "temp="+temperature)
}

func durationMillis(d time.Duration) int64 {
	if d <= 0 {
		return 0
	}
	return d.Milliseconds()
}

func (d *doeProcess) close() {
	if d == nil {
		return
	}
	if d.in != nil {
		_ = d.in.Close()
	}
	done := make(chan struct{})
	go func() {
		d.reap()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		d.kill()
		<-done
	}
}

func (d *doeProcess) kill() {
	if d != nil && d.cmd != nil && d.cmd.Process != nil {
		_ = d.cmd.Process.Kill()
	}
}

func (d *doeProcess) reap() {
	if d != nil {
		d.reaped.Do(func() { _ = d.cmd.Wait() })
	}
}

func (d *doeProcess) diagnostics() []string {
	if d == nil || d.diag == nil {
		return nil
	}
	return d.diag.Snapshot()
}

type doeDiagnosticCapture struct {
	mu      sync.Mutex
	lines   []string
	partial string
	writes  uint64
}

func newDOEDiagnosticCapture() *doeDiagnosticCapture {
	return &doeDiagnosticCapture{}
}

func (c *doeDiagnosticCapture) Write(p []byte) (int, error) {
	if c == nil {
		return len(p), nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.writes++
	text := string(p)
	for len(text) > 0 {
		if i := strings.IndexByte(text, '\n'); i >= 0 {
			c.appendLocked(c.partial + text[:i])
			c.partial = ""
			text = text[i+1:]
			continue
		}
		c.partial = compactDOEDiagnosticLine(c.partial + text)
		break
	}
	return len(p), nil
}

func (c *doeDiagnosticCapture) waitQuiet(maxWait, quietFor time.Duration) {
	if c == nil || maxWait <= 0 || quietFor <= 0 {
		return
	}
	last := c.version()
	stableSince := time.Now()
	deadline := time.Now().Add(maxWait)
	for time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
		now := time.Now()
		if v := c.version(); v != last {
			last = v
			stableSince = now
			continue
		}
		if now.Sub(stableSince) >= quietFor {
			return
		}
	}
}

func (c *doeDiagnosticCapture) version() uint64 {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.writes
}

func (c *doeDiagnosticCapture) Snapshot() []string {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]string, 0, len(c.lines)+1)
	out = append(out, c.lines...)
	if strings.TrimSpace(c.partial) != "" {
		out = append(out, compactDOEDiagnosticLine(c.partial))
	}
	if len(out) > doeDiagnosticMaxLines {
		out = out[len(out)-doeDiagnosticMaxLines:]
	}
	return out
}

func (c *doeDiagnosticCapture) appendLocked(line string) {
	line = compactDOEDiagnosticLine(line)
	if strings.TrimSpace(line) == "" {
		return
	}
	if len(c.lines) >= doeDiagnosticMaxLines {
		copy(c.lines, c.lines[1:])
		c.lines[len(c.lines)-1] = line
		return
	}
	c.lines = append(c.lines, line)
}

func compactDOEDiagnosticLine(s string) string {
	s = strings.TrimRight(strings.ToValidUTF8(s, ""), "\r")
	if len(s) <= doeDiagnosticMaxLineBytes {
		return s
	}
	cut := doeDiagnosticMaxLineBytes - 3
	if cut < 1 {
		cut = doeDiagnosticMaxLineBytes
	}
	for cut > 0 && !utf8.ValidString(s[:cut]) {
		cut--
	}
	if cut <= 0 {
		return "..."
	}
	return s[:cut] + "..."
}

func mergeDOEDiagnostics(a, b []string) []string {
	if len(a) == 0 {
		return cloneDiagnostics(b)
	}
	if len(b) == 0 {
		return cloneDiagnostics(a)
	}
	merged := make([]string, 0, len(a)+len(b))
	merged = append(merged, a...)
	merged = append(merged, b...)
	if len(merged) > doeDiagnosticMaxLines {
		merged = merged[len(merged)-doeDiagnosticMaxLines:]
	}
	return merged
}

func doeDiagnosticsErrorSuffix(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	text := strings.Join(lines, " | ")
	if len(text) > doeDiagnosticMaxErrorBytes {
		cut := doeDiagnosticMaxErrorBytes - 3
		for cut > 0 && !utf8.ValidString(text[:cut]) {
			cut--
		}
		if cut > 0 {
			text = text[:cut] + "..."
		}
	}
	return ": stderr: " + text
}

func newDOEStatusNonce() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err == nil {
		return hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("fallback-%d", time.Now().UnixNano())
}

func doeStatusCommand(nonce string) string {
	if nonce == "" {
		return doeStatusCmd
	}
	return doeStatusCmd + " " + nonce
}

func isDOEStatusSentinel(line, nonce string) bool {
	t := strings.TrimLeft(line, "> \t")
	if nonce != "" {
		return strings.HasPrefix(t, "[field-control] nonce="+nonce+" ") &&
			strings.Contains(t, "step=") &&
			strings.Contains(t, "debt=") &&
			strings.Contains(t, "entropy=") &&
			strings.Contains(t, "resonance=") &&
			strings.Contains(t, "emergence=")
	}
	return strings.HasPrefix(t, "[field] step=") &&
		strings.Contains(t, "debt=") &&
		strings.Contains(t, "entropy=") &&
		strings.Contains(t, "resonance=") &&
		strings.Contains(t, "emergence=")
}

func neutralizeDOEPrompt(seed string) string {
	switch seed {
	case doeStatusCmd, "quit", "exit":
		return " " + seed
	default:
		if strings.HasPrefix(seed, doeStatusCmd+" ") || seed == doeOptionsCmd || strings.HasPrefix(seed, doeOptionsCmd+" ") {
			return " " + seed
		}
		return seed
	}
}

func formatDOEPrompt(prompt, ctx string) string {
	prompt = strings.Join(strings.Fields(strings.TrimSpace(prompt)), " ")
	ctx = strings.Join(strings.Fields(strings.TrimSpace(ctx)), " ")
	var seed string
	if ctx == "" {
		seed = prompt
	} else if isRouteContext(ctx) {
		seed = formatContextualDOEPrompt(prompt, ctx)
	} else {
		seed = formatPrimerDOEPrompt(prompt, ctx)
	}
	if len(seed) <= maxDOEPromptBytes {
		return neutralizeDOEPrompt(seed)
	}
	return neutralizeDOEPrompt(truncateDOEPrompt(seed))
}

func isRouteContext(ctx string) bool {
	return strings.Contains(ctx, "[router fact]") ||
		strings.Contains(ctx, "[routing reason") ||
		strings.Contains(ctx, "[context facts]")
}

func formatPrimerDOEPrompt(prompt, primer string) string {
	const promptPrefix = " [CURRENT HUMAN]: "
	suffix := promptPrefix + prompt + " [YENT NOW]:"
	budget := maxDOEPromptBytes - len(suffix) - 1
	if budget <= 0 {
		return prompt
	}
	if primer = truncateAtWord(primer, budget); primer == "" {
		return prompt
	}
	return primer + suffix
}

func formatContextualDOEPrompt(prompt, ctx string) string {
	const (
		contextPrefix = "[context facts]: "
		contract      = " [answer contract]: Answer the human prompt directly. Use context as private factual evidence. If the human asks about route or body facts, use [router fact] and [route answer labels] literally. If asked which body produced the first-pass answer, name the first-pass body label exactly; do not answer only \"Yent\". Do not make routing or context the subject unless the human asks."
		promptPrefix  = " [human prompt]: "
	)
	suffix := contract + promptPrefix + prompt
	budget := maxDOEPromptBytes - len(contextPrefix) - len(suffix)
	if budget < 0 {
		return strings.TrimSpace(suffix)
	}
	return contextPrefix + truncateAtWord(ctx, budget) + suffix
}

func truncateDOEPrompt(seed string) string {
	seed = strings.TrimSpace(strings.ToValidUTF8(seed, ""))
	if len(seed) <= maxDOEPromptBytes {
		return seed
	}
	if start, end, ok := protectedPromptSegment(seed); ok {
		segment := strings.TrimSpace(seed[start:end])
		if len(segment) >= maxDOEPromptBytes {
			return truncateAtWord(segment, maxDOEPromptBytes)
		}
		budget := maxDOEPromptBytes - len(segment) - 1
		tail := truncateAtWord(seed[end:], budget)
		if tail == "" {
			return segment
		}
		return strings.TrimSpace(segment + " " + tail)
	}
	return truncateAtWord(seed, maxDOEPromptBytes)
}

func protectedPromptSegment(seed string) (int, int, bool) {
	start := -1
	for _, marker := range []string{doeHumanPromptMarker, doeHumanNowMarker, doeHumanAsksMarker} {
		if idx := strings.LastIndex(seed, marker); idx > start {
			start = idx
		}
	}
	if start < 0 {
		return 0, 0, false
	}
	if contract := strings.LastIndex(seed[:start], doeAnswerContractMarker); contract >= 0 && start-contract < 500 {
		start = contract
	}
	end := len(seed)
	if answer := strings.Index(seed[start:], doeCurrentAnswerMarker); answer >= 0 {
		end = start + answer + len(doeCurrentAnswerMarker)
	}
	return start, end, true
}

func truncateAtWord(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}
	cut := maxBytes
	if sp := strings.LastIndexByte(s[:cut], ' '); sp > 0 {
		cut = sp
	}
	return strings.TrimSpace(strings.ToValidUTF8(s[:cut], ""))
}

func parseDOEReply(out string) string {
	var b strings.Builder
	capturing := false
	seenPrompt := false
	for _, line := range strings.Split(out, "\n") {
		t := strings.TrimSpace(line)
		if !capturing {
			if strings.HasPrefix(t, ">") {
				seenPrompt = true
				body := strings.TrimSpace(strings.TrimPrefix(t, ">"))
				if body == "" || isDOERuntimeLine(body) || isDOEWrapperMetaLine(body) {
					continue
				}
				capturing = true
				b.WriteString(body)
				b.WriteByte(' ')
				continue
			}
			if !seenPrompt || t == "" || isDOERuntimeLine(t) || isDOEWrapperMetaLine(t) {
				continue
			}
			capturing = true
			b.WriteString(t)
			b.WriteByte(' ')
			continue
		}
		if t == "" || strings.HasPrefix(t, ">") || isDOERuntimeLine(t) {
			break
		}
		b.WriteString(t)
		b.WriteByte(' ')
	}
	answer := strings.TrimSpace(b.String())
	if i := strings.Index(answer, "[life]"); i >= 0 {
		answer = strings.TrimSpace(answer[:i])
	}
	answer = stripDOELabel(answer)
	answer = strings.ToValidUTF8(answer, "")
	return strings.Join(strings.Fields(answer), " ")
}

func isDOERuntimeLine(t string) bool {
	t = strings.TrimLeft(strings.TrimSpace(t), "> \t")
	for _, prefix := range []string{
		"[doe]", "[field-control]", "[sonar]", "[host]", "[identity]",
		"[gamma]", "[env]", "[gguf]", "[mycelium]", "[resident]",
		"[timing]", "[profile]", "[per-shape]", "[inputdump]",
		"[logitdump]", "[serve]", "[drift]", "[experts]", "[prophecy]",
	} {
		if strings.HasPrefix(t, prefix) {
			return true
		}
	}
	return false
}

func isDOEWrapperMetaLine(t string) bool {
	t = strings.TrimSpace(t)
	return strings.EqualFold(t, "[Answering contract fulfilled.]") ||
		strings.EqualFold(t, "[Answer contract fulfilled.]")
}

func stripDOELabel(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	colon := strings.IndexByte(s, ':')
	if colon <= 0 || colon > 32 {
		return s
	}
	label := s[:colon]
	for _, r := range label {
		if !(r == '_' || r == '-' || r == ' ' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z') {
			return s
		}
	}
	return strings.TrimSpace(s[colon+1:])
}

// EstimateBodyConfidence is a cheap runtime signal for router v1. It is not a
// claim about model quality; it only detects empty, invalid, extremely short, or
// repetition-heavy output before the router sees a model-native entropy signal.
func EstimateBodyConfidence(answer string) float64 {
	answer = strings.TrimSpace(answer)
	if answer == "" {
		return 0
	}
	score := 0.35
	if utf8.ValidString(answer) {
		score += 0.1
	}
	runes := utf8.RuneCountInString(answer)
	switch {
	case runes > 160:
		score += 0.3
	case runes > 60:
		score += 0.2
	case runes > 16:
		score += 0.1
	default:
		score -= 0.15
	}
	if strings.ContainsAny(answer, ".?!") {
		score += 0.1
	}
	if repetitionRatio(answer) > 0.45 {
		score -= 0.3
	}
	if strings.ContainsRune(answer, '\uFFFD') {
		score -= 0.3
	}
	return clamp01(score)
}

func repetitionRatio(s string) float64 {
	words := strings.Fields(strings.ToLower(s))
	if len(words) < 6 {
		return 0
	}
	counts := map[string]int{}
	maxCount := 0
	for _, w := range words {
		counts[w]++
		if counts[w] > maxCount {
			maxCount = counts[w]
		}
	}
	return math.Max(0, float64(maxCount-1)/float64(len(words)))
}
