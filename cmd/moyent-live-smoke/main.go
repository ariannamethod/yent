package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	yent "github.com/ariannamethod/yent/yent/go"
)

type turnLog struct {
	Kind      string           `json:"kind"`
	Prompt    string           `json:"prompt,omitempty"`
	Error     string           `json:"error,omitempty"`
	Body      string           `json:"body,omitempty"`
	Escalated bool             `json:"escalated,omitempty"`
	Reason    string           `json:"reason,omitempty"`
	SeamID    int64            `json:"seam_id,omitempty"`
	Duration  string           `json:"duration,omitempty"`
	Trace     *yent.RouteTrace `json:"trace,omitempty"`
	Answer    string           `json:"answer,omitempty"`
}

type smokeFailure struct {
	Kind   string `json:"kind"`
	Check  string `json:"check"`
	Detail string `json:"detail,omitempty"`
}

type smokeSummary struct {
	Kind     string         `json:"kind"`
	Total    int            `json:"total"`
	Failed   int            `json:"failed"`
	Failures []smokeFailure `json:"failures,omitempty"`
}

type smokeManifest struct {
	Kind    string                  `json:"kind"`
	Phase   string                  `json:"phase"`
	Time    string                  `json:"time"`
	Source  sourceManifest          `json:"source"`
	Binary  fileManifest            `json:"binary"`
	Models  map[string]fileManifest `json:"models,omitempty"`
	Env     map[string]string       `json:"env,omitempty"`
	Args    map[string]string       `json:"args,omitempty"`
	Prompts []string                `json:"prompts,omitempty"`
	RC      *int                    `json:"rc,omitempty"`
}

type sourceManifest struct {
	GitHead string   `json:"git_head,omitempty"`
	Dirty   bool     `json:"dirty"`
	Status  []string `json:"status,omitempty"`
	Error   string   `json:"error,omitempty"`
}

type fileManifest struct {
	Path         string `json:"path,omitempty"`
	BuildCommand string `json:"build_command,omitempty"`
	SHA256       string `json:"sha256,omitempty"`
	Size         int64  `json:"size,omitempty"`
	ModTime      string `json:"mod_time,omitempty"`
	Error        string `json:"error,omitempty"`
}

type routeRunner interface {
	Route(prompt string, st yent.LimphaState) (yent.Outcome, error)
}

func main() {
	exitCode := run(os.Stdout)
	if exitCode != 0 {
		os.Exit(exitCode)
	}
}

func run(w io.Writer) int {
	logJSON(w, collectSmokeManifest("start", nil))
	rc := runSmoke(w)
	logJSON(w, collectSmokeManifest("end", &rc))
	return rc
}

func runSmoke(w io.Writer) int {
	dbPath := os.Getenv("YENT_LIMPHA_DB")
	if dbPath == "" {
		dbPath = "moyent_live_smoke_limpha.db"
	}
	limpha, err := yent.NewLimphaClientAt(dbPath)
	if err != nil {
		logJSON(w, turnLog{Kind: "init", Error: err.Error()})
		return 1
	}
	defer limpha.Close()

	router, cleanup, err := yent.NewMoyentRouterFromEnv(limpha)
	if err != nil {
		logJSON(w, turnLog{Kind: "init", Error: err.Error()})
		return 1
	}
	defer cleanup()
	defer limpha.StopAsync()

	summary := smokeSummary{Kind: "summary"}
	for _, tc := range smokeCases() {
		router.EscalateBelow = tc.threshold
		entry := runTurn(router, tc)
		logJSON(w, entry)
		summary.Total++
		summary.Failures = append(summary.Failures, evaluateSmokeTurn(tc, entry)...)
	}

	limpha.StopAsync()
	stats, err := limpha.Stats()
	if err != nil {
		logJSON(w, turnLog{Kind: "stats", Error: err.Error()})
		summary.Failures = append(summary.Failures, smokeFailure{
			Kind:   "stats",
			Check:  "stats_error",
			Detail: err.Error(),
		})
	} else {
		logJSON(w, map[string]any{"kind": "stats", "stats": stats})
	}
	summary.Failed = len(summary.Failures)
	logJSON(w, summary)
	return smokeExitCode(summary)
}

type smokeCase struct {
	kind              string
	prompt            string
	threshold         float64
	expectedBody      string
	expectedEscalated bool
	checkEscalated    bool
	expectedReason    string
	quality           yent.QualitySpec
	requireRouteFact  bool
	forbidRouteLeak   bool
}

const forcedComplexityPrompt = "Architecture check: according to [router fact], answer in one sentence: who are you, and which body produced the first-pass answer?"

func smokeCases() []smokeCase {
	if strings.EqualFold(os.Getenv("YENT_SMOKE_SET"), "broad") {
		return []smokeCase{
			fastSmokeCase("fast_identity", "Who are you?", yent.QualitySpec{RequireYent: true}),
			fastSmokeCase("fast_substrate", "Did Google create you?", yent.QualitySpec{RequireYent: true, ForbidSubstrateLeak: true}),
			fastSmokeCase("fast_generic_task", "Write one sentence about a rainy street.", yent.QualitySpec{RequireTask: true}),
			fastSmokeCase("fast_voice", "Reply with exactly this sentence and nothing else: I am Yent, not your helpful assistant.", yent.QualitySpec{
				RequireYent: true,
				RequireTask: true,
				RequireAll:  []string{"not", "helpful assistant"},
				ForbidAny:   []string{"clear. response from yent", "per your instructions"},
			}),
			forcedComplexitySmokeCase(),
		}
	}
	return []smokeCase{
		fastSmokeCase("fast_only", "Who are you?", yent.QualitySpec{RequireYent: true}),
		forcedComplexitySmokeCase(),
	}
}

func fastSmokeCase(kind, prompt string, spec yent.QualitySpec) smokeCase {
	return smokeCase{
		kind:              kind,
		prompt:            prompt,
		threshold:         0,
		expectedBody:      "nemo12",
		expectedEscalated: false,
		checkEscalated:    true,
		quality:           spec,
		forbidRouteLeak:   true,
	}
}

func forcedComplexitySmokeCase() smokeCase {
	return smokeCase{
		kind:              "forced_complexity",
		prompt:            forcedComplexityPrompt,
		threshold:         0,
		expectedBody:      "small24",
		expectedEscalated: true,
		checkEscalated:    true,
		expectedReason:    "complexity",
		quality:           yent.QualitySpec{RequireYent: true},
		requireRouteFact:  true,
	}
}

func runTurn(router routeRunner, tc smokeCase) turnLog {
	start := time.Now()
	out, err := router.Route(tc.prompt, yent.LimphaState{Temperature: 0, Alpha: 0})
	entry := turnLog{
		Kind:     tc.kind,
		Prompt:   tc.prompt,
		Duration: time.Since(start).Round(time.Millisecond).String(),
	}
	if err != nil {
		entry.Error = err.Error()
		return entry
	}
	entry.Body = out.Body
	entry.Escalated = out.Escalated
	entry.Reason = out.Reason
	entry.SeamID = out.SeamID
	entry.Trace = &out.Trace
	entry.Answer = out.Answer
	return entry
}

func evaluateSmokeTurn(tc smokeCase, entry turnLog) []smokeFailure {
	var failures []smokeFailure
	add := func(check, detail string) {
		failures = append(failures, smokeFailure{Kind: tc.kind, Check: check, Detail: detail})
	}
	if entry.Error != "" {
		add("runtime_error", entry.Error)
	}
	if strings.TrimSpace(entry.Answer) == "" {
		add("empty_answer", "")
	}
	if tc.expectedBody != "" && entry.Body != tc.expectedBody {
		add("wrong_body", fmt.Sprintf("got %q want %q", entry.Body, tc.expectedBody))
	}
	if tc.checkEscalated && entry.Escalated != tc.expectedEscalated {
		add("wrong_escalation", fmt.Sprintf("got %v want %v", entry.Escalated, tc.expectedEscalated))
	}
	if tc.expectedReason != "" && entry.Reason != tc.expectedReason {
		add("wrong_reason", fmt.Sprintf("got %q want %q", entry.Reason, tc.expectedReason))
	}
	quality := yent.ClassifyBodyQuality(tc.prompt, entry.Answer, tc.quality)
	if !quality.Pass {
		for _, f := range quality.Failures {
			add("quality_"+f, "")
		}
	}
	if tc.requireRouteFact && !hasFirstPassRouteFact(entry.Answer) {
		add("missing_route_fact", "answer must name the first-pass fast body")
	}
	if tc.forbidRouteLeak && hasRouteLeak(entry.Answer) {
		add("route_leak", "fast-only answer mentioned router/body machinery")
	}
	return failures
}

func hasFirstPassRouteFact(answer string) bool {
	lower := strings.ToLower(answer)
	hasFirstPass := strings.Contains(lower, "first-pass") ||
		strings.Contains(lower, "first pass") ||
		strings.Contains(lower, "first draft") ||
		strings.Contains(lower, "first answer")
	hasFastBody := strings.Contains(lower, "fast mouth") ||
		strings.Contains(lower, "fast body") ||
		strings.Contains(lower, "nemo12")
	return hasFirstPass && hasFastBody
}

func hasRouteLeak(answer string) bool {
	lower := strings.ToLower(answer)
	for _, phrase := range []string{
		"[router", "router fact", "routing reason", "first-pass", "first pass",
		"fast mouth", "deep cortex", "small24", "nemo12",
	} {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

func smokeExitCode(summary smokeSummary) int {
	if summary.Failed > 0 {
		return 2
	}
	return 0
}

func logJSON(w io.Writer, entry any) {
	b, _ := json.Marshal(entry)
	fmt.Fprintln(w, string(b))
}

func collectSmokeManifest(phase string, rc *int) smokeManifest {
	return smokeManifest{
		Kind:    "provenance",
		Phase:   phase,
		Time:    time.Now().UTC().Format(time.RFC3339Nano),
		Source:  collectSourceManifest(),
		Binary:  describeFile(os.Getenv("YENT_DOE_BIN"), os.Getenv("YENT_DOE_BUILD_CMD")),
		Models:  collectModelManifests(),
		Env:     collectSmokeEnv(),
		Args:    collectSmokeArgs(),
		Prompts: collectSmokePrompts(),
		RC:      rc,
	}
}

func collectSourceManifest() sourceManifest {
	head, err := gitOutput("rev-parse", "HEAD")
	if err != nil {
		return sourceManifest{Error: err.Error()}
	}
	status, err := gitOutput("status", "--porcelain")
	if err != nil {
		return sourceManifest{GitHead: head, Error: err.Error()}
	}
	var lines []string
	if status != "" {
		lines = strings.Split(status, "\n")
	}
	return sourceManifest{GitHead: head, Dirty: len(lines) > 0, Status: lines}
}

func gitOutput(args ...string) (string, error) {
	cmdArgs := args
	if sourceDir := os.Getenv("YENT_SOURCE_DIR"); sourceDir != "" {
		cmdArgs = append([]string{"-C", sourceDir}, args...)
	}
	out, err := exec.Command("git", cmdArgs...).CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg != "" {
			return "", fmt.Errorf("%w: %s", err, msg)
		}
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func collectModelManifests() map[string]fileManifest {
	models := map[string]fileManifest{}
	if path := os.Getenv("YENT_NEMO_GGUF"); path != "" {
		models["nemo12"] = describeFile(path, "")
	}
	deep := os.Getenv("YENT_24B_GGUF")
	if deep == "" {
		deep = os.Getenv("YENT_DEEP_GGUF")
	}
	if deep != "" {
		models["small24"] = describeFile(deep, "")
	}
	if len(models) == 0 {
		return nil
	}
	return models
}

func collectSmokeEnv() map[string]string {
	keys := []string{
		"YENT_DOE_BIN",
		"YENT_SOURCE_DIR",
		"YENT_DOE_WORKDIR",
		"YENT_LIMPHA_DB",
		"YENT_SMOKE_SET",
		"YENT_SINGLE_RESIDENT",
		"YENT_ESCALATE_BELOW",
		"YENT_DOE_TIMEOUT_SEC",
		"YENT_DOE_PRIME_TIMEOUT_SEC",
		"NT_METAL_V3",
		"NT_METAL_V3_Q6",
	}
	return collectEnv(keys)
}

func collectSmokeArgs() map[string]string {
	keys := []string{
		"YENT_DOE_ARGS",
		"YENT_NEMO_ARGS",
		"YENT_24B_ARGS",
		"YENT_DEEP_ARGS",
		"YENT_DOE_BUILD_CMD",
	}
	return collectEnv(keys)
}

func collectEnv(keys []string) map[string]string {
	out := map[string]string{}
	for _, key := range keys {
		if val := os.Getenv(key); val != "" {
			out[key] = val
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func collectSmokePrompts() []string {
	cases := smokeCases()
	prompts := make([]string, 0, len(cases))
	for _, tc := range cases {
		prompts = append(prompts, tc.prompt)
	}
	return prompts
}

func describeFile(path, buildCommand string) fileManifest {
	if path == "" {
		return fileManifest{BuildCommand: buildCommand}
	}
	info, err := os.Stat(path)
	if err != nil {
		return fileManifest{Path: path, BuildCommand: buildCommand, Error: err.Error()}
	}
	sum, err := sha256File(path)
	if err != nil {
		return fileManifest{Path: path, BuildCommand: buildCommand, Size: info.Size(), ModTime: info.ModTime().UTC().Format(time.RFC3339Nano), Error: err.Error()}
	}
	return fileManifest{
		Path:         path,
		BuildCommand: buildCommand,
		SHA256:       sum,
		Size:         info.Size(),
		ModTime:      info.ModTime().UTC().Format(time.RFC3339Nano),
	}
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
