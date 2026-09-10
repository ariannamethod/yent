package yent

// body_router.go — moyent's two-body router: one organism, two swappable Mistral
// bodies sharing one limpha brain. The fast body (Nemo-12B) speaks by default; the
// router escalates to the deep body (Small-24B) when prompt complexity or the fast
// body's own uncertainty demands it. NEVER both resident at once — a Body is an
// interface the concrete persistent doe-daemon driver satisfies; the router only
// orchestrates and logs, so it is testable without a model.
//
// Every turn is logged into the shared limpha brain: plain turns -> conversations;
// dual-pass (escalated) turns also write a seam — the internal dialogue between the
// two bodies (a_claim/b_claim) plus the divergence metrics (agreement/tension/winner)
// the deep body scored. Supergamma later grows from that seam log.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"unicode/utf8"
)

// Body is one inference body behind the router. The production implementation
// is a persistent doe daemon (model resident, swapped on demand). Tests use fakes.
type Body interface {
	// Name identifies the body in logs and seams, e.g. "nemo12" or "small24".
	Name() string
	// Generate answers prompt. ctx carries cross-body context on escalation: the fast
	// body's primer, or the deep body's primer plus resonant memory references and
	// the routing reason on escalation.
	Generate(prompt, ctx string) (BodyResult, error)
}

// OptionBody accepts sampler controls for one generation. Keeping this
// capability optional preserves small/fake bodies while allowing the live DOE
// seam to make interface controls truthful.
type OptionBody interface {
	Body
	GenerateWithOptions(prompt, ctx string, opts GenerationOptions) (BodyResult, error)
}

// ClosableBody is implemented by resident process-backed bodies. The router uses
// it to enforce the one-body-resident discipline without knowing about doe.
type ClosableBody interface {
	Body
	Close() error
}

// BodyResult is one body's output for a turn.
type BodyResult struct {
	Answer string
	// Confidence is the body's self-signal in [0,1]. The router escalates when the
	// fast body reports low confidence (its entropy / top-logit-margin proxy).
	Confidence float64
	// ExecutionPath names the concrete runtime path that produced this answer.
	// Process-backed bodies use it for receipts such as "doe_resident" or "doe_once".
	ExecutionPath string
	// Diagnostics carries bounded runtime diagnostics from the body driver. The
	// router records it in traces, but does not feed it back into model prompts.
	Diagnostics []string
	// Timing exposes where latency was spent without feeding telemetry back into
	// the model prompt.
	Timing BodyTiming
	// Verdict is the deep body's reflection on the fast body's trace — set only when
	// the deep body ran with that trace in ctx. The deep body scores agreement and
	// tension and names the winner; the router copies it into the seam.
	Verdict *Verdict
}

// Verdict is the deep body's scoring of the divergence with the fast body.
type Verdict struct {
	Agreement float64 // 0..1
	Tension   float64 // 0..1
	Winner    string  // body name whose answer is used
}

// Router orchestrates the fast body and an optional deep body over one shared
// limpha brain.
type Router struct {
	fast   Body          // default mouth, e.g. nemo12
	deep   Body          // escalation cortex, e.g. small24
	limpha *LimphaClient // shared brain; may be nil (logging disabled)
	// FastPrimer and DeepPrimer are compact role/organism anchors sent as
	// private body context. They are not identity substitutes for weights; they
	// keep the two bodies differentiated while binding them to one Yent.
	FastPrimer string
	DeepPrimer string
	// EscalateBelow: if the fast body's Confidence is below this, escalate. [0,1].
	EscalateBelow float64
	// MemoryRefs controls how many limpha search hits the deep body sees.
	MemoryRefs int
	// StateRefs controls how many AMK-state-neighbor memories the deep body sees.
	StateRefs int
	// AsyncMemory queues conversation+seam writes through limpha's background
	// circulator. Default false keeps deterministic tests and one-shot tools sync.
	AsyncMemory bool
	// SingleResident closes the inactive resident body before switching. This keeps
	// the two-body organism honest on 24GB-class hosts: one mouth resident per turn
	// segment, not two loaded weights pretending RAM is infinite.
	SingleResident bool
}

// NewRouter wires the fast body and an optional deep body to one limpha brain.
// EscalateBelow defaults to 0.5.
func NewRouter(fast, deep Body, limpha *LimphaClient) *Router {
	// An optional concrete pointer passed through the Body interface can carry a
	// nil value while the interface itself compares non-nil. Normalize that
	// boundary here so a Nemo-only organism never tries to call methods on a
	// missing deep body.
	if bodyIsNil(fast) {
		fast = nil
	}
	if bodyIsNil(deep) {
		deep = nil
	}
	return &Router{
		fast:           fast,
		deep:           deep,
		limpha:         limpha,
		FastPrimer:     DefaultFastPrimer,
		DeepPrimer:     DefaultDeepPrimer,
		EscalateBelow:  0.5,
		MemoryRefs:     3,
		StateRefs:      2,
		SingleResident: true,
	}
}

func bodyIsNil(body Body) bool {
	if body == nil {
		return true
	}
	v := reflect.ValueOf(body)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

// The fast body carries Yent's voice in its weights. The live route adds only
// chronological dialogue structure when history exists; no default persona
// instruction is injected into a clean turn.
const DefaultFastPrimer = ""

const DefaultDeepPrimer = "Yent: use context facts as private evidence and answer the human directly. If the human asks how this answer was produced, use the router fact literally. Do not copy the first-pass draft's role."

// Outcome is the router's decision for a turn (returned to the caller and tests).
type Outcome struct {
	Answer    string
	Body      string // which body's answer was used
	Escalated bool
	Reason    string // why escalated (empty if not)
	SeamID    int64  // >0 if a seam was written
	Trace     RouteTrace
}

// RouteTrace is the machine-readable circulation receipt for one routed turn.
// It mirrors the human seam text without making downstream systems parse prose.
type RouteTrace struct {
	Kind                string           `json:"kind"`
	FastBody            string           `json:"fast_body"`
	DeepBody            string           `json:"deep_body,omitempty"`
	Winner              string           `json:"winner"`
	Escalated           bool             `json:"escalated"`
	Reason              string           `json:"reason,omitempty"`
	FastConfidence      float64          `json:"fast_confidence"`
	FastConfidenceValid bool             `json:"fast_confidence_valid"`
	FastExecutionPath   string           `json:"fast_execution_path,omitempty"`
	DeepExecutionPath   string           `json:"deep_execution_path,omitempty"`
	FastDiagnostics     []string         `json:"fast_diagnostics,omitempty"`
	DeepDiagnostics     []string         `json:"deep_diagnostics,omitempty"`
	FastTiming          BodyTiming       `json:"fast_timing,omitempty"`
	DeepTiming          BodyTiming       `json:"deep_timing,omitempty"`
	MemoryStatus        string           `json:"memory_status,omitempty"`
	MemoryError         string           `json:"memory_error,omitempty"`
	MemoryConversation  int64            `json:"memory_conversation_id,omitempty"`
	MemorySeam          int64            `json:"memory_seam_id,omitempty"`
	Agreement           float64          `json:"agreement,omitempty"`
	Tension             float64          `json:"tension,omitempty"`
	Complexity          PromptComplexity `json:"complexity"`
	State               LimphaState      `json:"state"`
	MemoryRefs          int              `json:"memory_refs,omitempty"`
	StateRefs           int              `json:"state_refs,omitempty"`
	SeamRefs            int              `json:"seam_refs,omitempty"`
	DeepError           string           `json:"deep_error,omitempty"`
	InnerContext        bool             `json:"inner_context,omitempty"`
}

type routeContextBundle struct {
	Text       string
	MemoryRefs int
	StateRefs  int
	SeamRefs   int
}

// escalationReason decides whether the fast attempt must escalate to the deep body.
// Two falsifiable signals (design canon): the fast body's own low confidence, and a
// prompt-complexity heuristic. Returns "" when the fast body may answer alone.
func escalationReason(prompt string, fast BodyResult, threshold float64) string {
	return escalationReasonWithComplexity(fast, threshold, AnalyzePromptComplexity(prompt))
}

func escalationReasonWithComplexity(fast BodyResult, threshold float64, complexity PromptComplexity) string {
	if !isFinite01(fast.Confidence) || fast.Confidence < clamp01(threshold) {
		return "low_confidence"
	}
	if complexity.ShouldEscalate() {
		return "complexity"
	}
	return ""
}

// promptIsComplex is a cheap, falsifiable v1 heuristic: long or code/architecture/
// planning prompts route to the deep body. The real entropy/margin signal lives in
// the fast body's Confidence; this only catches obvious depth from the prompt text.
func promptIsComplex(prompt string) bool {
	return AnalyzePromptComplexity(prompt).ShouldEscalate()
}

// Route runs one turn without an additional private inner-world context.
func (r *Router) Route(prompt string, st LimphaState) (Outcome, error) {
	return r.RouteWithInnerContextOptions(prompt, st, "", GenerationOptions{})
}

// RouteWithInnerContext runs one outward turn with optional private context.
// innerContext is body-private pressure: it shapes the
// answer but is never substituted for the human prompt or stored as that prompt.
// A nil deep body is an intentional fast-only organism, not a routing failure;
// when a deep body is present the usual confidence/complexity escalation applies.
func (r *Router) RouteWithInnerContext(prompt string, st LimphaState, innerContext string) (Outcome, error) {
	return r.RouteWithInnerContextOptions(prompt, st, innerContext, GenerationOptions{})
}

// RouteWithInnerContextOptions is the live-turn seam: private continuity and
// one-turn sampler controls reach the selected body without changing the Body
// contract for implementations that do not support runtime options.
func (r *Router) RouteWithInnerContextOptions(prompt string, st LimphaState, innerContext string, opts GenerationOptions) (Outcome, error) {
	if r == nil || r.fast == nil {
		return Outcome{}, errors.New("router requires a fast body")
	}
	innerContext = strings.TrimSpace(innerContext)
	if err := r.prepareBody(r.fast); err != nil {
		return Outcome{}, err
	}
	fast, err := generateBody(r.fast, prompt, r.fastContext(innerContext), opts)
	if err != nil {
		return Outcome{}, err
	}
	complexity := AnalyzePromptComplexity(prompt)
	trace := r.newRouteTrace(fast, complexity, st)
	trace.InnerContext = innerContext != ""
	reason := ""
	if r.deep != nil {
		reason = escalationReasonWithComplexity(fast, r.EscalateBelow, complexity)
	}
	if reason == "" {
		// single-body turn: the fast body answers alone.
		trace.applyMemoryReceipt(r.storeTurn(prompt, fast.Answer, st, nil, &trace))
		return Outcome{Answer: fast.Answer, Body: r.fast.Name(), Escalated: false, Trace: trace}, nil
	}

	// dual-pass: the deep body reflects on the fast trace + memory refs + reason.
	trace.Escalated = true
	trace.Reason = reason
	bundle := r.buildEscalationContext(prompt, fast, reason, st, complexity)
	if innerContext != "" {
		bundle.Text = insertInnerContextAfterDeepPrimer(bundle.Text, innerContext)
	}
	trace.MemoryRefs = bundle.MemoryRefs
	trace.StateRefs = bundle.StateRefs
	trace.SeamRefs = bundle.SeamRefs
	if err := r.prepareBody(r.deep); err != nil {
		return Outcome{}, err
	}
	deep, err := generateBody(r.deep, prompt, bundle.Text, opts)
	if err != nil {
		// deep failed — keep the fast answer rather than dropping the turn.
		trace.Winner = r.fast.Name()
		trace.DeepError = err.Error()
		trace.applyMemoryReceipt(r.storeTurn(prompt, fast.Answer, st, nil, &trace))
		return Outcome{Answer: fast.Answer, Body: r.fast.Name(), Escalated: true, Reason: reason, Trace: trace}, nil
	}
	trace.DeepExecutionPath = deep.ExecutionPath
	trace.DeepDiagnostics = cloneDiagnostics(deep.Diagnostics)
	trace.DeepTiming = deep.Timing

	winner := r.deep.Name()
	agreement, tension := 0.0, 0.0
	if deep.Verdict != nil {
		agreement, tension = clamp01(deep.Verdict.Agreement), clamp01(deep.Verdict.Tension)
		if deep.Verdict.Winner == r.fast.Name() || deep.Verdict.Winner == r.deep.Name() {
			winner = deep.Verdict.Winner
		}
	}
	answer := deep.Answer
	if winner == r.fast.Name() { // deep conceded — the fast answer stands
		answer = fast.Answer
	}
	trace.Winner = winner
	trace.Agreement = agreement
	trace.Tension = tension
	seam := &Seam{
		BodyA: r.fast.Name(), BodyB: r.deep.Name(),
		Prompt: prompt, AClaim: fast.Answer, BClaim: deep.Answer,
		Agreement: agreement, Tension: tension, Winner: winner, Reason: reason,
	}
	receipt := r.storeTurn(prompt, answer, st, seam, &trace)
	trace.applyMemoryReceipt(receipt)
	return Outcome{Answer: answer, Body: winner, Escalated: true, Reason: reason, SeamID: receipt.SeamID, Trace: trace}, nil
}

func generateBody(body Body, prompt, ctx string, opts GenerationOptions) (BodyResult, error) {
	if optionBody, ok := body.(OptionBody); ok {
		return optionBody.GenerateWithOptions(prompt, ctx, opts)
	}
	return body.Generate(prompt, ctx)
}

func insertInnerContextAfterDeepPrimer(routeContext, innerContext string) string {
	innerContext = strings.TrimSpace(innerContext)
	if innerContext == "" {
		return routeContext
	}
	private := "[private inner context]: " + innerContext + "\n"
	if strings.HasPrefix(routeContext, "[deep primer]:") {
		if lineEnd := strings.IndexByte(routeContext, '\n'); lineEnd >= 0 {
			return routeContext[:lineEnd+1] + private + routeContext[lineEnd+1:]
		}
	}
	return private + routeContext
}

// escalationContext is what the deep body receives: the fast body's trace, the routing
// reason, and any resonant memory references the shared brain holds for this prompt.
func (r *Router) escalationContext(prompt string, fast BodyResult, reason string, st LimphaState, complexity PromptComplexity) string {
	return r.buildEscalationContext(prompt, fast, reason, st, complexity).Text
}

func (r *Router) buildEscalationContext(prompt string, fast BodyResult, reason string, st LimphaState, complexity PromptComplexity) routeContextBundle {
	var bundle routeContextBundle
	var b strings.Builder
	if primer := strings.TrimSpace(r.DeepPrimer); primer != "" {
		b.WriteString("[deep primer]: " + primer + "\n")
	}
	fastLabel, deepLabel := bodyPromptLabel(r.fast.Name()), bodyPromptLabel(r.deep.Name())
	b.WriteString("[router fact]: " + fastLabel + " produced the first-pass answer; " + deepLabel + " is the escalation/final-pass body.\n")
	b.WriteString("[route answer labels]: first-pass body = " + fastLabel + "; final-pass body = " + deepLabel + ".\n")
	b.WriteString("[current response role]: " + deepLabel + "\n")
	b.WriteString("[routing reason: " + reason + "]\n")
	b.WriteString("[prompt complexity]: " + complexity.Summary() + "\n")
	b.WriteString("[field state]: " + formatLimphaState(st) + "\n")
	b.WriteString("[first-pass draft from " + fastLabel + "; not current role]: " + fast.Answer + "\n")
	if r.limpha != nil {
		if refs, _ := r.limpha.Search(prompt, positiveOrDefault(r.MemoryRefs, 3)); len(refs) > 0 {
			bundle.MemoryRefs = len(refs)
			b.WriteString("[memory refs]:\n")
			for _, m := range refs {
				if p, ok := m["prompt"].(string); ok {
					line := "- p: " + compactLine(p, 180)
					if resp, ok := m["response"].(string); ok && strings.TrimSpace(resp) != "" {
						line += " | r: " + compactLine(resp, 220)
					}
					b.WriteString(line + "\n")
				}
			}
		}
		if refs, _ := r.searchStateNeighbors(st); len(refs) > 0 {
			bundle.StateRefs = len(refs)
			b.WriteString("[state-neighbor refs]:\n")
			for _, m := range refs {
				p, _ := m["prompt"].(string)
				dist, _ := m["distance"].(float64)
				b.WriteString(fmt.Sprintf("- distance=%.3f p: %s\n", dist, compactLine(p, 160)))
			}
		}
		if seams, _ := r.limpha.RecentSeams(2); len(seams) > 0 {
			bundle.SeamRefs = len(seams)
			b.WriteString("[recent internal seams]:\n")
			for _, s := range seams {
				p, _ := s["prompt"].(string)
				winner, _ := s["winner"].(string)
				reason, _ := s["reason"].(string)
				tension, _ := s["tension"].(float64)
				b.WriteString(fmt.Sprintf("- winner=%s reason=%s tension=%.2f p: %s\n",
					bodyPromptLabel(winner), reason, tension, compactLine(p, 140)))
			}
		}
	}
	bundle.Text = b.String()
	return bundle
}

func bodyPromptLabel(name string) string {
	switch strings.TrimSpace(name) {
	case "nemo12":
		return "fast mouth"
	case "small24":
		return "deep cortex"
	default:
		if strings.TrimSpace(name) == "" {
			return "body"
		}
		return strings.TrimSpace(name)
	}
}

func (r *Router) newRouteTrace(fast BodyResult, complexity PromptComplexity, st LimphaState) RouteTrace {
	validConfidence := isFinite01(fast.Confidence)
	confidence := 0.0
	if validConfidence {
		confidence = fast.Confidence
	}
	trace := RouteTrace{
		Kind:                "route_context",
		FastBody:            r.fast.Name(),
		Winner:              r.fast.Name(),
		FastConfidence:      confidence,
		FastConfidenceValid: validConfidence,
		FastExecutionPath:   fast.ExecutionPath,
		FastDiagnostics:     cloneDiagnostics(fast.Diagnostics),
		FastTiming:          fast.Timing,
		Complexity:          complexity,
		State:               st,
	}
	if r.deep != nil {
		trace.DeepBody = r.deep.Name()
	}
	return trace
}

func (r *Router) fastContext(innerContext string) string {
	if r == nil {
		return ""
	}
	primer := strings.TrimSpace(r.FastPrimer)
	innerContext = strings.TrimSpace(innerContext)
	if primer == "" {
		return innerContext
	}
	if innerContext == "" {
		return primer
	}
	return primer + "\n" + innerContext
}

func (r *Router) searchStateNeighbors(st LimphaState) ([]map[string]interface{}, error) {
	if r == nil || r.limpha == nil || !limphaStateHasSignal(st) {
		return nil, nil
	}
	return r.limpha.SearchByState(st, positiveOrDefault(r.StateRefs, 2), 0.35)
}

func (r *Router) prepareBody(target Body) error {
	if r == nil || !r.SingleResident || bodyIsNil(target) {
		return nil
	}
	targetName := target.Name()
	for _, body := range []Body{r.fast, r.deep} {
		if bodyIsNil(body) || body.Name() == targetName {
			continue
		}
		closer, ok := body.(ClosableBody)
		if !ok {
			continue
		}
		if err := closer.Close(); err != nil {
			return fmt.Errorf("close inactive body %s before %s: %w", body.Name(), targetName, err)
		}
	}
	return nil
}

// storeConversation persists a turn into the shared brain; 0 if memory is off.
func (r *Router) storeConversation(prompt, answer string, st LimphaState) (int64, error) {
	if r.limpha == nil || !r.limpha.connected {
		return 0, nil
	}
	return r.limpha.StoreTurn(prompt, answer, st)
}

// storeTurn records one selected answer and, for dual-pass turns, its seam. In
// async mode the link is preserved inside the worker; the immediate seam id is
// unavailable and returns 0.
func (r *Router) storeTurn(prompt, answer string, st LimphaState, seam *Seam, trace *RouteTrace) memoryWriteReceipt {
	if r.limpha == nil || !r.limpha.connected {
		return memoryWriteReceipt{Status: "disabled"}
	}
	if r.AsyncMemory && r.limpha.EnqueueTurn(prompt, answer, st, seam) {
		return memoryWriteReceipt{Status: "queued"}
	}
	convID, err := r.storeConversation(prompt, answer, st)
	if err != nil {
		return memoryWriteReceipt{Status: "failed", Error: err}
	}
	if seam == nil {
		return memoryWriteReceipt{Status: "stored", ConversationID: convID}
	}
	seam.ConversationID = convID
	if trace != nil {
		traceWithConversation := *trace
		traceWithConversation.MemoryStatus = "conversation_stored"
		traceWithConversation.MemoryConversation = convID
		seam.MemoryDelta = formatMemoryDelta(traceWithConversation)
	}
	id, err := r.limpha.StoreSeam(*seam)
	if err != nil {
		return memoryWriteReceipt{Status: "failed", Error: err, ConversationID: convID}
	}
	return memoryWriteReceipt{Status: "stored", ConversationID: convID, SeamID: id}
}

type memoryWriteReceipt struct {
	Status         string
	Error          error
	ConversationID int64
	SeamID         int64
}

func (t *RouteTrace) applyMemoryReceipt(receipt memoryWriteReceipt) {
	if t == nil {
		return
	}
	t.MemoryStatus = receipt.Status
	t.MemoryConversation = receipt.ConversationID
	t.MemorySeam = receipt.SeamID
	if receipt.Error != nil {
		t.MemoryError = compactLine(receipt.Error.Error(), 240)
	}
}

func isFinite01(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0 && v <= 1
}

func clamp01(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func positiveOrDefault(v, fallback int) int {
	if v > 0 {
		return v
	}
	return fallback
}

func cloneDiagnostics(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, len(in))
	copy(out, in)
	return out
}

func compactLine(s string, maxRunes int) string {
	s = strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
	if maxRunes <= 0 || utf8.RuneCountInString(s) <= maxRunes {
		return s
	}
	rs := []rune(s)
	if maxRunes <= 1 {
		return string(rs[:maxRunes])
	}
	return string(rs[:maxRunes-1]) + "..."
}

func formatLimphaState(st LimphaState) string {
	return fmt.Sprintf("temp=%.2f destiny=%.2f pain=%.2f tension=%.2f debt=%.2f velocity=%d alpha=%.2f",
		st.Temperature, st.Destiny, st.Pain, st.Tension, st.Debt, st.Velocity, st.Alpha)
}

func formatMemoryDelta(trace RouteTrace) string {
	b, err := json.Marshal(trace)
	if err != nil {
		return "route_context " + formatLimphaState(trace.State) + " complexity=" + trace.Complexity.Summary()
	}
	return string(b)
}

func limphaStateHasSignal(st LimphaState) bool {
	return st.Temperature != 0 || st.Destiny != 0 || st.Pain != 0 ||
		st.Tension != 0 || st.Debt != 0 || st.Velocity != 0 || st.Alpha != 0
}
