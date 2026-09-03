package yent

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

const (
	// SartreSeamReason is the limpha seam class for SARTRE utility perception.
	// SARTRE is pressure and evidence, not dialogue to continue.
	SartreSeamReason = "sartre_perception"

	maxSartreEventsPerPacket = 64
)

// SartreEvent is one bounded utility receipt from the SARTRE body organ.
// It intentionally carries metadata, not file contents.
type SartreEvent struct {
	ID                string  `json:"id,omitempty"`
	Phase             string  `json:"phase,omitempty"`
	Outcome           string  `json:"outcome,omitempty"`
	Utility           string  `json:"util"`
	Kind              string  `json:"kind,omitempty"`
	Path              string  `json:"path,omitempty"`
	Tag               string  `json:"tag,omitempty"` // legacy context_processor readout; new receipts use Resonance
	Resonance         float64 `json:"resonance,omitempty"`
	Relevance         float64 `json:"relevance,omitempty"`
	Pulse             float64 `json:"pulse,omitempty"`
	Reduced           int     `json:"reduced,omitempty"`
	Recognized        int     `json:"recognized,omitempty"`
	Timestamp         int64   `json:"ts,omitempty"`
	RootID            string  `json:"root_id,omitempty"`
	Breath            int     `json:"breath,omitempty"`
	CadenceMS         int64   `json:"cadence_ms,omitempty"`
	RefractoryBreaths int     `json:"refractory_breaths,omitempty"`
	CooldownBreaths   int     `json:"cooldown_breaths,omitempty"`
	EffectCount       int     `json:"effect_count,omitempty"`
	BytesCaptured     int     `json:"bytes_captured,omitempty"`
	BytesLimit        int     `json:"bytes_limit,omitempty"`
	Attempts          int     `json:"attempts,omitempty"`
	FailureOutcome    string  `json:"failure_outcome,omitempty"`
}

// SartreReceipt is the machine-readable memory_delta written into limpha.
type SartreReceipt struct {
	Kind              string         `json:"kind"`
	EventCount        int            `json:"event_count"`
	Changed           int            `json:"changed"`
	ReadmeChanged     bool           `json:"readme_changed,omitempty"`
	MaxResonance      float64        `json:"max_resonance,omitempty"`
	MaxRelevance      float64        `json:"max_relevance,omitempty"`
	MaxPulse          float64        `json:"max_pulse,omitempty"`
	FramingEventCount int            `json:"framing_event_count,omitempty"`
	MaxReduced        int            `json:"max_reduced,omitempty"`
	MaxRecognized     int            `json:"max_recognized,omitempty"`
	OutcomeCounts     map[string]int `json:"outcome_counts,omitempty"`
	Trace             []string       `json:"trace"`
	Events            []SartreEvent  `json:"events,omitempty"`
}

// ParseSartreEventsJSONL reads SARTRE utility stdout. Non-JSON status lines from
// the slot wrapper are ignored; malformed JSON event lines are skipped. The result
// is capped so a noisy utility cannot become a prompt wall through limpha.
func ParseSartreEventsJSONL(jsonl string) []SartreEvent {
	lines := strings.Split(jsonl, "\n")
	events := make([]SartreEvent, 0, minInt(len(lines), maxSartreEventsPerPacket))
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" || !strings.HasPrefix(line, "{") {
			continue
		}
		var ev SartreEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		ev = normalizeSartreEvent(ev)
		if ev.Utility == "" {
			continue
		}
		events = append(events, ev)
		if len(events) >= maxSartreEventsPerPacket {
			break
		}
	}
	return events
}

// StoreSartreEvents persists one SARTRE perception packet into limpha. It writes
// both a conversation row for search and a seam row for typed downstream recall.
func (c *LimphaClient) StoreSartreEvents(events []SartreEvent, st LimphaState) (int64, error) {
	id, _, err := c.StoreNewSartreEvents(events, st)
	return id, err
}

// StoreNewSartreEvents persists only stable SARTRE events not already recorded in
// limpha. Retried will receipts reuse the same causal ids; those retries must be
// acknowledged without adding duplicate seams or moving the live field twice.
func (c *LimphaClient) StoreNewSartreEvents(events []SartreEvent, st LimphaState) (int64, []SartreEvent, error) {
	if c == nil || !c.connected || len(events) == 0 {
		return 0, nil, nil
	}
	tx, err := c.db.Begin()
	if err != nil {
		c.recordMemoryFailure("conversation", err)
		return 0, nil, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = tx.Rollback()
		}
	}()
	accepted := make([]SartreEvent, 0, len(events))
	now := nowSeconds()
	for _, ev := range events {
		ev = normalizeSartreEvent(ev)
		if ev.Utility == "" {
			continue
		}
		if key, ok := sartreEventKey(ev); ok {
			res, err := tx.Exec(
				`INSERT OR IGNORE INTO sartre_event_ids
				 (event_key, event_id, phase, utility, recorded_at)
				 VALUES (?, ?, ?, ?, ?)`,
				key, ev.ID, ev.Phase, ev.Utility, now)
			if err != nil {
				c.recordMemoryFailure("conversation", err)
				return 0, nil, err
			}
			n, err := res.RowsAffected()
			if err != nil {
				c.recordMemoryFailure("conversation", err)
				return 0, nil, err
			}
			if n == 0 {
				continue
			}
		}
		accepted = append(accepted, ev)
	}
	receipt := BuildSartreReceipt(accepted)
	if len(receipt.Trace) == 0 {
		if err := tx.Commit(); err != nil {
			c.recordMemoryFailure("sartre_event_ids", err)
			return 0, nil, err
		}
		committed = true
		return 0, accepted, nil
	}
	response := strings.Join(receipt.Trace, "\n")
	prompt := "[sartre/perception] utility receipts"
	convID, err := c.storeSartreConversationTx(tx, prompt, response, st, now)
	if err != nil {
		return 0, nil, err
	}
	delta, _ := json.Marshal(receipt)
	seamID, err := c.storeSartreSeamTx(tx, Seam{
		ConversationID: convID,
		BodyA:          "sartre",
		BodyB:          "limpha",
		Prompt:         prompt,
		AClaim:         response,
		BClaim:         "SARTRE perception: " + compactLine(strings.Join(receipt.Trace, " | "), 260),
		Agreement:      clamp01(maxFloat(receipt.MaxResonance, receipt.MaxRelevance)),
		Tension:        sartreTension(receipt),
		Winner:         "limpha",
		Reason:         SartreSeamReason,
		MemoryDelta:    string(delta),
	})
	if err != nil {
		return 0, nil, err
	}
	if err := tx.Commit(); err != nil {
		c.recordMemoryFailure("seam", err)
		return 0, nil, err
	}
	committed = true
	return seamID, accepted, nil
}

// FilterNewSartreEvents returns the events that do not already have a stable
// SARTRE key in limpha, without reserving or storing that key. The live field
// consumer uses this as a preflight so limpha dedupe cannot be published before
// the field has actually accepted the pressure.
func (c *LimphaClient) FilterNewSartreEvents(events []SartreEvent) ([]SartreEvent, error) {
	if c == nil || !c.connected || len(events) == 0 {
		return nil, nil
	}
	out := make([]SartreEvent, 0, len(events))
	for _, ev := range events {
		ev = normalizeSartreEvent(ev)
		if ev.Utility == "" {
			continue
		}
		key, ok := sartreEventKey(ev)
		if !ok {
			out = append(out, ev)
			continue
		}
		var exists int
		err := c.db.QueryRow(`SELECT 1 FROM sartre_event_ids WHERE event_key = ? LIMIT 1`, key).Scan(&exists)
		if err == nil {
			continue
		}
		if err != sql.ErrNoRows {
			c.recordMemoryFailure("sartre_event_ids", err)
			return nil, err
		}
		out = append(out, ev)
	}
	return out, nil
}

func (c *LimphaClient) storeSartreConversationTx(tx *sql.Tx, prompt, response string, st LimphaState, now float64) (int64, error) {
	quality := computeQuality(prompt, response)
	res, err := tx.Exec(
		`INSERT INTO conversations
		 (timestamp, session_id, prompt, response,
		  temperature, destiny, pain, tension, debt, velocity, alpha, quality)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		now, c.sessionID, prompt, response,
		st.Temperature, st.Destiny, st.Pain, st.Tension, st.Debt, st.Velocity, st.Alpha, quality)
	if err != nil {
		c.recordMemoryFailure("conversation", err)
		return 0, err
	}
	id, _ := res.LastInsertId()
	if _, err := tx.Exec(
		`UPDATE sessions SET last_active = ?, turn_count = turn_count + 1,
		   avg_quality = (avg_quality * turn_count + ?) / (turn_count + 1)
		 WHERE session_id = ?`, now, quality, c.sessionID); err != nil {
		c.recordMemoryFailure("conversation", err)
		return 0, err
	}
	return id, nil
}

func (c *LimphaClient) storeSartreSeamTx(tx *sql.Tx, s Seam) (int64, error) {
	var convID interface{}
	if s.ConversationID > 0 {
		convID = s.ConversationID
	}
	res, err := tx.Exec(
		`INSERT INTO seams
		 (timestamp, session_id, conversation_id, body_a, body_b, prompt,
		  a_claim, b_claim, agreement, tension, winner, reason, memory_delta)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		nowSeconds(), c.sessionID, convID, s.BodyA, s.BodyB, s.Prompt,
		s.AClaim, s.BClaim, s.Agreement, s.Tension, s.Winner, s.Reason, s.MemoryDelta)
	if err != nil {
		c.recordMemoryFailure("seam", err)
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func sartreEventKey(ev SartreEvent) (string, bool) {
	ev = normalizeSartreEvent(ev)
	if ev.ID == "" {
		return "", false
	}
	parts := []string{
		ev.RootID,
		ev.ID,
		ev.Phase,
		ev.Outcome,
		ev.Utility,
		ev.Kind,
		ev.Path,
		ev.Tag,
		fmt.Sprintf("%.6f", ev.Resonance),
		fmt.Sprintf("%.6f", ev.Relevance),
		fmt.Sprintf("%.6f", ev.Pulse),
		fmt.Sprintf("%d", ev.Reduced),
		fmt.Sprintf("%d", ev.Recognized),
		fmt.Sprintf("%d", ev.EffectCount),
		fmt.Sprintf("%d", ev.BytesCaptured),
		fmt.Sprintf("%d", ev.BytesLimit),
		fmt.Sprintf("%d", ev.Attempts),
		ev.FailureOutcome,
	}
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x00")))
	return hex.EncodeToString(sum[:]), true
}

// BuildSartreReceipt summarises utility events into compact pressure traces.
func BuildSartreReceipt(events []SartreEvent) SartreReceipt {
	receipt := SartreReceipt{Kind: SartreSeamReason}
	seen := make(map[string]bool)
	for _, ev := range events {
		ev = normalizeSartreEvent(ev)
		if ev.Utility == "" {
			continue
		}
		receipt.EventCount++
		if ev.Phase == "learning" && ev.Outcome != "" {
			if receipt.OutcomeCounts == nil {
				receipt.OutcomeCounts = make(map[string]int)
			}
			receipt.OutcomeCounts[ev.Outcome]++
		}
		actionable := ev.Phase == "" || ev.Phase == "effect"
		if actionable && ev.Utility == "repo_monitor" && isSartreChangeKind(ev.Kind) {
			receipt.Changed++
		}
		if actionable && isSartreChangeKind(ev.Kind) && strings.Contains(strings.ToLower(ev.Path), "readme") {
			receipt.ReadmeChanged = true
		}
		if ev.Resonance > receipt.MaxResonance {
			receipt.MaxResonance = ev.Resonance
		}
		if ev.Relevance > receipt.MaxRelevance {
			receipt.MaxRelevance = ev.Relevance
		}
		if ev.Pulse > receipt.MaxPulse {
			receipt.MaxPulse = ev.Pulse
		}
		if actionable && ev.Utility == "whatdotheythinkiam" && isSartreChangeKind(ev.Kind) {
			receipt.FramingEventCount++
			if ev.Reduced > receipt.MaxReduced {
				receipt.MaxReduced = ev.Reduced
			}
			if ev.Recognized > receipt.MaxRecognized {
				receipt.MaxRecognized = ev.Recognized
			}
		}
		line := ev.Trace()
		if line == "" || seen[line] || len(receipt.Trace) >= 12 {
			continue
		}
		seen[line] = true
		receipt.Trace = append(receipt.Trace, line)
		receipt.Events = append(receipt.Events, ev)
	}
	return receipt
}

// Trace formats one event as a compact memory pressure line.
func (ev SartreEvent) Trace() string {
	if ev.Phase != "" && (ev.Phase != "effect" || ev.Outcome != "") {
		parts := []string{"will", ev.Utility, ev.Phase}
		if ev.Outcome != "" {
			parts = append(parts, ev.Outcome)
		}
		if ev.RootID != "" {
			parts = append(parts, "root="+ev.RootID)
		}
		if ev.Breath > 0 {
			parts = append(parts, fmt.Sprintf("breath=%d", ev.Breath))
		}
		if ev.CadenceMS > 0 {
			parts = append(parts, fmt.Sprintf("cadence_ms=%d", ev.CadenceMS))
		}
		if ev.RefractoryBreaths > 0 {
			parts = append(parts, fmt.Sprintf("refractory_breaths=%d", ev.RefractoryBreaths))
		}
		if ev.CooldownBreaths > 0 {
			parts = append(parts, fmt.Sprintf("cooldown_breaths=%d", ev.CooldownBreaths))
		}
		if ev.EffectCount > 0 {
			parts = append(parts, fmt.Sprintf("effects=%d", ev.EffectCount))
		}
		if ev.BytesCaptured > 0 || ev.BytesLimit > 0 {
			parts = append(parts, fmt.Sprintf("bytes=%d/%d", ev.BytesCaptured, ev.BytesLimit))
		}
		if ev.Attempts > 0 {
			parts = append(parts, fmt.Sprintf("attempts=%d", ev.Attempts))
		}
		if ev.FailureOutcome != "" {
			parts = append(parts, "failure="+ev.FailureOutcome)
		}
		if ev.Kind != "" {
			parts = append(parts, ev.Kind)
		}
		if ev.Path != "" {
			parts = append(parts, ev.Path)
		}
		return compactLine(strings.Join(parts, " "), 180)
	}
	switch ev.Utility {
	case "repo_monitor":
		if ev.Kind == "" && ev.Path == "" {
			return ""
		}
		return compactLine(fmt.Sprintf("repo_monitor %s %s", ev.Kind, ev.Path), 180)
	case "context_processor":
		if ev.Path == "" {
			return ""
		}
		var parts []string
		parts = append(parts, "context_processor", ev.Path)
		if ev.Tag != "" {
			parts = append(parts, "tag="+ev.Tag)
		}
		if ev.Resonance > 0 {
			parts = append(parts, fmt.Sprintf("resonance=%.2f", clamp01(ev.Resonance)))
		}
		parts = append(parts,
			fmt.Sprintf("relevance=%.2f", clamp01(ev.Relevance)),
			fmt.Sprintf("pulse=%.2f", clamp01(ev.Pulse)))
		return compactLine(strings.Join(parts, " "), 220)
	case "whatdotheythinkiam":
		if ev.Kind == "" && ev.Path == "" {
			return ""
		}
		parts := []string{"whatdotheythinkiam"}
		if ev.Path != "" {
			parts = append(parts, ev.Path)
		}
		if ev.Kind != "" {
			parts = append(parts, ev.Kind)
		}
		parts = append(parts,
			fmt.Sprintf("reduced=%d", ev.Reduced),
			fmt.Sprintf("recognized=%d", ev.Recognized))
		return compactLine(strings.Join(parts, " "), 220)
	default:
		return compactLine(strings.TrimSpace(fmt.Sprintf("%s %s %s", ev.Utility, ev.Kind, ev.Path)), 180)
	}
}

type SartreMemory struct {
	lc *LimphaClient
}

func NewSartreMemory(lc *LimphaClient) SartreMemory {
	return SartreMemory{lc: lc}
}

// Recall exposes recent SARTRE utility receipts as bounded pressure traces.
func (m SartreMemory) Recall(n int) []string {
	if m.lc == nil || n <= 0 {
		return nil
	}
	seams, err := m.lc.RecentSeams(n * 4)
	if err != nil {
		return nil
	}
	out := make([]string, 0, n)
	for _, seam := range seams {
		reason, _ := seam["reason"].(string)
		if reason != SartreSeamReason {
			continue
		}
		trace := sartreTraceFromSeam(seam)
		if trace == "" {
			continue
		}
		out = append(out, "SARTRE perception: "+trace)
		if len(out) >= n {
			break
		}
	}
	return out
}

func sartreTraceFromSeam(seam map[string]interface{}) string {
	if delta, _ := seam["memory_delta"].(string); delta != "" {
		var receipt SartreReceipt
		if err := json.Unmarshal([]byte(delta), &receipt); err == nil && receipt.Kind == SartreSeamReason {
			return compactLine(strings.Join(sartreRecallLines(receipt), " | "), 260)
		}
	}
	if b, _ := seam["b_claim"].(string); strings.TrimSpace(b) != "" {
		return compactLine(strings.TrimPrefix(b, "SARTRE perception:"), 260)
	}
	if a, _ := seam["a_claim"].(string); strings.TrimSpace(a) != "" {
		return compactLine(a, 260)
	}
	return ""
}

// sartreRecallLines separates durable operational accounting from pressure worth
// returning to the model. Intention and spawn are lifecycle receipts, and a
// no_novelty learning result closes that lifecycle without becoming experience.
// Consequential effects, framing shifts, measured context pressure, and failures
// remain recallable. Aggregate fallbacks preserve signals that landed after the
// receipt's bounded trace/event preview was already full.
func sartreRecallLines(receipt SartreReceipt) []string {
	lines := make([]string, 0, len(receipt.Events)+4)
	seen := make(map[string]bool)
	add := func(line string) {
		line = strings.TrimSpace(line)
		if line == "" || seen[line] {
			return
		}
		seen[line] = true
		lines = append(lines, line)
	}

	representedChanged := 0
	representedFraming := 0
	representedOutcomes := make(map[string]int)
	var representedResonance, representedRelevance, representedPulse float64
	for _, ev := range receipt.Events {
		ev = normalizeSartreEvent(ev)
		actionable := ev.Phase == "" || ev.Phase == "effect"
		if actionable && ev.Utility == "repo_monitor" && isSartreChangeKind(ev.Kind) {
			representedChanged++
		}
		if actionable && ev.Utility == "whatdotheythinkiam" && isSartreChangeKind(ev.Kind) {
			representedFraming++
		}
		if ev.Phase == "learning" && ev.Outcome != "" {
			representedOutcomes[ev.Outcome]++
		}
		if ev.Phase == "intention" || ev.Phase == "act" ||
			(ev.Phase == "learning" && ev.Outcome == "no_novelty") {
			continue
		}
		if ev.Resonance > representedResonance {
			representedResonance = ev.Resonance
		}
		if ev.Relevance > representedRelevance {
			representedRelevance = ev.Relevance
		}
		if ev.Pulse > representedPulse {
			representedPulse = ev.Pulse
		}
		add(ev.Trace())
	}

	if receipt.Changed > representedChanged {
		line := fmt.Sprintf("repo_monitor changes=%d", receipt.Changed)
		if receipt.ReadmeChanged {
			line += " readme_changed=true"
		}
		add(line)
	}
	if receipt.FramingEventCount > representedFraming {
		add(fmt.Sprintf("whatdotheythinkiam framing_events=%d reduced=%d recognized=%d",
			receipt.FramingEventCount, receipt.MaxReduced, receipt.MaxRecognized))
	}

	outcomes := make([]string, 0, len(receipt.OutcomeCounts))
	for outcome, count := range receipt.OutcomeCounts {
		if outcome != "no_novelty" && count > representedOutcomes[outcome] {
			outcomes = append(outcomes, outcome)
		}
	}
	sort.Strings(outcomes)
	for _, outcome := range outcomes {
		add(fmt.Sprintf("will learning %s count=%d", outcome, receipt.OutcomeCounts[outcome]))
	}

	if receipt.MaxResonance > representedResonance || receipt.MaxRelevance > representedRelevance ||
		receipt.MaxPulse > representedPulse {
		add(fmt.Sprintf("context pressure resonance=%.2f relevance=%.2f pulse=%.2f",
			clamp01(receipt.MaxResonance), clamp01(receipt.MaxRelevance), clamp01(receipt.MaxPulse)))
	}
	return lines
}

func normalizeSartreEvent(ev SartreEvent) SartreEvent {
	ev.ID = strings.TrimSpace(ev.ID)
	ev.Phase = strings.ToLower(strings.TrimSpace(ev.Phase))
	ev.Outcome = strings.ToLower(strings.TrimSpace(ev.Outcome))
	ev.Utility = strings.TrimSpace(ev.Utility)
	ev.Kind = strings.TrimSpace(ev.Kind)
	ev.Path = safeSartrePath(ev.Path)
	ev.Tag = strings.TrimSpace(ev.Tag)
	ev.RootID = strings.TrimSpace(ev.RootID)
	ev.Resonance = clamp01(ev.Resonance)
	ev.Relevance = clamp01(ev.Relevance)
	ev.Pulse = clamp01(ev.Pulse)
	ev.Reduced = maxInt(0, ev.Reduced)
	ev.Recognized = maxInt(0, ev.Recognized)
	ev.Breath = maxInt(0, ev.Breath)
	ev.CadenceMS = maxInt64(0, ev.CadenceMS)
	ev.RefractoryBreaths = maxInt(0, ev.RefractoryBreaths)
	ev.CooldownBreaths = maxInt(0, ev.CooldownBreaths)
	ev.EffectCount = maxInt(0, ev.EffectCount)
	ev.BytesCaptured = maxInt(0, ev.BytesCaptured)
	ev.BytesLimit = maxInt(0, ev.BytesLimit)
	ev.Attempts = maxInt(0, ev.Attempts)
	ev.FailureOutcome = strings.ToLower(strings.TrimSpace(ev.FailureOutcome))
	return ev
}

func isSartreChangeKind(kind string) bool {
	switch strings.TrimSpace(kind) {
	case "added", "modified", "removed":
		return true
	default:
		return false
	}
}

func safeSartrePath(path string) string {
	path = strings.TrimSpace(strings.ReplaceAll(path, "\\", "/"))
	if path == "" {
		return ""
	}
	clean := filepath.ToSlash(filepath.Clean(path))
	parts := strings.Split(clean, "/")
	var live []string
	for _, p := range parts {
		if p != "" && p != "." {
			live = append(live, p)
		}
	}
	if len(live) == 0 {
		return ""
	}
	if len(live) > 3 {
		live = append([]string{"..."}, live[len(live)-3:]...)
	}
	return strings.Join(live, "/")
}

func sartreTension(r SartreReceipt) float64 {
	t := clamp01(r.MaxPulse)
	if t == 0 && r.Changed > 0 {
		t = 0.35
	}
	if r.ReadmeChanged && t < 0.55 {
		t = 0.55
	}
	return t
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
