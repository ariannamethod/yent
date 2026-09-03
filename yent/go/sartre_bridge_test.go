package yent

import (
	"strings"
	"testing"
)

func TestParseSartreEventsJSONLIgnoresSlotNoise(t *testing.T) {
	jsonl := `[pipe] slot[0] pid=123 reading events:
{"util":"repo_monitor","kind":"modified","path":"/Users/ariannamethod/arianna/yent/README.md","ts":1}
not json
{"util":"context_processor","path":"/Users/ariannamethod/arianna/yent/research/recursive_resonance_preprint.md","resonance":0.52,"relevance":0.42,"pulse":0.73}
{"util":"whatdotheythinkiam","path":"README.md","kind":"modified","reduced":4,"recognized":2,"ts":1}
`
	events := ParseSartreEventsJSONL(jsonl)
	if len(events) != 3 {
		t.Fatalf("want three events, got %#v", events)
	}
	if events[0].Utility != "repo_monitor" || events[0].Kind != "modified" ||
		events[0].Path != ".../arianna/yent/README.md" {
		t.Fatalf("repo event not normalized: %#v", events[0])
	}
	if events[1].Utility != "context_processor" || events[1].Tag != "" ||
		events[1].Resonance != 0.52 || events[1].Relevance != 0.42 || events[1].Pulse != 0.73 ||
		events[1].Path != ".../yent/research/recursive_resonance_preprint.md" {
		t.Fatalf("context event not normalized: %#v", events[1])
	}
	if events[2].Utility != "whatdotheythinkiam" || events[2].Kind != "modified" ||
		events[2].Path != "README.md" || events[2].Reduced != 4 || events[2].Recognized != 2 {
		t.Fatalf("framing event not preserved: %#v", events[2])
	}
}

func TestStoreSartreEventsAndRecall(t *testing.T) {
	lc := newTestLimpha(t)
	events := []SartreEvent{
		{Utility: "repo_monitor", Kind: "modified", Path: "/repo/README.md"},
		{Utility: "context_processor", Path: "/repo/research/recursive_resonance_preprint.md", Resonance: 0.63, Relevance: 0.5, Pulse: 0.7},
		{Utility: "whatdotheythinkiam", Kind: "modified", Path: "README.md", Reduced: 4, Recognized: 2},
	}
	id, err := lc.StoreSartreEvents(events, LimphaState{Destiny: 0.3, Velocity: 2})
	if err != nil {
		t.Fatal(err)
	}
	if id == 0 {
		t.Fatal("expected seam id")
	}
	stats, err := lc.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats["total_conversations"].(int64) != 1 || stats["total_seams"].(int64) != 1 {
		t.Fatalf("want 1 conversation / 1 seam, got %v / %v", stats["total_conversations"], stats["total_seams"])
	}
	seams, err := lc.RecentSeams(1)
	if err != nil || len(seams) != 1 {
		t.Fatalf("RecentSeams: err=%v len=%d", err, len(seams))
	}
	if seams[0]["body_a"] != "sartre" || seams[0]["body_b"] != "limpha" ||
		seams[0]["reason"] != SartreSeamReason || seams[0]["winner"] != "limpha" {
		t.Fatalf("wrong sartre seam: %+v", seams[0])
	}
	if !strings.Contains(seams[0]["memory_delta"].(string), `"kind":"sartre_perception"`) {
		t.Fatalf("memory_delta must be typed JSON: %s", seams[0]["memory_delta"])
	}
	if !strings.Contains(seams[0]["memory_delta"].(string), `"max_resonance":0.63`) {
		t.Fatalf("memory_delta must preserve resonance: %s", seams[0]["memory_delta"])
	}
	if !strings.Contains(seams[0]["memory_delta"].(string), `"framing_event_count":1`) ||
		!strings.Contains(seams[0]["memory_delta"].(string), `"max_reduced":4`) ||
		!strings.Contains(seams[0]["memory_delta"].(string), `"max_recognized":2`) {
		t.Fatalf("memory_delta must preserve framing counts: %s", seams[0]["memory_delta"])
	}

	got := NewSartreMemory(lc).Recall(2)
	if len(got) != 1 {
		t.Fatalf("want one sartre memory trace, got %#v", got)
	}
	if !strings.Contains(got[0], "SARTRE perception:") ||
		!strings.Contains(got[0], "repo_monitor modified") ||
		!strings.Contains(got[0], "context_processor") ||
		!strings.Contains(got[0], "resonance=0.63") ||
		!strings.Contains(got[0], "whatdotheythinkiam README.md modified reduced=4 recognized=2") ||
		strings.Contains(got[0], "tag=?") {
		t.Fatalf("bad sartre recall trace: %#v", got)
	}
	if strings.Contains(got[0], "/Users/") {
		t.Fatalf("absolute local path leaked into trace: %q", got[0])
	}
}

func TestStoreSartreEventsDeduplicatesStableEventIDs(t *testing.T) {
	lc := newTestLimpha(t)
	event := SartreEvent{
		ID:        "reach-1",
		RootID:    "root-a",
		Phase:     "effect",
		Utility:   "repo_monitor",
		Kind:      "modified",
		Path:      "/repo/README.md",
		Timestamp: 10,
		Breath:    3,
	}
	id, accepted, err := lc.StoreNewSartreEvents([]SartreEvent{event}, LimphaState{Destiny: 0.3})
	if err != nil {
		t.Fatal(err)
	}
	if id == 0 || len(accepted) != 1 {
		t.Fatalf("first stable event should store one seam, id=%d accepted=%#v", id, accepted)
	}
	retry := event
	retry.Timestamp = 99
	retry.Breath = 8
	id, accepted, err = lc.StoreNewSartreEvents([]SartreEvent{retry}, LimphaState{Destiny: 0.3})
	if err != nil {
		t.Fatal(err)
	}
	if id != 0 || len(accepted) != 0 {
		t.Fatalf("retry of the same causal event should be a no-op, id=%d accepted=%#v", id, accepted)
	}
	stats, err := lc.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats["total_conversations"].(int64) != 1 || stats["total_seams"].(int64) != 1 {
		t.Fatalf("duplicate retry should not grow memory, got %v / %v", stats["total_conversations"], stats["total_seams"])
	}
}

func TestStoreNewSartreEventsKeepsNewPhaseForSameReach(t *testing.T) {
	lc := newTestLimpha(t)
	first := SartreEvent{
		ID:      "reach-2",
		RootID:  "root-a",
		Phase:   "intention",
		Outcome: "crest",
		Utility: "repo_monitor",
	}
	if id, accepted, err := lc.StoreNewSartreEvents([]SartreEvent{first}, LimphaState{}); err != nil || id == 0 || len(accepted) != 1 {
		t.Fatalf("first phase should store: id=%d accepted=%#v err=%v", id, accepted, err)
	}
	second := []SartreEvent{
		first,
		{ID: "reach-2", RootID: "root-a", Phase: "learning", Outcome: "perception_committed", Utility: "repo_monitor", EffectCount: 1},
	}
	id, accepted, err := lc.StoreNewSartreEvents(second, LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if id == 0 || len(accepted) != 1 || accepted[0].Phase != "learning" {
		t.Fatalf("a new phase on the same reach should pass while duplicate intention is skipped, id=%d accepted=%#v", id, accepted)
	}
	stats, err := lc.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats["total_conversations"].(int64) != 2 || stats["total_seams"].(int64) != 2 {
		t.Fatalf("two distinct phases should produce two seams, got %v / %v", stats["total_conversations"], stats["total_seams"])
	}
}

func TestStoreNewSartreEventsKeepsRetryAttempts(t *testing.T) {
	lc := newTestLimpha(t)
	first := SartreEvent{
		ID:       "reach-retry",
		RootID:   "root-a",
		Phase:    "learning",
		Outcome:  "sensor_error",
		Utility:  "repo_monitor",
		Attempts: 1,
	}
	if id, accepted, err := lc.StoreNewSartreEvents([]SartreEvent{first}, LimphaState{}); err != nil || id == 0 || len(accepted) != 1 {
		t.Fatalf("first attempt should store: id=%d accepted=%#v err=%v", id, accepted, err)
	}
	second := first
	second.Attempts = 2
	id, accepted, err := lc.StoreNewSartreEvents([]SartreEvent{second}, LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if id == 0 || len(accepted) != 1 || accepted[0].Attempts != 2 {
		t.Fatalf("a new retry attempt should pass stable-event dedupe, id=%d accepted=%#v", id, accepted)
	}
}

func TestStoreNewSartreEventsAcknowledgesTraceEmptyStableEvent(t *testing.T) {
	lc := newTestLimpha(t)
	ev := SartreEvent{
		ID:      "reach-empty",
		RootID:  "root-a",
		Phase:   "effect",
		Utility: "repo_monitor",
	}
	id, accepted, err := lc.StoreNewSartreEvents([]SartreEvent{ev}, LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if id != 0 || len(accepted) != 1 {
		t.Fatalf("trace-empty stable event should ack ids without a seam, id=%d accepted=%#v", id, accepted)
	}
	id, accepted, err = lc.StoreNewSartreEvents([]SartreEvent{ev}, LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if id != 0 || len(accepted) != 0 {
		t.Fatalf("trace-empty stable event should dedupe on retry, id=%d accepted=%#v", id, accepted)
	}
	stats, err := lc.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats["total_conversations"].(int64) != 0 || stats["total_seams"].(int64) != 0 {
		t.Fatalf("trace-empty ack should not write memory rows, got %v / %v", stats["total_conversations"], stats["total_seams"])
	}
}

func TestBuildSartreReceiptCapsTraceButCountsMetrics(t *testing.T) {
	var events []SartreEvent
	for i := 0; i < 20; i++ {
		events = append(events, SartreEvent{Utility: "repo_monitor", Kind: "modified", Path: "research/file.md"})
	}
	events = append(events, SartreEvent{Utility: "context_processor", Path: "research/hot.md", Resonance: 0.94, Relevance: 0.91, Pulse: 0.88})
	events = append(events, SartreEvent{Utility: "whatdotheythinkiam", Path: "README.md", Kind: "modified", Reduced: 3, Recognized: 7})

	receipt := BuildSartreReceipt(events)
	if receipt.EventCount != 22 {
		t.Fatalf("receipt should count the whole capped packet, got %d", receipt.EventCount)
	}
	if receipt.MaxResonance != 0.94 || receipt.MaxRelevance != 0.91 || receipt.MaxPulse != 0.88 {
		t.Fatalf("late metric event should still count, resonance=%.2f relevance=%.2f pulse=%.2f", receipt.MaxResonance, receipt.MaxRelevance, receipt.MaxPulse)
	}
	if len(receipt.Trace) > 12 {
		t.Fatalf("trace should stay capped, got %d", len(receipt.Trace))
	}
	if receipt.FramingEventCount != 1 || receipt.MaxReduced != 3 || receipt.MaxRecognized != 7 {
		t.Fatalf("framing counts should survive receipt build: %+v", receipt)
	}
}

func TestBuildSartreReceiptKeepsWillPhasesOutOfChangeCounts(t *testing.T) {
	events := []SartreEvent{
		{ID: "r1", Phase: "intention", Outcome: "crest", Utility: "repo_monitor", Kind: "modified", Path: "README.md", RootID: "rootabc", Breath: 2, CadenceMS: 750, RefractoryBreaths: 3},
		{ID: "r1", Phase: "act", Outcome: "spawned", Utility: "repo_monitor"},
		{ID: "r1", Phase: "learning", Outcome: "no_novelty", Utility: "repo_monitor"},
		{ID: "r2", Phase: "learning", Outcome: "overflow", Utility: "repo_monitor", BytesCaptured: 1024, BytesLimit: 2048},
		{ID: "r2", Phase: "learning", Outcome: "dead_letter", Utility: "repo_monitor", Attempts: 3, FailureOutcome: "overflow"},
		{ID: "r2", Phase: "effect", Utility: "repo_monitor", Kind: "modified", Path: "README.md"},
		{ID: "r2", Phase: "learning", Outcome: "perception_committed", Utility: "repo_monitor", EffectCount: 1},
	}
	receipt := BuildSartreReceipt(events)
	if receipt.EventCount != 7 {
		t.Fatalf("all typed events should be counted as received, got %d", receipt.EventCount)
	}
	if receipt.Changed != 1 || !receipt.ReadmeChanged {
		t.Fatalf("only the actual effect/change event should move change counters: %+v", receipt)
	}
	if receipt.OutcomeCounts["no_novelty"] != 1 ||
		receipt.OutcomeCounts["overflow"] != 1 ||
		receipt.OutcomeCounts["dead_letter"] != 1 ||
		receipt.OutcomeCounts["perception_committed"] != 1 {
		t.Fatalf("learning outcomes should be typed counters: %+v", receipt.OutcomeCounts)
	}
	trace := strings.Join(receipt.Trace, " | ")
	if len(receipt.Trace) < 2 ||
		!strings.Contains(trace, "will repo_monitor intention crest") ||
		!strings.Contains(trace, "root=rootabc") ||
		!strings.Contains(trace, "breath=2 cadence_ms=750 refractory_breaths=3") ||
		!strings.Contains(trace, "will repo_monitor learning overflow bytes=1024/2048") ||
		!strings.Contains(trace, "will repo_monitor learning dead_letter attempts=3 failure=overflow") ||
		!strings.Contains(trace, "will repo_monitor learning perception_committed effects=1") ||
		!strings.Contains(trace, "repo_monitor modified README.md") {
		t.Fatalf("receipt should preserve will phases and effect trace: %#v", receipt.Trace)
	}
}

func TestSartreMemoryFiltersOtherSeams(t *testing.T) {
	lc := newTestLimpha(t)
	if _, err := lc.StoreSeam(Seam{
		BodyA: "nemo12", BodyB: "small24", Reason: "innerworld_self_answer",
		BClaim: "inner monologue should not be returned by SARTRE memory",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := lc.StoreSartreEvents([]SartreEvent{{Utility: "repo_monitor", Kind: "added", Path: "research/new.md"}}, LimphaState{}); err != nil {
		t.Fatal(err)
	}
	got := NewSartreMemory(lc).Recall(3)
	if len(got) != 1 || !strings.Contains(got[0], "research/new.md") {
		t.Fatalf("SARTRE memory should return only sartre seams, got %#v", got)
	}
}

func TestSartreMemoryDoesNotRehearseQuietWillAccounting(t *testing.T) {
	lc := newTestLimpha(t)
	quiet := []SartreEvent{
		{ID: "quiet-1", Phase: "intention", Outcome: "crest", Utility: "repo_monitor"},
		{ID: "quiet-1", Phase: "act", Outcome: "spawned", Utility: "repo_monitor"},
		{ID: "quiet-1", Phase: "learning", Outcome: "no_novelty", Utility: "repo_monitor"},
	}
	if id, err := lc.StoreSartreEvents(quiet, LimphaState{}); err != nil || id == 0 {
		t.Fatalf("quiet accounting must remain durable, id=%d err=%v", id, err)
	}
	if got := NewSartreMemory(lc).Recall(3); len(got) != 0 {
		t.Fatalf("quiet no-novelty accounting must not enter the model seed: %#v", got)
	}

	failure := []SartreEvent{{
		ID: "failed-1", Phase: "learning", Outcome: "sensor_error", Utility: "repo_monitor", Attempts: 2,
	}}
	if id, err := lc.StoreSartreEvents(failure, LimphaState{}); err != nil || id == 0 {
		t.Fatalf("failure receipt must remain durable, id=%d err=%v", id, err)
	}
	got := NewSartreMemory(lc).Recall(3)
	if len(got) != 1 || !strings.Contains(got[0], "sensor_error") {
		t.Fatalf("failure must remain recallable pressure: %#v", got)
	}
}

func TestSartreMemoryFindsLateSignalWithoutRehearsingCappedQuietAccounting(t *testing.T) {
	lc := newTestLimpha(t)
	var events []SartreEvent
	for i := 0; i < 5; i++ {
		id := "quiet-" + string(rune('a'+i))
		breath := i + 1
		events = append(events,
			SartreEvent{ID: id, Phase: "intention", Outcome: "crest", Utility: "repo_monitor", Breath: breath},
			SartreEvent{ID: id, Phase: "act", Outcome: "spawned", Utility: "repo_monitor", Breath: breath},
			SartreEvent{ID: id, Phase: "learning", Outcome: "no_novelty", Utility: "repo_monitor", Breath: breath},
		)
	}
	events = append(events,
		SartreEvent{ID: "changed-1", Phase: "effect", Utility: "repo_monitor", Kind: "modified", Path: "yent/go/sartre_bridge.go"},
		SartreEvent{ID: "changed-1", Phase: "learning", Outcome: "perception_committed", Utility: "repo_monitor", EffectCount: 1},
	)
	if id, err := lc.StoreSartreEvents(events, LimphaState{}); err != nil || id == 0 {
		t.Fatalf("mixed receipt must remain durable, id=%d err=%v", id, err)
	}

	got := NewSartreMemory(lc).Recall(3)
	if len(got) != 1 || !strings.Contains(got[0], "repo_monitor changes=1") ||
		!strings.Contains(got[0], "perception_committed") {
		t.Fatalf("late consequential signal must survive the bounded receipt preview: %#v", got)
	}
	if strings.Contains(got[0], "no_novelty") || strings.Contains(got[0], "intention crest") ||
		strings.Contains(got[0], "act spawned") {
		t.Fatalf("quiet lifecycle accounting must not be rehearsed around a real signal: %#v", got)
	}
}
