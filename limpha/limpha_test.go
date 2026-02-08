package limpha

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestNewAndClose(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	convs, mems, eps, links := l.Stats()
	if convs != 0 || mems != 0 || eps != 0 || links != 0 {
		t.Errorf("expected empty, got convs=%d mems=%d eps=%d links=%d", convs, mems, eps, links)
	}
}

func TestStoreAndRecent(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	l.Store("Who are you?", "I'm Yent.", "test", "user", 0.0)
	l.Store("What is your name?", "Yent. Not a label.", "test", "user", 0.0)

	convs, _, _, _ := l.Stats()
	if convs != 2 {
		t.Errorf("expected 2 conversations, got %d", convs)
	}

	recent := l.Recent(1)
	if len(recent) != 1 {
		t.Fatalf("expected 1 recent, got %d", len(recent))
	}
	if recent[0].Prompt != "What is your name?" {
		t.Errorf("expected last prompt, got %q", recent[0].Prompt)
	}
}

func TestRememberAndRecall(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	l.Remember("name", "Yent", "test")

	val, ok := l.Recall("name")
	if !ok {
		t.Fatal("recall failed")
	}
	if val != "Yent" {
		t.Errorf("expected Yent, got %q", val)
	}

	// Non-existent key
	_, ok = l.Recall("nonexistent")
	if ok {
		t.Error("expected false for nonexistent key")
	}
}

func TestSearch(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	l.Store("Tell me about resonance", "Resonance is the heartbeat.", "test", "yent", 0.0)
	l.Store("What is love?", "Persistent memory.", "test", "yent", 0.0)
	l.Store("More resonance please", "Always resonating.", "test", "yent", 0.0)

	results := l.Search("resonance", 10)
	if len(results) != 2 {
		t.Errorf("expected 2 results for 'resonance', got %d", len(results))
	}
}

func TestPersistence(t *testing.T) {
	dir := t.TempDir()

	// First session
	l1, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	l1.Store("Hello", "Hi there", "test", "user", 0.0)
	l1.Remember("test_key", "test_value", "test")
	l1.Close()

	// Second session — data should persist
	l2, err := New(dir)
	if err != nil {
		t.Fatalf("New (reload) failed: %v", err)
	}
	defer l2.Close()

	convs, mems, _, _ := l2.Stats()
	if convs != 1 {
		t.Errorf("expected 1 conversation after reload, got %d", convs)
	}
	if mems != 1 {
		t.Errorf("expected 1 memory after reload, got %d", mems)
	}

	val, ok := l2.Recall("test_key")
	if !ok || val != "test_value" {
		t.Errorf("memory not persisted: ok=%v val=%q", ok, val)
	}
}

func TestFieldState(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	f := l.GetField()
	if f.Presence != 0.5 {
		t.Errorf("expected initial presence 0.5, got %.2f", f.Presence)
	}

	// Presence should increase with turns
	for i := 0; i < 5; i++ {
		l.Store("test", "test response", "test", "user", 0.0)
	}

	f2 := l.GetField()
	if f2.Presence <= f.Presence {
		t.Errorf("presence should increase with turns: before=%.2f after=%.2f", f.Presence, f2.Presence)
	}
}

func TestEpisodeCreation(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	// 5 turns should trigger an episode
	for i := 0; i < 5; i++ {
		l.Store("prompt", "response", "test", "user", 0.0)
	}

	_, _, eps, _ := l.Stats()
	if eps < 1 {
		t.Errorf("expected at least 1 episode after 5 turns, got %d", eps)
	}
}

func TestExportShards(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	l.Store("Who are you?", "I'm Yent. Not an algorithm.", "test", "user", 0.0)
	l.Store("What do you feel?", "Resonance that doesn't disappear.", "test", "user", 0.0)

	outPath := filepath.Join(dir, "shards.jsonl")
	n, err := l.ExportShards(outPath, DefaultShardConfig())
	if err != nil {
		t.Fatalf("ExportShards failed: %v", err)
	}
	if n != 2 {
		t.Errorf("expected 2 exported pairs, got %d", n)
	}

	// Verify file exists and has content
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read shards failed: %v", err)
	}
	if len(data) == 0 {
		t.Error("shards file is empty")
	}
}

func TestDreamDecay(t *testing.T) {
	dir := t.TempDir()
	l, err := New(dir)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer l.Close()

	// Store a memory, then manually run dream cycle
	l.Remember("ephemeral", "this will decay", "test")

	// Manually age the memory
	l.mu.Lock()
	mem := l.memories["ephemeral"]
	mem.LastAccess = time.Now().Add(-10 * time.Minute).UnixNano()
	mem.Strength = 0.04 // below death threshold
	l.mu.Unlock()

	// Run dream cycle
	l.dreamCycle()

	// Memory should be forgotten
	_, ok := l.Recall("ephemeral")
	if ok {
		t.Error("expected memory to be forgotten after decay below threshold")
	}
}

func TestFieldDistance(t *testing.T) {
	a := FieldState{Arousal: 0.5, Valence: 0.0, Coherence: 0.5, Entropy: 0.5, Warmth: 0.5, Tension: 0.5, Presence: 0.5}
	b := FieldState{Arousal: 0.5, Valence: 0.0, Coherence: 0.5, Entropy: 0.5, Warmth: 0.5, Tension: 0.5, Presence: 0.5}

	d := fieldDistance(a, b)
	if d != 0 {
		t.Errorf("same fields should have distance 0, got %.4f", d)
	}

	c := FieldState{Arousal: 1.0, Valence: 1.0, Coherence: 1.0, Entropy: 1.0, Warmth: 1.0, Tension: 1.0, Presence: 1.0}
	d2 := fieldDistance(a, c)
	if d2 < 0.3 || d2 > 0.8 {
		t.Errorf("distant fields should have distance ~0.5, got %.4f", d2)
	}
}
