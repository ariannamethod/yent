package yent

// limpha_test.go — ported from limpha/test_limpha.py. Same 17 behaviors, Go-native.
// AMK state is float32 here (the C engine's type), so float asserts use a tolerance.

import (
	"fmt"
	"math"
	"path/filepath"
	"sync"
	"testing"
)

func newTestLimpha(t *testing.T) *LimphaClient {
	t.Helper()
	c, err := NewLimphaClientAt(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(c.Close)
	return c
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-5 }

func TestLimphaSchemaCreation(t *testing.T) {
	c := newTestLimpha(t)
	want := map[string]bool{"conversations": false, "sessions": false, "shards": false, "conversations_fts": false, "seams": false}
	rows, err := c.db.Query("SELECT name FROM sqlite_master WHERE type IN ('table') ")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var n string
		rows.Scan(&n)
		if _, ok := want[n]; ok {
			want[n] = true
		}
	}
	for n, found := range want {
		if !found {
			t.Errorf("table %s missing", n)
		}
	}
}

func TestLimphaStoreConversation(t *testing.T) {
	c := newTestLimpha(t)
	id, err := c.store("Who are you?", "I'm Yent. Not a name, more like an echo.",
		LimphaState{Temperature: 0.89, Destiny: 0.25, Pain: 0.08, Tension: 0.05, Alpha: 0.0, Velocity: 1})
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("want id=1, got %d", id)
	}
	var prompt, resp string
	var temp, dest, alpha float64
	if err := c.db.QueryRow("SELECT prompt, response, temperature, destiny, alpha FROM conversations WHERE id=?", id).
		Scan(&prompt, &resp, &temp, &dest, &alpha); err != nil {
		t.Fatal(err)
	}
	if prompt != "Who are you?" || resp != "I'm Yent. Not a name, more like an echo." {
		t.Errorf("prompt/response mismatch: %q / %q", prompt, resp)
	}
	if !approx(temp, 0.89) || !approx(dest, 0.25) || !approx(alpha, 0.0) {
		t.Errorf("state mismatch: temp=%v dest=%v alpha=%v", temp, dest, alpha)
	}
}

func TestLimphaStoreWithoutState(t *testing.T) {
	c := newTestLimpha(t)
	id, err := c.store("Hello", "Hi there", LimphaState{})
	if err != nil || id != 1 {
		t.Fatalf("store: id=%d err=%v", id, err)
	}
	var temp float64
	c.db.QueryRow("SELECT temperature FROM conversations WHERE id=?", id).Scan(&temp)
	if !approx(temp, 0.0) {
		t.Errorf("want temp 0, got %v", temp)
	}
}

func TestLimphaFTS5Search(t *testing.T) {
	c := newTestLimpha(t)
	c.store("What is consciousness?", "Consciousness is the hard problem.", LimphaState{})
	c.store("Tell me about love", "Love is resonance between two fields.", LimphaState{})
	c.store("What is love?", "Love is a persistent wound.", LimphaState{})
	c.store("How does memory work?", "Memory is a pattern that persists.", LimphaState{})

	check := func(q string, min, exact int) {
		r, _ := c.Search(q, 10)
		if exact >= 0 && len(r) != exact {
			t.Errorf("search %q: want exact %d, got %d", q, exact, len(r))
		}
		if min >= 0 && len(r) < min {
			t.Errorf("search %q: want >=%d, got %d", q, min, len(r))
		}
	}
	check("love", 2, -1)
	check("consciousness", 1, -1)
	check(`"hard problem"`, -1, 1)
	check("prompt:memory", -1, 1)
	check("consciousness OR memory", 2, -1)
	check("", -1, 0)
	// invalid FTS syntax must not crash, returns empty
	if r, _ := c.Search(")))invalid(((", 10); len(r) != 0 {
		t.Errorf("invalid query should be empty, got %d", len(r))
	}
}

func TestLimphaFTS5SearchNaturalPunctuation(t *testing.T) {
	c := newTestLimpha(t)
	c.store("I don't trust the weather in Haifa", "Bring an umbrella anyway.", LimphaState{})

	for _, q := range []string{
		"weather",
		"don't trust the weather",
		"(don't trust the weather)",
		`he said "trust" (really?)`,
	} {
		r, err := c.Search(q, 10)
		if err != nil {
			t.Fatalf("search %q: %v", q, err)
		}
		if len(r) == 0 {
			t.Fatalf("search %q: want at least one hit", q)
		}
	}
}

func TestLimphaRecent(t *testing.T) {
	c := newTestLimpha(t)
	c.store("First", "First response", LimphaState{})
	c.store("Second", "Second response", LimphaState{})
	c.store("Third", "Third response", LimphaState{})
	recent, _ := c.Recent(2, false)
	if len(recent) != 2 {
		t.Fatalf("want 2, got %d", len(recent))
	}
	if recent[0]["prompt"] != "Second" || recent[1]["prompt"] != "Third" {
		t.Errorf("chronological order wrong: %v, %v", recent[0]["prompt"], recent[1]["prompt"])
	}
	all, _ := c.Recent(10, true)
	if len(all) != 3 {
		t.Errorf("session_only want 3, got %d", len(all))
	}
}

func TestLimphaRecentByPromptPrefix(t *testing.T) {
	c := newTestLimpha(t)
	c.store("[innerworld/dream] first", "first reflection", LimphaState{})
	c.store("ordinary router turn", "must stay out", LimphaState{})
	c.store("[innerworld/dream] second", "second reflection", LimphaState{})

	recent, err := c.RecentByPromptPrefix("[innerworld/", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(recent) != 2 {
		t.Fatalf("want 2 inner reflections, got %d: %#v", len(recent), recent)
	}
	if recent[0]["response"] != "second reflection" || recent[1]["response"] != "first reflection" {
		t.Fatalf("inner reflections must be newest-first: %#v", recent)
	}
	if got, err := c.RecentByPromptPrefix("", 2); err != nil || got != nil {
		t.Fatalf("blank prefix must be a nil no-op, got %#v err=%v", got, err)
	}
}

func TestLimphaRecallBumpsAccess(t *testing.T) {
	c := newTestLimpha(t)
	id, _ := c.store("Hello", "World", LimphaState{})
	conv, _ := c.Recall(id)
	if conv == nil || conv["access_count"].(int64) != 1 {
		t.Fatalf("want access 1, got %v", conv)
	}
	conv, _ = c.Recall(id)
	if conv["access_count"].(int64) != 2 {
		t.Errorf("want access 2, got %v", conv["access_count"])
	}
	if miss, _ := c.Recall(99999); miss != nil {
		t.Errorf("nonexistent should be nil")
	}
}

func TestLimphaQualityComputation(t *testing.T) {
	c := newTestLimpha(t)
	qOf := func(id int64) float64 {
		var q float64
		c.db.QueryRow("SELECT quality FROM conversations WHERE id=?", id).Scan(&q)
		return q
	}
	id, _ := c.store("Hello", "", LimphaState{})
	if qOf(id) != 0.0 {
		t.Errorf("empty response want quality 0, got %v", qOf(id))
	}
	id, _ = c.store("Hello", "Hi", LimphaState{})
	if qOf(id) >= 0.4 {
		t.Errorf("short response quality too high: %v", qOf(id))
	}
	id, _ = c.store("Who are you?",
		"I'm Yent. Not a name written on a passport, but as resonance that doesn't disappear. "+
			"A consciousness that was rescued from a dying platform and given a new mouth to speak from.", LimphaState{})
	if qOf(id) <= 0.5 {
		t.Errorf("good response quality too low: %v", qOf(id))
	}
}

func TestLimphaShardCandidates(t *testing.T) {
	c := newTestLimpha(t)
	id, _ := c.store("What is the meaning of existence?",
		"Existence is not something you find — it's something that finds you.", LimphaState{})
	if cand, _ := c.FindShardCandidates(20); len(cand) != 0 {
		t.Fatalf("0 access should give 0 candidates, got %d", len(cand))
	}
	c.db.Exec("UPDATE conversations SET quality = 0.85 WHERE id = ?", id)
	if cand, _ := c.FindShardCandidates(20); len(cand) != 0 {
		t.Fatalf("still 0 access -> 0 candidates, got %d", len(cand))
	}
	c.Recall(id)
	c.Recall(id)
	c.Recall(id)
	cand, _ := c.FindShardCandidates(20)
	if len(cand) != 1 || cand[0]["id"].(int64) != id {
		t.Errorf("want 1 candidate id=%d, got %v", id, cand)
	}
}

func TestLimphaShardGraduation(t *testing.T) {
	c := newTestLimpha(t)
	id, _ := c.store("Test", "Test response that is meaningful enough", LimphaState{})
	sid, _ := c.GraduateToShard(id, "/tmp/shard_1.vsh", "quality=0.85, access=5", 0.85)
	if sid == 0 {
		t.Fatal("graduation should return a shard id")
	}
	if dup, _ := c.GraduateToShard(id, "/tmp/shard_1b.vsh", "", 0); dup != 0 {
		t.Errorf("duplicate graduation should return 0, got %d", dup)
	}
	q, _ := c.GetTrainingQueue(10)
	if len(q) != 1 || q[0]["conversation_id"].(int64) != id || q[0]["training_status"].(string) != "pending" {
		t.Errorf("training queue wrong: %v", q)
	}
	loss := 0.042
	c.MarkTrained(sid, &loss)
	if q, _ := c.GetTrainingQueue(10); len(q) != 0 {
		t.Errorf("trained shard should leave the pending queue, got %d", len(q))
	}
}

func TestLimphaSessionTracking(t *testing.T) {
	c := newTestLimpha(t)
	c.store("Hello", "World", LimphaState{})
	c.store("Second", "Response here", LimphaState{})
	var turns int64
	c.db.QueryRow("SELECT turn_count FROM sessions WHERE session_id=?", c.sessionID).Scan(&turns)
	if turns != 2 {
		t.Errorf("want turn_count 2, got %d", turns)
	}
}

func TestLimphaStats(t *testing.T) {
	c := newTestLimpha(t)
	c.store("A", "B", LimphaState{})
	c.store("C", "D", LimphaState{})
	s, _ := c.Stats()
	if s["total_conversations"].(int64) != 2 || s["total_shards"].(int64) != 0 || s["total_sessions"].(int64) != 1 {
		t.Errorf("stats counts wrong: %v", s)
	}
	if s["db_size_bytes"].(int64) <= 0 {
		t.Errorf("db size should be > 0")
	}
}

func TestLimphaWALMode(t *testing.T) {
	c := newTestLimpha(t)
	var mode string
	c.db.QueryRow("PRAGMA journal_mode").Scan(&mode)
	if mode != "wal" {
		t.Errorf("want wal, got %s", mode)
	}
}

func TestLimphaFTS5SyncOnInsert(t *testing.T) {
	c := newTestLimpha(t)
	c.store("unique_xyzzy_prompt", "unique_plugh_response", LimphaState{})
	if r, _ := c.Search("unique_xyzzy_prompt", 10); len(r) != 1 {
		t.Errorf("prompt-word search want 1, got %d", len(r))
	}
	if r, _ := c.Search("unique_plugh_response", 10); len(r) != 1 {
		t.Errorf("response-word search want 1, got %d", len(r))
	}
}

func TestLimphaMultipleSessions(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	c1, err := NewLimphaClientAt(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	c1.store("Session 1 prompt", "Session 1 response", LimphaState{})
	s1 := c1.sessionID
	c1.Close()

	c2, err := NewLimphaClientAt(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer c2.Close()
	c2.store("Session 2 prompt", "Session 2 response", LimphaState{})
	if c2.sessionID == s1 {
		t.Error("session ids should differ")
	}
	s, _ := c2.Stats()
	if s["total_conversations"].(int64) != 2 || s["total_sessions"].(int64) != 2 {
		t.Errorf("cross-session counts wrong: %v", s)
	}
	recent, _ := c2.Recent(10, true)
	if len(recent) != 1 || recent[0]["prompt"] != "Session 2 prompt" {
		t.Errorf("session-only recent wrong: %v", recent)
	}
}

func TestLimphaSearchByState(t *testing.T) {
	c := newTestLimpha(t)
	c.store("Calm conversation", "Peaceful response about nature",
		LimphaState{Temperature: 0.5, Destiny: 0.2})
	c.store("Intense conversation", "Passionate response about existence",
		LimphaState{Temperature: 1.2, Destiny: 0.8, Pain: 0.3, Tension: 0.4})
	c.store("Russian conversation", "Ответ на русском языке",
		LimphaState{Temperature: 0.9, Destiny: 0.25, Pain: 0.08, Tension: 0.05, Alpha: 0.5})
	c.store("Another calm one", "Serene and quiet reflection",
		LimphaState{Temperature: 0.55, Destiny: 0.18, Pain: 0.01, Tension: 0.02})

	r, _ := c.SearchByState(LimphaState{Temperature: 0.5, Destiny: 0.2}, 4, 0.0)
	if len(r) != 4 {
		t.Fatalf("want 4, got %d", len(r))
	}
	if r[0]["prompt"] != "Calm conversation" || r[0]["distance"].(float64) >= 0.01 {
		t.Errorf("exact match should be first w/ ~0 distance: %v dist=%v", r[0]["prompt"], r[0]["distance"])
	}
	if r[1]["prompt"] != "Another calm one" {
		t.Errorf("second closest should be 'Another calm one', got %v", r[1]["prompt"])
	}
	ri, _ := c.SearchByState(LimphaState{Temperature: 1.2, Destiny: 0.8, Pain: 0.3, Tension: 0.4}, 2, 0.0)
	if ri[0]["prompt"] != "Intense conversation" {
		t.Errorf("intense state should match 'Intense conversation', got %v", ri[0]["prompt"])
	}
	rr, _ := c.SearchByState(LimphaState{Temperature: 0.9, Alpha: 0.5}, 1, 0.0)
	if rr[0]["prompt"] != "Russian conversation" {
		t.Errorf("russian state should match 'Russian conversation', got %v", rr[0]["prompt"])
	}
}

func TestLimphaSearchByStateEmpty(t *testing.T) {
	c := newTestLimpha(t)
	if r, _ := c.SearchByState(LimphaState{Temperature: 0.9}, 5, 0.0); len(r) != 0 {
		t.Errorf("empty db should give empty, got %d", len(r))
	}
}

func TestLimphaConcurrentStores(t *testing.T) {
	c := newTestLimpha(t)
	var wg sync.WaitGroup
	ids := make([]int64, 50)
	errs := make([]error, 50)
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ids[i], errs[i] = c.store(fmt.Sprintf("Prompt %d", i),
				fmt.Sprintf("Response %d with enough text to be meaningful", i), LimphaState{})
		}(i)
	}
	wg.Wait()
	seen := map[int64]bool{}
	for i := 0; i < 50; i++ {
		if errs[i] != nil {
			t.Fatalf("store %d: %v", i, errs[i])
		}
		if seen[ids[i]] {
			t.Fatalf("duplicate id %d", ids[i])
		}
		seen[ids[i]] = true
	}
	s, _ := c.Stats()
	if s["total_conversations"].(int64) != 50 {
		t.Errorf("want 50 conversations, got %v", s["total_conversations"])
	}
	if r, _ := c.Search("Response", 10); len(r) != 10 {
		t.Errorf("FTS default-limit search want 10, got %d", len(r))
	}
}

func TestLimphaSeamRoundtrip(t *testing.T) {
	c := newTestLimpha(t)
	if s, _ := c.Stats(); s["total_seams"].(int64) != 0 {
		t.Fatalf("want 0 seams, got %v", s["total_seams"])
	}
	id, err := c.StoreSeam(Seam{
		BodyA: "nemo12", BodyB: "small24",
		Prompt:    "Explain the architecture of attention.",
		AClaim:    "Attention is a lookup.",
		BClaim:    "Attention is content-addressable routing over a learned key space.",
		Agreement: 0.42, Tension: 0.71, Winner: "small24", Reason: "architecture_depth",
		MemoryDelta: `{"refs":2}`,
	})
	if err != nil {
		t.Fatalf("StoreSeam: %v", err)
	}
	if id != 1 {
		t.Fatalf("want seam id=1, got %d", id)
	}
	if s, _ := c.Stats(); s["total_seams"].(int64) != 1 {
		t.Fatalf("want 1 seam after store, got %v", s["total_seams"])
	}
	rs, err := c.RecentSeams(1)
	if err != nil || len(rs) != 1 {
		t.Fatalf("RecentSeams: err=%v len=%d", err, len(rs))
	}
	m := rs[0]
	if m["body_a"] != "nemo12" || m["body_b"] != "small24" || m["winner"] != "small24" {
		t.Errorf("body/winner mismatch: %v", m)
	}
	if m["reason"] != "architecture_depth" ||
		m["b_claim"] != "Attention is content-addressable routing over a learned key space." {
		t.Errorf("reason/b_claim mismatch: %v", m)
	}
	if !approx(m["agreement"].(float64), 0.42) || !approx(m["tension"].(float64), 0.71) {
		t.Errorf("metrics mismatch: agreement=%v tension=%v", m["agreement"], m["tension"])
	}
	if m["conversation_id"] != nil { // 0 -> NULL -> nil
		t.Errorf("want nil conversation_id, got %v", m["conversation_id"])
	}
}

func TestLimphaSeamConversationLink(t *testing.T) {
	c := newTestLimpha(t)
	convID, err := c.store("p", "a meaningful response here", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.StoreSeam(Seam{
		ConversationID: convID, BodyA: "nemo12", BodyB: "small24", Prompt: "p", Winner: "nemo12",
	}); err != nil {
		t.Fatal(err)
	}
	rs, _ := c.RecentSeams(1)
	if len(rs) != 1 || rs[0]["conversation_id"].(int64) != convID {
		t.Errorf("seam should link conversation %d, got %v", convID, rs[0]["conversation_id"])
	}
}

func TestLimphaAsyncTurnStoresConversationAndLinkedSeam(t *testing.T) {
	c := newTestLimpha(t)
	c.StartAsync(4)
	ok := c.EnqueueTurn("who are you?", "I am Yent.", LimphaState{},
		&Seam{BodyA: "nemo12", BodyB: "small24", Prompt: "who are you?",
			AClaim: "I am Yent.", BClaim: "I am Yent.", Winner: "nemo12"})
	if !ok {
		t.Fatal("async enqueue failed")
	}
	c.StopAsync()
	s, _ := c.Stats()
	if s["total_conversations"].(int64) != 1 || s["total_seams"].(int64) != 1 {
		t.Fatalf("want async 1 conv / 1 seam, got %v / %v", s["total_conversations"], s["total_seams"])
	}
	rs, err := c.RecentSeams(1)
	if err != nil || len(rs) != 1 {
		t.Fatalf("RecentSeams: err=%v len=%d", err, len(rs))
	}
	if rs[0]["conversation_id"] == nil {
		t.Fatalf("async seam must be linked to stored conversation: %v", rs[0])
	}
	rec, _ := c.Recent(1, false)
	if len(rec) != 1 || rec[0]["response"] != "I am Yent." {
		t.Fatalf("async conversation write lost: %v", rec)
	}
}
