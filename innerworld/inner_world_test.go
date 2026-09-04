package innerworld

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeMemory returns canned past monologues for the recall path.
type fakeMemory struct{ past []string }

func (m fakeMemory) Recall(n int) []string {
	if n < len(m.past) {
		return m.past[:n]
	}
	return m.past
}

func TestRecallFoldsIn(t *testing.T) {
	// circle 0's seed must carry the recalled thoughts AND the prompt — the organism
	// thinks with what it thought before, NO-SEED-FROM-PROMPT still intact.
	iw := NewInnerWorld(fakeBody{}, &fakeField{}, tempDivergence)
	iw.SetMemory(fakeMemory{past: []string{"the void is patient", "form before name"}})
	r := <-iw.Think("what is code")
	if len(r.Circles) == 0 {
		t.Fatal("no circles")
	}
	seed := r.Circles[0].Seed
	if !strings.Contains(seed, "the void is patient") {
		t.Errorf("circle 0 seed should carry the recalled thought, got %q", seed)
	}
	if !strings.Contains(seed, "what is code") {
		t.Errorf("circle 0 seed should still carry the prompt, got %q", seed)
	}
	if !strings.Contains(seed, "not dialogue to continue or imitate") ||
		!strings.Contains(seed, "field traces") ||
		!strings.Contains(seed, "Think fresh from the current human turn") {
		t.Errorf("recall should be framed as bounded pressure, got %q", seed)
	}
	if strings.Contains(seed, "Recalling earlier thoughts") {
		t.Errorf("raw recall framing should not survive, got %q", seed)
	}
}

func TestRecallNilSafe(t *testing.T) {
	// no memory -> seed is just the prompt's inner transform, unchanged behavior.
	iw := NewInnerWorld(fakeBody{}, &fakeField{}, tempDivergence)
	r := <-iw.Think("plain prompt")
	if len(r.Circles) == 0 {
		t.Fatal("no circles")
	}
	if strings.Contains(r.Circles[0].Seed, "Recalling") {
		t.Errorf("no memory must not inject recall, got %q", r.Circles[0].Seed)
	}
}

func TestThinkAsync(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{}, tempDivergence)
	ch := iw.Think("what does it mean to exist as code")
	select {
	case r := <-ch:
		if len(r.Circles) != 3 {
			t.Fatalf("want 3 circles, got %d", len(r.Circles))
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Think did not deliver a reflection")
	}
}

func TestDue(t *testing.T) {
	now := time.Now()

	// drift: high field debt, cooldown elapsed -> drift fires
	iw := NewInnerWorld(fakeBody{}, &fakeField{debt: 2.0}, tempDivergence)
	iw.lastActive = now // not idle
	iw.lastFire[trigDrift] = now.Add(-10 * time.Second)
	if trig, ok := iw.due(now); !ok || trig != trigDrift {
		t.Errorf("want drift due, got trig=%d ok=%v", trig, ok)
	}
	// just fired -> on cooldown, nothing due
	iw.lastFire[trigDrift] = now
	if _, ok := iw.due(now); ok {
		t.Errorf("drift should be on cooldown, but something fired")
	}

	// silence: low debt, idle long enough -> silence fires
	iw2 := NewInnerWorld(fakeBody{}, &fakeField{debt: 0}, tempDivergence)
	iw2.lastActive = now.Add(-10 * time.Second)
	iw2.lastFire[trigSilence] = now.Add(-10 * time.Second)
	if trig, ok := iw2.due(now); !ok || trig != trigSilence {
		t.Errorf("want silence due, got trig=%d ok=%v", trig, ok)
	}

	// not idle, low debt -> nothing due
	iw3 := NewInnerWorld(fakeBody{}, &fakeField{debt: 0}, tempDivergence)
	iw3.lastActive = now
	if _, ok := iw3.due(now); ok {
		t.Errorf("nothing should be due (not idle, low debt)")
	}
}

func TestDream(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{debt: 2.0}, tempDivergence)
	var got Reflection
	iw.SetOnDream(func(r Reflection) { got = r }) // dream calls OnDream synchronously
	before := time.Now()
	iw.lastActive = before.Add(-10 * time.Second)

	r := iw.dream(trigSilence)
	if len(r.Circles) != 3 {
		t.Fatalf("want 3 dream circles, got %d", len(r.Circles))
	}
	if len(got.Circles) != 3 {
		t.Errorf("OnDream not fired with 3 circles, got %d", len(got.Circles))
	}
	// Any autonomous speech starts the refractory for every trigger. Otherwise
	// drift and silence can alternate into a back-to-back double dream.
	for trig, fired := range iw.lastFire {
		if fired.Before(before) {
			t.Errorf("trigger %d refractory not set to dream-completion time", trig)
		}
	}
	if _, ok := iw.due(time.Now()); ok {
		t.Error("another autonomous trigger fired immediately after a completed dream")
	}
}

func TestBreatheStops(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{}, tempDivergence)
	iw.br.Tick = time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { iw.Breathe(ctx); close(done) }()
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Breathe did not return on ctx cancel")
	}
}

func TestBreatheFires(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{debt: 2.0}, tempDivergence)
	iw.br.Tick = time.Millisecond
	iw.br.Cooldown[trigDrift] = time.Millisecond
	var n int32
	iw.SetOnDream(func(Reflection) { atomic.AddInt32(&n, 1) })

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Millisecond)
	defer cancel()
	iw.Breathe(ctx)

	if atomic.LoadInt32(&n) < 1 {
		t.Errorf("breathe with high drift should dream at least once, got %d", n)
	}
}

func TestBreatheStopsAfterDreamCancelsContext(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{debt: 2.0}, tempDivergence)
	iw.br.Tick = time.Millisecond
	iw.br.Cooldown[trigDrift] = 0
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var n int32
	iw.SetOnDream(func(Reflection) {
		atomic.AddInt32(&n, 1)
		cancel()
	})

	done := make(chan struct{})
	go func() { iw.Breathe(ctx); close(done) }()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Breathe did not return after dream callback cancelled ctx")
	}
	if got := atomic.LoadInt32(&n); got != 1 {
		t.Fatalf("want exactly one dream after callback cancels ctx, got %d", got)
	}
}

func TestCloneIsolation(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{}, tempDivergence)
	r := <-iw.Think("a question")
	if len(r.Circles) == 0 {
		t.Fatal("no circles returned")
	}
	r.Circles[0].Text = "MUTATED" // mutate the caller's copy
	iw.mu.Lock()
	internal := iw.circles[0].Text
	iw.mu.Unlock()
	if internal == "MUTATED" {
		t.Errorf("external mutation leaked into iw.circles — clone failed")
	}
}

func TestConcurrentSafe(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{debt: 2.0}, tempDivergence)
	iw.br.Tick = time.Millisecond
	iw.br.Cooldown[trigDrift] = time.Millisecond
	iw.SetOnDream(func(Reflection) {})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); iw.Breathe(ctx) }() // autonomous dreaming
	for i := 0; i < 20; i++ {
		<-iw.Think("turn") // human turns, concurrent with the dreaming
	}
	wg.Wait()
}

func TestReflectGate(t *testing.T) {
	// inject a high-coupling Larynx and a low roll: with strong agitation the deep
	// body must turn inward and answer itself.
	iw := NewInnerWorld(fakeBody{}, &fakeField{debt: 5.0}, tempDivergence)
	iw.SetLarynx(fixedLarynx{0.9})
	iw.SetRoll(func() float32 { return 0.0 }) // roll below any positive probability
	r := <-iw.Think("a hard question")
	if r.Coupling != 0.9 {
		t.Errorf("coupling should come from the injected Larynx, got %.2f", r.Coupling)
	}
	if r.SelfAnswerProb <= 0 || r.SelfAnswerProb > 1 {
		t.Errorf("self-answer probability out of (0,1]: %.3f", r.SelfAnswerProb)
	}
	if !r.SelfAnswered {
		t.Errorf("roll 0 against prob %.3f should self-answer", r.SelfAnswerProb)
	}

	// a roll above the probability must not self-answer
	iw.SetRoll(func() float32 { return 1.0 })
	r2 := <-iw.Think("again")
	if r2.SelfAnswered {
		t.Errorf("roll 1.0 should never self-answer (prob %.3f)", r2.SelfAnswerProb)
	}
}

type fixedLarynx struct{ c float32 }

func (f fixedLarynx) Couple([]Circle) float32 { return f.c }

func TestBreatheZeroTick(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{}, tempDivergence)
	iw.SetBreath(Breath{Tick: 0}) // a non-positive tick would panic time.NewTicker
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	iw.Breathe(ctx) // must not panic; returns on ctx
}

func TestSetBreath(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{}, tempDivergence)
	b := Breath{Tick: 7 * time.Millisecond, Silence: time.Second, DriftDebt: 5}
	iw.SetBreath(b)
	iw.mu.Lock()
	got := iw.br
	iw.mu.Unlock()
	if got.Tick != b.Tick || got.DriftDebt != b.DriftDebt {
		t.Errorf("SetBreath did not apply: %+v", got)
	}
}

// recordingBody is a closable fake that counts generations and closes, so the deep
// self-answer and the single-resident swap can be asserted.
type recordingBody struct {
	mu     sync.Mutex
	answer string
	gens   int
	closes int
}

func (b *recordingBody) Generate(seed string, _ float32) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.gens++
	if b.answer != "" {
		return b.answer
	}
	return seed + " ·" // shift the text so divergence is not degenerate
}

func (b *recordingBody) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closes++
	return nil
}

func (b *recordingBody) counts() (gens, closes int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.gens, b.closes
}

func TestDeepSelfAnswer(t *testing.T) {
	iw := NewInnerWorld(fakeBody{}, &fakeField{debt: 5.0}, tempDivergence)
	deep := &recordingBody{answer: "the deep body turns inward"}
	iw.SetDeep(deep)
	iw.SetLarynx(fixedLarynx{0.9})

	// gate fires (roll below any positive prob): the deep body actually answers.
	iw.SetRoll(func() float32 { return 0.0 })
	r := <-iw.Think("a hard question")
	if !r.SelfAnswered {
		t.Fatalf("roll 0 should fire the gate")
	}
	if r.DeepAnswer != "the deep body turns inward" {
		t.Errorf("DeepAnswer should carry the deep body's text, got %q", r.DeepAnswer)
	}
	if gens, _ := deep.counts(); gens != 1 {
		t.Errorf("deep body should generate exactly once on a fired gate, got %d", gens)
	}

	// gate does not fire (roll above prob): the deep body stays silent.
	iw.SetRoll(func() float32 { return 1.0 })
	r2 := <-iw.Think("again")
	if r2.SelfAnswered || r2.DeepAnswer != "" {
		t.Errorf("gate false -> no deep answer, got answered=%v deep=%q", r2.SelfAnswered, r2.DeepAnswer)
	}
	if gens, _ := deep.counts(); gens != 1 {
		t.Errorf("deep body must not generate when the gate is false; gens=%d", gens)
	}
}

func TestSingleResidentSwap(t *testing.T) {
	// A fired gate frees the fast body before the deep body speaks; the next think
	// frees the deep body before raising circles again. One body resident at a time.
	fast := &recordingBody{}
	deep := &recordingBody{answer: "deep"}
	iw := NewInnerWorld(fast, &fakeField{debt: 5.0}, tempDivergence)
	iw.SetDeep(deep)
	iw.SetLarynx(fixedLarynx{0.9})
	iw.SetRoll(func() float32 { return 0.0 }) // gate fires every turn

	<-iw.Think("q1")
	if _, fc := fast.counts(); fc < 1 {
		t.Errorf("fast body must be closed before the deep body speaks; closes=%d", fc)
	}

	<-iw.Think("q2") // ensureFastResident must close the deep body before fast runs
	if _, dc := deep.counts(); dc < 1 {
		t.Errorf("deep body must be closed when swapping back to fast; closes=%d", dc)
	}
}

func TestDeepSkipsEmptyCircles(t *testing.T) {
	// fast body returns nothing -> no circles -> even a fired gate must not swap or
	// wake the deep body on an empty stream (the empty-deepSeed fix).
	deep := &recordingBody{answer: "should never be generated"}
	iw := NewInnerWorld(emptyBody{}, &fakeField{debt: 5.0}, tempDivergence)
	iw.SetDeep(deep)
	iw.SetLarynx(fixedLarynx{0.9})
	iw.SetRoll(func() float32 { return 0.0 }) // gate would fire

	r := <-iw.Think("a question")
	if len(r.Circles) != 0 {
		t.Fatalf("empty body should yield no circles, got %d", len(r.Circles))
	}
	if r.DeepAnswer != "" {
		t.Errorf("deep must stay silent on an empty circle stream, got %q", r.DeepAnswer)
	}
	if gens, closes := deep.counts(); gens != 0 || closes != 0 {
		t.Errorf("deep body must not be touched on empty circles; gens=%d closes=%d", gens, closes)
	}
}
