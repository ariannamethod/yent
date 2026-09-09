package innerworld

import (
	"fmt"
	"strings"
	"sync"
	"testing"
)

// fakeBody encodes its seed origin and temperature into the thought, so the test
// drives a deterministic, monotonic divergence without a real model.
type fakeBody struct{}

func (fakeBody) Generate(seed string, temp float32) string {
	head := seed
	if i := strings.IndexByte(seed, ' '); i > 0 {
		head = seed[:i]
	}
	return fmt.Sprintf("thought<%s|t=%.2f>", head, temp)
}

// tempDivergence parses the temperature a thought ran at and treats hotter
// thoughts as further drift — a deterministic proxy for the production embedding
// distance. Reads the outermost t= (the circle's own temperature).
func tempDivergence(_, b string) float32 {
	var t float32
	if i := strings.LastIndex(b, "t="); i >= 0 {
		fmt.Sscanf(b[i+2:], "%f", &t)
	}
	return t / 2 // 0.7->0.35, 0.9->0.45, 1.1->0.55
}

// fakeField records the AML commands and physics steps the inner world drives.
// It is safe for concurrent use, as production Field implementations must be.
type fakeField struct {
	mu      sync.Mutex
	scripts []string
	steps   int
	debt    float32
	failOn  string
}

func (f *fakeField) Exec(s string) error {
	if f.failOn != "" && strings.Contains(s, f.failOn) {
		return fmt.Errorf("fake field rejected %q", s)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.scripts = append(f.scripts, s)
	return nil
}
func (f *fakeField) Step(dt float32) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.steps++
	f.debt += dt * 0.1
}
func (f *fakeField) Debt() float32 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.debt
}
func (f *fakeField) Destiny() float32 { return 0 }

func (f *fakeField) scriptList() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, len(f.scripts))
	copy(out, f.scripts)
	return out
}

func TestOverthinkCircles(t *testing.T) {
	const prompt = "what does it mean to exist as code"
	field := &fakeField{}
	circles := Overthink(prompt, fakeBody{}, field, tempDivergence, DefaultConfig())

	if len(circles) != 3 {
		t.Fatalf("want 3 circles, got %d", len(circles))
	}

	// drift increases: each circle drifts further from the last
	for i := 1; i < len(circles); i++ {
		if circles[i].Drift < circles[i-1].Drift {
			t.Errorf("drift not increasing at circle %d: %.3f < %.3f", i, circles[i].Drift, circles[i-1].Drift)
		}
	}

	// seed-from-prior: the water ripples out, each circle grows from the last
	for i := 1; i < len(circles); i++ {
		if circles[i].Seed != circles[i-1].Text {
			t.Errorf("circle %d seed %q != prior text %q", i, circles[i].Seed, circles[i-1].Text)
		}
	}

	// NO-SEED-FROM-PROMPT: circle 0 seeds from the inner transform, not the raw prompt
	if circles[0].Seed == prompt {
		t.Errorf("circle 0 seeded from raw prompt — NO-SEED violated")
	}
	if !strings.Contains(circles[0].Seed, prompt) {
		t.Errorf("circle 0 inner seed should carry the prompt, got %q", circles[0].Seed)
	}

	// field driven: each circle emits PROPHECY + VELOCITY and steps the physics
	scripts := field.scriptList()
	var prophecy, velocity int
	for _, s := range scripts {
		if strings.HasPrefix(s, "PROPHECY ") {
			prophecy++
		}
		if strings.HasPrefix(s, "VELOCITY ") {
			velocity++
		}
	}
	if prophecy != 3 || velocity != 3 {
		t.Errorf("want 3 PROPHECY + 3 VELOCITY execs, got %d + %d", prophecy, velocity)
	}
	if field.steps != 3 {
		t.Errorf("want 3 physics steps, got %d", field.steps)
	}

	// later circles run faster: VELOCITY RUN appears at circle 2
	if !containsStr(scripts, "VELOCITY RUN") {
		t.Errorf("circle 2 should drive VELOCITY RUN, scripts=%v", scripts)
	}
}

func containsStr(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

// monoBody / monoDiv produce drift that rises strictly with temperature, so the
// repel loop can always reach a target drift by heating up.
type monoBody struct{}

func (monoBody) Generate(_ string, temp float32) string {
	return fmt.Sprintf("t=%.4f", temp)
}

func monoDiv(_, b string) float32 {
	var t float32
	if i := strings.LastIndex(b, "t="); i >= 0 {
		fmt.Sscanf(b[i+2:], "%f", &t)
	}
	return t / 2
}

func TestNilSafe(t *testing.T) {
	cfg := DefaultConfig()
	if c := Overthink("q", nil, &fakeField{}, monoDiv, cfg); c != nil {
		t.Errorf("nil body should yield nil circles, got %d", len(c))
	}
	if c := Overthink("q", monoBody{}, &fakeField{}, nil, cfg); c != nil {
		t.Errorf("nil divergence should yield nil circles, got %d", len(c))
	}
	// nil field must not panic; circles still produced
	if c := Overthink("q", monoBody{}, nil, monoDiv, cfg); len(c) != cfg.N {
		t.Errorf("nil field should still produce %d circles, got %d", cfg.N, len(c))
	}
}

func TestRepelEnforcesDrift(t *testing.T) {
	// base temp 0.7 gives drift 0.35, below the target 0.5; the repel loop must
	// heat up until drift reaches at least the prior circle's drift.
	cfg := Config{TempBase: 0.7, RepelStep: 0.2, MaxRepel: 4}
	_, drift, temp := generateDivergent(monoBody{}, "seed", "prev", monoDiv, 0.7, 0.5, cfg)
	if drift < 0.5 {
		t.Errorf("repel should reach drift >= 0.5, got %.3f at temp %.2f", drift, temp)
	}
}

// emptyBody is a body that returns nothing — a timed-out or wedged voice.
type emptyBody struct{}

func (emptyBody) Generate(string, float32) string { return "" }

type afterwaveRecordingBody struct {
	calls int
	seeds []string
}

func (b *afterwaveRecordingBody) Generate(seed string, temp float32) string {
	b.calls++
	b.seeds = append(b.seeds, seed)
	return fmt.Sprintf("afterwave<t=%.2f>", temp)
}

func TestOverthinkEmptyStops(t *testing.T) {
	// a body that returns "" must not produce empty circles or drive the field —
	// the ripple ends instead of cascading garbage (the metal-run cascade fix).
	field := &fakeField{}
	c := Overthink("q", emptyBody{}, field, tempDivergence, DefaultConfig())
	if len(c) != 0 {
		t.Errorf("an empty body should produce no circles, got %d", len(c))
	}
	if scripts := field.scriptList(); len(scripts) != 0 {
		t.Errorf("an empty body must not drive the field, got %d commands", len(scripts))
	}
}

func TestAfterwaveIsOneUnforcedRippleFromSpokenAnswer(t *testing.T) {
	body := &afterwaveRecordingBody{}
	field := &fakeField{}
	circles := Afterwave("I already said this aloud.", body, field, tempDivergence, DefaultConfig())
	if len(circles) != 1 {
		t.Fatalf("afterwave circles = %d, want exactly one", len(circles))
	}
	if body.calls != 1 {
		t.Fatalf("afterwave generations = %d, want exactly one (no repel retries)", body.calls)
	}
	if strings.Contains(body.seeds[0], "Turn the question inward") {
		t.Fatalf("afterwave reused the pre-answer coercive seed: %q", body.seeds[0])
	}
	if !strings.Contains(body.seeds[0], "I already said this aloud.") {
		t.Fatalf("afterwave seed lost the spoken answer: %q", body.seeds[0])
	}
	if circles[0].Temp != DefaultConfig().TempBase {
		t.Fatalf("afterwave temp = %.2f, want base %.2f", circles[0].Temp, DefaultConfig().TempBase)
	}
}
