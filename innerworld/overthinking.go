// Package innerworld is Yent's inner life — the layer that continues around and
// after outward speech. It retains the original multi-circle Overthink primitive
// for autonomous/development paths and provides a single unforced Afterwave for
// response-first live conversation.
//
// The package is pure logic over two interfaces — Body (an inference voice) and
// Field (the shared AML physics). Production wires the real fast body and the
// AML kernel; tests wire fakes. No cgo lives here.
package innerworld

import (
	"fmt"
	"strings"
)

// Body is one inference voice. Strike 1 uses the fast mouth (nemo12). Real on
// Metal; a fake in tests.
type Body interface {
	// Generate produces an inner thought from a seed at the given temperature.
	Generate(seed string, temp float32) string
}

// Field is the shared AML physics (a wrapper over yent.AMK in production; a fake
// in tests). The inner world drives it with AML commands and reads the breath
// back; it never owns a private field — one organism, one field. Implementations
// MUST be safe for concurrent use (the real yent.AMK locks internally).
type Field interface {
	Exec(script string) error // run an AML command, e.g. "PROPHECY 7", "VELOCITY RUN"
	Step(dt float32)          // advance the physics
	Debt() float32            // prophecy debt accumulator
	Destiny() float32         // bias toward the most-probable path
}

// Divergence measures how far b drifts from a, semantically and thematically,
// in [0,1] (0 = identical, 1 = unrelated). Production combines an embedding
// cosine distance with a topic shift; tests use a token-distance proxy.
type Divergence func(a, b string) float32

// Circle is one inner thought — never user-facing.
type Circle struct {
	Index int     // 0..N-1
	Seed  string  // what it grew from: the prior circle, or the inner seed for circle 0
	Text  string  // the thought
	Drift float32 // divergence from the previous circle, [0,1]; non-decreasing per circle
	Temp  float32 // temperature it ran at (after any repel)
}

// Config tunes the overthinking.
type Config struct {
	N         int     // number of circles
	TempBase  float32 // temperature of the first circle
	TempRamp  float32 // temperature added per circle — each circle hotter, so it drifts further
	RepelStep float32 // extra temperature per repel retry when a circle did not drift further
	MaxRepel  int     // max repel retries to enforce monotonic drift
	RecallN   int     // how many past inner monologues to fold into the seed (0 = none)
}

// DefaultConfig is the Strike-1 default: three circles, warming as they ripple out,
// recalling up to three past inner monologues.
func DefaultConfig() Config {
	return Config{N: 3, TempBase: 0.7, TempRamp: 0.2, RepelStep: 0.15, MaxRepel: 3, RecallN: 3}
}

// innerSeed turns the user prompt into an internal seed. This is
// NO-SEED-FROM-PROMPT: the deep body never sees the raw prompt, only the circles
// that grew from this transformed seed. The organism answers the question's
// pattern, not its words.
func innerSeed(prompt string) string {
	return "Turn the question inward, do not answer its surface: " + prompt
}

// Overthink raises the circles on the fast body, drives the field by each
// circle's drift, and returns the circles — the inner monologue the deep body
// (and the field) will consume. The circles are inner; nothing here is shown to
// the user. A nil fast body or nil divergence yields no circles (no panic).
func Overthink(prompt string, fast Body, field Field, div Divergence, cfg Config) []Circle {
	return ripple(innerSeed(prompt), prompt, fast, field, div, cfg, true)
}

// Afterwave raises one private ripple from an answer that has already been
// spoken. Unlike pre-answer Overthink, it neither demands increasing distance
// nor retries a generation to manufacture it: drift is observed, not enforced.
// The caller is responsible for running this off the outward-answer path.
func Afterwave(spoken string, fast Body, field Field, div Divergence, cfg Config) []Circle {
	spoken = strings.TrimSpace(spoken)
	if spoken == "" {
		return nil
	}
	if cfg.N <= 0 {
		cfg = DefaultConfig()
	}
	cfg.N = 1
	cfg.TempRamp = 0
	cfg.MaxRepel = 0
	seed := "[private afterwave; outward speech already delivered]\n" + spoken
	return ripple(seed, spoken, fast, field, div, cfg, false)
}

func ripple(seed, previous string, fast Body, field Field, div Divergence, cfg Config, repel bool) []Circle {
	if fast == nil || div == nil {
		return nil
	}
	if cfg.N <= 0 {
		cfg = DefaultConfig()
	}
	circles := make([]Circle, 0, cfg.N)

	prev := previous         // circle 0 drifts measured from the originating text
	prevDrift := float32(-1) // circle 0 has no monotonic constraint
	for i := 0; i < cfg.N; i++ {
		baseTemp := cfg.TempBase + cfg.TempRamp*float32(i)
		text, drift, temp := "", float32(0), baseTemp
		if repel {
			text, drift, temp = generateDivergent(fast, seed, prev, div, baseTemp, prevDrift, cfg)
		} else {
			text = fast.Generate(seed, baseTemp)
			drift = div(prev, text)
		}

		// a body that returns nothing (timeout/error) stops the ripple — do not
		// append empty circles, do not drive the field with garbage, do not seed the
		// next circle from "". The chain ends with whatever real thoughts came before.
		if strings.TrimSpace(text) == "" {
			break
		}

		circles = append(circles, Circle{Index: i, Seed: seed, Text: text, Drift: drift, Temp: temp})
		if !driveField(field, drift, i) {
			// a broken field stops driving the physics; the inner monologue still
			// runs, but we do not step a field whose commands are failing.
			field = nil
		}

		// the water ripples out: each circle seeds the next from itself, not the prompt.
		seed = text
		prev = text
		prevDrift = drift
	}
	return circles
}

// generateDivergent produces a thought that drifts at least as far from prev as
// the previous circle did. If the raw generation did not drift further, it repels
// — pushes the temperature up and retries — keeping the furthest attempt. This
// makes "each circle drifts further than the last" a guarantee, not a hope.
func generateDivergent(fast Body, seed, prev string, div Divergence, baseTemp, prevDrift float32, cfg Config) (string, float32, float32) {
	bestText, bestDrift, bestTemp := "", float32(-1), baseTemp
	temp := baseTemp
	for r := 0; r <= cfg.MaxRepel; r++ {
		text := fast.Generate(seed, temp)
		drift := div(prev, text)
		if drift > bestDrift {
			bestText, bestDrift, bestTemp = text, drift, temp
		}
		if drift >= prevDrift {
			break // already drifted at least as far as the last circle
		}
		temp += cfg.RepelStep // repel: hotter, to push the thought further out
	}
	return bestText, bestDrift, bestTemp
}

// driveField turns a circle's drift into AML field commands. The further a
// thought wandered, the deeper the prophecy horizon; later circles move faster.
// Returns false if a field command failed (the caller stops driving the field).
func driveField(field Field, drift float32, i int) bool {
	if field == nil {
		return true
	}
	prophecy := 1 + int(drift*7) // 1..8
	if prophecy > 64 {
		prophecy = 64
	}
	vel := "WALK"
	if i >= 2 {
		vel = "RUN"
	}
	if err := field.Exec(fmt.Sprintf("PROPHECY %d", prophecy)); err != nil {
		return false
	}
	if err := field.Exec("VELOCITY " + vel); err != nil {
		return false
	}
	field.Step(1.0)
	return true
}
