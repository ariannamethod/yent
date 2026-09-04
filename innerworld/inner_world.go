package innerworld

import (
	"context"
	"math/rand"
	"strings"
	"sync"
	"time"
)

// Autonomous breath triggers.
const (
	trigDrift   = iota // the field wandered: respond with a dream
	trigSilence        // no one has spoken for a while: the idle dreamer
	nTrig
)

// Breath tunes the autonomous inner life — when, and how often, the organism
// dreams unprompted.
type Breath struct {
	Tick      time.Duration        // how often the inner world is evaluated
	Silence   time.Duration        // idle time before the silence dreamer fires
	DriftDebt float32              // field debt above which the drift dreamer fires
	Cooldown  [nTrig]time.Duration // per-trigger cooldown so she breathes between dreams
}

// DefaultBreath is the Strike-1 pacing.
func DefaultBreath() Breath {
	return Breath{
		Tick:      time.Second,
		Silence:   4 * time.Second,
		DriftDebt: 1.0,
		Cooldown:  [nTrig]time.Duration{trigDrift: 3 * time.Second, trigSilence: 4 * time.Second},
	}
}

// Reflection is the full result of one inner monologue: the circles, the Larynx
// coupling, the deep-self-answer probability, whether the deep body turned inward
// this time, and — when it did — the deep body's actual inner answer. Inner only —
// none of it reaches the user.
type Reflection struct {
	Circles        []Circle
	Coupling       float32             // Larynx coupling over the circles [0,1]
	SelfAnswerProb float32             // gate probability the deep body answers itself [0,1]
	SelfAnswered   bool                // the unpredictable roll's outcome this time
	DeepAnswer     string              // small24's inner answer to the circles; empty unless SelfAnswered with a deep body
	MemoryPressure MemoryFieldPressure // slow pressure applied before circles; Score==0 means none
}

// Memory lets the inner world recall its own past monologues so new thinking is
// shaped by what it thought before. It is READ-ONLY here on purpose: the runtime
// persists reflections (the dock writes them to limpha), so the inner world only
// reads them back and the write path is never duplicated. nil = no recall.
type Memory interface {
	// Recall returns up to n recent inner thoughts, most recent first, as compact
	// text lines to fold into the next overthinking seed.
	Recall(n int) []string
}

// InnerWorld hosts Yent's inner life over the fast body, the shared AML field,
// and the Larynx membrane. Think runs the overthinking for a human turn off the
// answer path; Breathe keeps the organism dreaming between turns. Only one inner
// monologue runs at a time — the body has a single voice — so Think and the
// autonomous dream are serialized.
type InnerWorld struct {
	fast       Body
	deep       Body // the deep body (small24); nil = no deep self-answer, gate stays a boolean
	field      Field
	div        Divergence
	larynx     Larynx
	memory     Memory      // past-monologue recall; nil = no recall (read-only, runtime writes)
	cooc       *CoocGraph  // inner co-occurrence memory (Go form); nil = circles do not seed/feel a cooc field
	scar       *ScarMemory // sea of rejected thoughts (Go form); nil = no scarring
	flow       Flow        // native AML body (form A): when set, it IS the single cooc+scar+field physics
	sense      Sense       // environment perception (SARTRE): present-world reflex onto the field; nil = no environment
	metricSink MetricSink  // reciprocal telemetry: field weather -> SARTRE metrics hub; nil = no publish
	cfg        Config
	br         Breath

	scarThreshold float32  // prophecy-debt above which a thought is scarred (rejected by the field)
	feelEnabled   bool     // High brain: when on, the circles' emotional valence drives the affect axis
	feelIntensity float32  // last turn's emotional intensity (guarded by genMu); pulls up resonant scars
	feelMath      FeelMath // feeling-math backend (real Julia in production); nil = Go lexical proxy

	genMu        sync.Mutex // one inner voice at a time: serializes Overthink + deep self-answer (body access)
	deepResident bool       // guarded by genMu: the deep body is the currently-resident one (single-resident swap)

	mu           sync.Mutex // guards circles, lastActive, lastFire, onDream, larynx, roll, asleep, sleepLatched
	circles      []Circle
	lastActive   time.Time
	lastFire     [nTrig]time.Time // each trigger's refractory is measured from the last autonomous speech
	onDream      func(Reflection)
	roll         func() float32 // [0,1) draw for the deep-self-answer gate
	asleep       bool           // guarded by mu: the organism is in the consolidation sleep
	sleepLatched bool           // guarded by mu: sleep ran for the current critical-mass episode

	// Dreaming (Level B skeleton): when the field reaches critical mass the organism
	// sleeps and runs its consolidators in order. sleepTrigger reports critical mass;
	// nil = never sleeps (backward-compatible). Consolidation stages (cooc, weights,
	// scar, emotion) plug into consolidators; the skeleton only sequences them.
	sleepTrigger  SleepTrigger
	consolidators []Consolidation
	onSleep       func(stage string) // optional observer of consolidation stages (inner only)
}

// NewInnerWorld builds the inner world over a fast body, the shared field, and a
// divergence measure. The Larynx defaults to the portable Go membrane and the
// gate rolls a real random; both can be overridden for tests.
func NewInnerWorld(fast Body, field Field, div Divergence) *InnerWorld {
	return &InnerWorld{
		fast:       fast,
		field:      field,
		div:        div,
		larynx:     textureLarynx{},
		cfg:        DefaultConfig(),
		br:         DefaultBreath(),
		roll:       func() float32 { return rand.Float32() },
		lastActive: time.Now(),
	}
}

// SetDeep wires the deep body (small24). With a deep body, a fired self-answer
// gate makes the deep body actually generate an inner answer to the circles;
// without one, the gate stays a boolean. Set before Think/Breathe start.
func (iw *InnerWorld) SetDeep(deep Body) {
	iw.genMu.Lock()
	iw.deep = deep
	iw.genMu.Unlock()
}

// SetMemory wires the past-monologue recall, so new thinking is shaped by what the
// organism thought before. Read-only: the runtime persists reflections; this only
// reads them back. Set before Think/Breathe start.
func (iw *InnerWorld) SetMemory(m Memory) {
	iw.genMu.Lock()
	iw.memory = m
	iw.genMu.Unlock()
}

// SetFlow wires the native AML body as the single inner-world physics (form A): one
// organism holds the cooc graph, the scar sea, and the field. With it, the circles
// ingest into the field's own cooc (am_ingest_tokens), high-debt thoughts scar
// natively (the SCAR operator), the seed is pulled by the field's cooc and resurfaced
// scars, and a FlowConsolidator harvests in sleep — no parallel Go cooc/scar. When a
// flow is set it takes precedence over SetCooc/SetScar. Set before Think/Breathe start.
func (iw *InnerWorld) SetFlow(f Flow) {
	iw.genMu.Lock()
	iw.flow = f
	iw.genMu.Unlock()
}

// SetScarThreshold sets the prophecy-debt above which a thought is scarred, for the
// flow path (the Go path sets it through SetScar). Set before Think/Breathe start.
func (iw *InnerWorld) SetScarThreshold(t float32) {
	iw.genMu.Lock()
	iw.scarThreshold = t
	iw.genMu.Unlock()
}

// recallSeed folds recent inner monologues into the prompt before a new ripple, so
// the organism thinks with what it thought before. NO-SEED-FROM-PROMPT still holds:
// the recall and prompt are transformed by innerSeed inside Overthink before circle
// 0. Caller holds genMu. No memory, no recalls, or RecallN<=0 returns the prompt
// unchanged (backward-compatible). Past text is framed as pressure/trace, not as
// dialogue to continue; raw recall otherwise overheats into imitation loops.
func (iw *InnerWorld) recallSeed(prompt string) string {
	return iw.recallSeedWithTraces(prompt, iw.memoryTracesLocked())
}

// closeIfResident closes a body that owns a resident process (the doe daemon), so
// single-resident hosts never hold two bodies' weights at once. A body that does
// not implement Close (a fake, a pure-Go voice) is left untouched.
func closeIfResident(b Body) {
	if c, ok := b.(interface{ Close() error }); ok {
		_ = c.Close()
	}
}

// ensureFastResidentLocked swaps back to the fast body before raising circles, if
// the deep body was left resident by a prior self-answer. Caller holds genMu.
func (iw *InnerWorld) ensureFastResidentLocked() {
	if iw.deepResident && iw.deep != nil {
		closeIfResident(iw.deep)
		iw.deepResident = false
	}
}

// deepAnswerLocked has the deep body answer the circles, once the gate has fired.
// Single-resident: the fast body is freed before the deep body speaks. Caller
// holds genMu, so fast and deep never run at once. Returns "" if no deep body.
func (iw *InnerWorld) deepAnswerLocked(circles []Circle) string {
	if iw.deep == nil {
		return ""
	}
	seed := deepSeed(circles)
	if seed == "" {
		return "" // no inner thought to answer — do not swap residents or wake the deep body
	}
	closeIfResident(iw.fast) // free the fast weights before the deep body loads
	iw.deepResident = true
	return iw.deep.Generate(seed, 0)
}

// deepSeed is what crosses the membrane to the deep body: the fast body's stream
// of thought, the circles joined in order — never the raw user prompt
// (NO-SEED-FROM-PROMPT). The deep body answers the inner field, not the surface.
func deepSeed(circles []Circle) string {
	var b strings.Builder
	for i, c := range circles {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(c.Text)
	}
	return strings.TrimSpace(b.String())
}

// SetOnDream registers the observer for autonomous dreams. Inner only — the
// reflection handed to it is a copy and never reaches the user.
func (iw *InnerWorld) SetOnDream(f func(Reflection)) {
	iw.mu.Lock()
	iw.onDream = f
	iw.mu.Unlock()
}

// SetLarynx overrides the membrane (the Zig binding in production, a fake in tests).
func (iw *InnerWorld) SetLarynx(l Larynx) {
	iw.mu.Lock()
	iw.larynx = l
	iw.mu.Unlock()
}

// SetRoll overrides the gate's random draw so the deep-self-answer decision is
// deterministic in tests.
func (iw *InnerWorld) SetRoll(f func() float32) {
	iw.mu.Lock()
	iw.roll = f
	iw.mu.Unlock()
}

// SetBreath overrides the autonomous-breath pacing (tick, idle, cooldowns).
func (iw *InnerWorld) SetBreath(b Breath) {
	iw.mu.Lock()
	iw.br = b
	iw.mu.Unlock()
}

func cloneCircles(c []Circle) []Circle {
	if c == nil {
		return nil
	}
	out := make([]Circle, len(c))
	copy(out, c)
	return out
}

// fieldDebt reads the field's debt, or 0 when there is no field.
func (iw *InnerWorld) fieldDebt() float32 {
	if iw.field == nil {
		return 0
	}
	return iw.field.Debt()
}

// reflect turns the circles into a full inner reflection: the Larynx coupling,
// the gate probability, and the unpredictable self-answer decision. The field
// debt is snapshotted by the caller under genMu, so the probability belongs to
// the batch that just drove the field — not to an interleaved one.
func (iw *InnerWorld) reflect(circles []Circle, debt float32) Reflection {
	iw.mu.Lock()
	larynx := iw.larynx
	roll := iw.roll
	iw.mu.Unlock()

	coupling := larynx.Couple(circles)
	var drift float32
	if n := len(circles); n > 0 {
		drift = circles[n-1].Drift
	}
	prob := DeepGate(debt, drift, coupling)
	return Reflection{
		Circles:        circles,
		Coupling:       coupling,
		SelfAnswerProb: prob,
		SelfAnswered:   SelfAnswers(prob, roll()),
	}
}

// think runs one overthinking pass under the single-voice lock, reflects on it,
// and stores a copy of the circles.
func (iw *InnerWorld) think(prompt string) Reflection {
	iw.genMu.Lock()
	iw.ensureFastResidentLocked()
	traces := iw.memoryTracesLocked()
	memoryPressure, _ := iw.applyMemoryPressureLocked(traces) // the past as slow field pressure
	iw.applySenseLocked()                                     // the present world as fast field reflex (SARTRE)
	circles := Overthink(iw.recallSeedWithTraces(iw.coocBias(iw.scarSurface(prompt)), traces), iw.fast, iw.field, iw.div, iw.cfg)
	debt := iw.fieldDebt()                                               // snapshot under genMu: belongs to this batch
	iw.observeLocked(circles)                                            // circles seed the cooc field (circles->field)
	iw.scarLocked(circles, debt)                                         // a thought that broke prophecy becomes a scar
	iw.highFeelLocked(circles)                                           // the circles' feeling drives the affect axis (High brain)
	iw.publishMetricsLocked("human_turn", circles, debt, memoryPressure) // reciprocal field-weather telemetry
	r := iw.reflect(circles, debt)
	r.MemoryPressure = memoryPressure
	if r.SelfAnswered {
		r.DeepAnswer = iw.deepAnswerLocked(circles) // deep body speaks, under the single voice
	}
	iw.genMu.Unlock()

	iw.mu.Lock()
	iw.circles = cloneCircles(circles)
	iw.mu.Unlock()
	r.Circles = cloneCircles(circles)
	return r
}

// Think runs the overthinking for a human turn asynchronously: it returns at once
// with a channel that delivers the reflection (a copy of the circles plus the
// coupling and gate decision) when ready, so the answer path is never blocked.
func (iw *InnerWorld) Think(prompt string) <-chan Reflection {
	out := make(chan Reflection, 1)
	iw.mu.Lock()
	iw.lastActive = time.Now()
	iw.mu.Unlock()
	go func() {
		out <- iw.think(prompt)
		close(out)
	}()
	return out
}

// due reports which autonomous trigger, if any, should fire at time now. The
// most-overdue eligible trigger wins, so a steadily high-debt field can never
// starve the silence dreamer.
func (iw *InnerWorld) due(now time.Time) (int, bool) {
	iw.mu.Lock()
	defer iw.mu.Unlock()

	best, bestOver := -1, time.Duration(0)
	consider := func(trig int, eligible bool) {
		if !eligible {
			return
		}
		over := now.Sub(iw.lastFire[trig]) - iw.br.Cooldown[trig]
		if over < 0 {
			return // still on cooldown
		}
		if best < 0 || over > bestOver {
			best, bestOver = trig, over
		}
	}
	consider(trigDrift, iw.field != nil && iw.field.Debt() >= iw.br.DriftDebt)
	consider(trigSilence, now.Sub(iw.lastActive) >= iw.br.Silence)

	if best < 0 {
		return 0, false
	}
	return best, true
}

// dream runs the overthinking unprompted, on the organism's own last thought —
// the ripple continues with no one speaking. When it completes, every autonomous
// trigger enters its own configured refractory: drift and silence are different
// reasons to speak, not two voices entitled to answer back-to-back. Inner only;
// OnDream receives a copy.
func (iw *InnerWorld) dream(trigger int) Reflection {
	iw.mu.Lock()
	seed := "I keep thinking, with no one here."
	if n := len(iw.circles); n > 0 {
		seed = iw.circles[n-1].Text
	}
	iw.mu.Unlock()

	iw.genMu.Lock()
	iw.ensureFastResidentLocked()
	traces := iw.memoryTracesLocked()
	memoryPressure, _ := iw.applyMemoryPressureLocked(traces) // the past as slow field pressure
	iw.applySenseLocked()                                     // the present world as fast field reflex (SARTRE)
	circles := Overthink(iw.recallSeedWithTraces(iw.coocBias(iw.scarSurface(seed)), traces), iw.fast, iw.field, iw.div, iw.cfg)
	debt := iw.fieldDebt()                                          // snapshot under genMu: belongs to this batch
	iw.observeLocked(circles)                                       // dreams seed the cooc field too (circles->field)
	iw.scarLocked(circles, debt)                                    // a dissonant dream scars too
	iw.highFeelLocked(circles)                                      // even alone, a dream colours the mood (High brain)
	iw.publishMetricsLocked("dream", circles, debt, memoryPressure) // reciprocal field-weather telemetry
	r := iw.reflect(circles, debt)
	r.MemoryPressure = memoryPressure
	if r.SelfAnswered {
		r.DeepAnswer = iw.deepAnswerLocked(circles) // even alone, the deep body may answer the dream
	}
	iw.genMu.Unlock()

	iw.mu.Lock()
	iw.circles = cloneCircles(circles)
	completed := time.Now()
	for trig := range iw.lastFire {
		iw.lastFire[trig] = completed // one voice: any dream paces every trigger
	}
	onDream := iw.onDream
	iw.mu.Unlock()

	r.Circles = cloneCircles(circles)
	if onDream != nil {
		onDream(r)
	}
	return r
}

// Breathe runs the autonomous inner life until ctx is cancelled. Between human
// turns the field keeps drifting, and when a trigger crosses its threshold the
// organism dreams unprompted. She is never muted, only paced.
func (iw *InnerWorld) Breathe(ctx context.Context) {
	iw.mu.Lock()
	tick := iw.br.Tick
	iw.mu.Unlock()
	if tick <= 0 {
		tick = time.Second // guard: a non-positive tick would panic time.NewTicker
	}
	t := time.NewTicker(tick)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-t.C:
			// Critical mass takes priority over dreaming: when the field is full, the
			// organism sleeps and consolidates instead of raising another circle. A
			// single critical-mass episode sleeps once; it re-arms only after the
			// trigger falls false, so a long autumn does not harvest every tick.
			critical := iw.criticalMass()
			if iw.readyToSleep(critical) {
				iw.sleep(ctx)
				select {
				case <-ctx.Done():
					return
				default:
				}
			} else if !critical {
				if trigger, ok := iw.due(now); ok {
					iw.dream(trigger)
					select {
					case <-ctx.Done():
						return
					default:
					}
				}
			}
			// pick up a tick changed by SetBreath while breathing
			iw.mu.Lock()
			nt := iw.br.Tick
			iw.mu.Unlock()
			if nt > 0 && nt != tick {
				tick = nt
				t.Reset(nt)
			}
		}
	}
}
