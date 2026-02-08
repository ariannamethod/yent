package tests

import (
	"math"
	"testing"

	yent "github.com/ariannamethod/yent/yent/go"
)

// TestAMKInit verifies kernel initialization defaults
func TestAMKInit(t *testing.T) {
	amk := yent.NewAMK()
	if amk == nil {
		t.Fatal("NewAMK returned nil")
	}
	s := amk.GetState()
	// Default prophecy = 7
	if s.Prophecy != 7 {
		t.Errorf("default prophecy: got %d, expected 7", s.Prophecy)
	}
	// Default base temp = 1.0, velocity = WALK, effective = 1.0 * 0.85 = 0.85
	if math.Abs(float64(s.BaseTemperature-1.0)) > 0.01 {
		t.Errorf("default base_temp: got %.3f, expected 1.0", s.BaseTemperature)
	}
	if math.Abs(float64(s.EffectiveTemp-0.85)) > 0.01 {
		t.Errorf("default effective_temp: got %.3f, expected 0.85 (WALK)", s.EffectiveTemp)
	}
	if s.VelocityMode != yent.VelWalk {
		t.Errorf("default velocity: got %d, expected %d (WALK)", s.VelocityMode, yent.VelWalk)
	}
}

// TestAMKExecProphecy tests PROPHECY DSL command
func TestAMKExecProphecy(t *testing.T) {
	amk := yent.NewAMK()
	if err := amk.Exec("PROPHECY 13"); err != nil {
		t.Fatalf("Exec PROPHECY: %v", err)
	}
	s := amk.GetState()
	if s.Prophecy != 13 {
		t.Errorf("prophecy after set: got %d, expected 13", s.Prophecy)
	}
}

// TestAMKExecDestiny tests DESTINY DSL command
func TestAMKExecDestiny(t *testing.T) {
	amk := yent.NewAMK()
	if err := amk.Exec("DESTINY 0.7"); err != nil {
		t.Fatalf("Exec DESTINY: %v", err)
	}
	s := amk.GetState()
	if math.Abs(float64(s.Destiny-0.7)) > 0.01 {
		t.Errorf("destiny after set: got %.3f, expected 0.700", s.Destiny)
	}
}

// TestAMKVelocityRun tests VELOCITY RUN -> higher effective temperature
func TestAMKVelocityRun(t *testing.T) {
	amk := yent.NewAMK()
	if err := amk.Exec("VELOCITY RUN"); err != nil {
		t.Fatalf("Exec VELOCITY RUN: %v", err)
	}
	amk.Step(0.1)
	s := amk.GetState()
	if s.VelocityMode != yent.VelRun {
		t.Errorf("velocity mode: got %d, expected %d (RUN)", s.VelocityMode, yent.VelRun)
	}
	// RUN: effective = base * 1.2 = 1.0 * 1.2 = 1.2
	if s.EffectiveTemp < 1.0 {
		t.Errorf("RUN effective_temp: got %.3f, expected >= 1.0", s.EffectiveTemp)
	}
}

// TestAMKVelocityNoMove tests VELOCITY NOMOVE -> low temperature
func TestAMKVelocityNoMove(t *testing.T) {
	amk := yent.NewAMK()
	if err := amk.Exec("VELOCITY NOMOVE"); err != nil {
		t.Fatalf("Exec VELOCITY NOMOVE: %v", err)
	}
	amk.Step(0.1)
	s := amk.GetState()
	if s.VelocityMode != yent.VelNoMove {
		t.Errorf("velocity mode: got %d, expected %d (NOMOVE)", s.VelocityMode, yent.VelNoMove)
	}
	// NOMOVE: effective = base * 0.5 = 1.0 * 0.5 = 0.5
	if s.EffectiveTemp > 0.6 {
		t.Errorf("NOMOVE effective_temp: got %.3f, expected <= 0.6", s.EffectiveTemp)
	}
}

// TestAMKStepDebtDecay verifies debt decays over physics steps
func TestAMKStepDebtDecay(t *testing.T) {
	amk := yent.NewAMK()
	// Inject debt directly via DSL
	if err := amk.Exec("PROPHECY_DEBT 5.0"); err != nil {
		t.Fatalf("Exec PROPHECY_DEBT: %v", err)
	}
	s1 := amk.GetState()
	if s1.Debt < 4.0 {
		t.Fatalf("debt before stepping: got %.3f, expected ~5.0", s1.Debt)
	}
	// Step many times — debt should decay (debt_decay = 0.998 per step)
	for i := 0; i < 200; i++ {
		amk.Step(0.05)
	}
	s2 := amk.GetState()
	if s2.Debt >= s1.Debt {
		t.Errorf("debt should decay: before=%.3f after=%.3f", s1.Debt, s2.Debt)
	}
}

// TestAMKSufferingLogits verifies suffering modulates logits
func TestAMKSufferingLogits(t *testing.T) {
	amk := yent.NewAMK()
	// Set high pain + tension
	if err := amk.Exec("PAIN 0.8"); err != nil {
		t.Fatalf("Exec PAIN: %v", err)
	}
	if err := amk.Exec("TENSION 0.6"); err != nil {
		t.Fatalf("Exec TENSION: %v", err)
	}

	logits := []float32{10.0, -5.0, 3.0, -8.0, 1.0}
	original := make([]float32, len(logits))
	copy(original, logits)

	amk.ApplySufferingToLogits(logits)

	// With high pain+tension, logits should be dampened (multiplied by factor < 1)
	dampened := false
	for i := range logits {
		if math.Abs(float64(logits[i]-original[i])) > 0.01 {
			dampened = true
			break
		}
	}
	if !dampened {
		t.Error("suffering should dampen logits but nothing changed")
	}

	// Dampened logits should be closer to zero (absolute value decreases)
	for i := range logits {
		if math.Abs(float64(logits[i])) > math.Abs(float64(original[i]))+0.01 {
			t.Errorf("logit[%d] got further from zero: %.3f -> %.3f", i, original[i], logits[i])
		}
	}
}

// TestAMKResetField verifies field reset clears suffering but preserves config
func TestAMKResetField(t *testing.T) {
	amk := yent.NewAMK()
	amk.Exec("PROPHECY 42")
	amk.Exec("PAIN 0.8")
	amk.Exec("TENSION 0.5")

	amk.ResetField()
	s := amk.GetState()

	// reset_field clears suffering/debt
	if math.Abs(float64(s.Pain)) > 0.01 {
		t.Errorf("after reset pain: got %.3f, expected 0", s.Pain)
	}
	if math.Abs(float64(s.Tension)) > 0.01 {
		t.Errorf("after reset tension: got %.3f, expected 0", s.Tension)
	}

	// reset_field preserves config (prophecy, destiny, velocity)
	if s.Prophecy != 42 {
		t.Errorf("reset should preserve prophecy: got %d, expected 42", s.Prophecy)
	}
}

// TestAMKGetTemperature verifies temperature accessor
func TestAMKGetTemperature(t *testing.T) {
	amk := yent.NewAMK()
	temp := amk.GetTemperature()
	// Default: WALK mode, base=1.0, effective=0.85
	if temp <= 0 || temp > 2.0 {
		t.Errorf("temperature out of range: %.3f", temp)
	}
	if math.Abs(float64(temp-0.85)) > 0.01 {
		t.Errorf("default temperature: got %.3f, expected 0.85", temp)
	}
}

// TestAMKGetDestinyBias verifies destiny bias accessor
func TestAMKGetDestinyBias(t *testing.T) {
	amk := yent.NewAMK()
	amk.Exec("DESTINY 0.35")
	bias := amk.GetDestinyBias()
	if math.Abs(float64(bias-0.35)) > 0.01 {
		t.Errorf("destiny bias: got %.3f, expected 0.350", bias)
	}
}

// TestAMKBaseTemp tests BASE_TEMP DSL command
func TestAMKBaseTemp(t *testing.T) {
	amk := yent.NewAMK()
	if err := amk.Exec("BASE_TEMP 1.5"); err != nil {
		t.Fatalf("Exec BASE_TEMP: %v", err)
	}
	s := amk.GetState()
	if math.Abs(float64(s.BaseTemperature-1.5)) > 0.01 {
		t.Errorf("base_temp: got %.3f, expected 1.5", s.BaseTemperature)
	}
	// WALK effective = 1.5 * 0.85 = 1.275
	if math.Abs(float64(s.EffectiveTemp-1.275)) > 0.02 {
		t.Errorf("effective_temp: got %.3f, expected ~1.275", s.EffectiveTemp)
	}
}

// TestAMKEnableDisablePack verifies pack management
func TestAMKEnableDisablePack(t *testing.T) {
	amk := yent.NewAMK()
	if err := amk.Exec("IMPORT NOTORCH"); err != nil {
		t.Fatalf("Exec IMPORT NOTORCH: %v", err)
	}
	amk.DisablePack(yent.PackNoTorch)
	// No panic, no error — pack state is internal
}
