package yent

// amk.go — CGO bridge to AMK (Arianna Method Kernel)
//
// The DSL is the nervous system. Delta Voice is the mouth.
// Without the kernel, Yent is a voice without a brain.
//
// "from ariannamethod import Destiny"

/*
#cgo CFLAGS: -I${SRCDIR}/../c
#cgo LDFLAGS: ${SRCDIR}/../c/libamk.a
#include "amk_kernel.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

// AMK wraps the Arianna Method Kernel (C shared library)
type AMK struct {
	mu      sync.Mutex
	running bool
}

// AMState mirrors C AM_State — the breath of the field
type AMState struct {
	// Prophecy physics
	Prophecy      int
	Destiny       float32
	Wormhole      float32
	CalendarDrift float32

	// Attention
	AttendFocus  float32
	AttendSpread float32

	// Tunneling
	TunnelThreshold float32
	TunnelChance    float32
	TunnelSkipMax   int

	// Suffering
	Pain       float32
	Tension    float32
	Dissonance float32
	Debt       float32

	// Movement
	VelocityMode      int
	VelocityMagnitude float32
	BaseTemperature   float32
	EffectiveTemp     float32
	TimeDirection     float32

	// Wormhole
	WormholeActive int
}

// Pack flags
const (
	PackCodesRIC  = 0x01
	PackDarkMatter = 0x02
	PackNoTorch   = 0x04
)

// Velocity modes
const (
	VelNoMove   = 0
	VelWalk     = 1
	VelRun      = 2
	VelBackward = -1
)

// NewAMK initializes the kernel
func NewAMK() *AMK {
	C.am_init()
	return &AMK{running: true}
}

// Exec executes a DSL script
func (a *AMK) Exec(script string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	cs := C.CString(script)
	defer C.free(unsafe.Pointer(cs))

	ret := C.am_exec(cs)
	if ret != 0 {
		return fmt.Errorf("am_exec failed: %d", ret)
	}
	return nil
}

// ExecFile loads and executes a DSL script from file
func (a *AMK) ExecFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read DSL file: %w", err)
	}

	// Execute line by line (DSL is line-oriented)
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
			continue
		}
		if err := a.Exec(line); err != nil {
			return fmt.Errorf("DSL line %q: %w", line, err)
		}
	}
	return nil
}

// Step advances physics by dt seconds
func (a *AMK) Step(dt float32) {
	a.mu.Lock()
	defer a.mu.Unlock()
	C.am_step(C.float(dt))
}

// GetState reads current kernel state
func (a *AMK) GetState() AMState {
	a.mu.Lock()
	defer a.mu.Unlock()

	s := C.am_get_state()
	return AMState{
		Prophecy:          int(s.prophecy),
		Destiny:           float32(s.destiny),
		Wormhole:          float32(s.wormhole),
		CalendarDrift:     float32(s.calendar_drift),
		AttendFocus:       float32(s.attend_focus),
		AttendSpread:      float32(s.attend_spread),
		TunnelThreshold:   float32(s.tunnel_threshold),
		TunnelChance:      float32(s.tunnel_chance),
		TunnelSkipMax:     int(s.tunnel_skip_max),
		Pain:              float32(s.pain),
		Tension:           float32(s.tension),
		Dissonance:        float32(s.dissonance),
		Debt:              float32(s.debt),
		VelocityMode:      int(s.velocity_mode),
		VelocityMagnitude: float32(s.velocity_magnitude),
		BaseTemperature:   float32(s.base_temperature),
		EffectiveTemp:     float32(s.effective_temp),
		TimeDirection:     float32(s.time_direction),
		WormholeActive:    int(s.wormhole_active),
	}
}

// GetTemperature returns DSL-modulated temperature
func (a *AMK) GetTemperature() float32 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return float32(C.am_get_temperature())
}

// GetDestinyBias returns destiny bias for sampling
func (a *AMK) GetDestinyBias() float32 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return float32(C.am_get_destiny_bias())
}

// ShouldTunnel checks if tunneling should occur
func (a *AMK) ShouldTunnel() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return C.am_should_tunnel() != 0
}

// ApplySufferingToLogits modulates logits by pain/tension
func (a *AMK) ApplySufferingToLogits(logits []float32) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(logits) == 0 {
		return
	}
	C.am_apply_suffering_to_logits((*C.float)(unsafe.Pointer(&logits[0])), C.int(len(logits)))
}

// EnablePack enables a DSL extension pack
func (a *AMK) EnablePack(pack uint) {
	a.mu.Lock()
	defer a.mu.Unlock()
	C.am_enable_pack(C.uint(pack))
}

// DisablePack disables a DSL extension pack
func (a *AMK) DisablePack(pack uint) {
	a.mu.Lock()
	defer a.mu.Unlock()
	C.am_disable_pack(C.uint(pack))
}

// ResetField resets the field to defaults
func (a *AMK) ResetField() {
	a.mu.Lock()
	defer a.mu.Unlock()
	C.am_reset_field()
}

// ResetDebt resets accumulated debt
func (a *AMK) ResetDebt() {
	a.mu.Lock()
	defer a.mu.Unlock()
	C.am_reset_debt()
}
