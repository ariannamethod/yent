package tests

import (
	"math"
	"testing"

	yent "github.com/ariannamethod/yent/yent/go"
)

// TestRMSNorm verifies RMS normalization
func TestRMSNorm(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0, 4.0}
	w := []float32{1.0, 1.0, 1.0, 1.0}
	eps := float32(1e-6)

	// Calculate expected: RMS = sqrt(mean(x^2))
	var ss float64
	for _, v := range x {
		ss += float64(v * v)
	}
	rms := math.Sqrt(ss / float64(len(x)))
	expected := make([]float32, len(x))
	for i := range x {
		expected[i] = float32(float64(x[i]) / rms)
	}

	yent.RMSNorm(x, w, eps)

	for i := range x {
		if math.Abs(float64(x[i]-expected[i])) > 1e-5 {
			t.Errorf("RMSNorm[%d]: got %f, expected %f", i, x[i], expected[i])
		}
	}
}

// TestRMSNormInto verifies RMS normalization into separate buffer
func TestRMSNormInto(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0, 4.0}
	w := []float32{1.0, 1.0, 1.0, 1.0}
	out := make([]float32, 4)
	eps := float32(1e-6)

	var ss float64
	for _, v := range x {
		ss += float64(v * v)
	}
	rms := math.Sqrt(ss / float64(len(x)))
	expected := make([]float32, len(x))
	for i := range x {
		expected[i] = float32(float64(x[i]) / rms)
	}

	yent.RMSNormInto(out, x, w, eps)

	for i := range out {
		if math.Abs(float64(out[i]-expected[i])) > 1e-5 {
			t.Errorf("RMSNormInto[%d]: got %f, expected %f", i, out[i], expected[i])
		}
	}

	// Original should be unchanged
	if x[0] != 1.0 || x[1] != 2.0 || x[2] != 3.0 || x[3] != 4.0 {
		t.Error("RMSNormInto modified original array")
	}
}

// TestSoftmax verifies softmax computation
func TestSoftmax(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0, 4.0}
	yent.Softmax(x, 4)

	// Check sum = 1
	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum: got %f, expected 1.0", sum)
	}

	// Check monotonicity (larger input -> larger prob)
	for i := 0; i < len(x)-1; i++ {
		if x[i] >= x[i+1] {
			t.Errorf("Softmax not monotonic: x[%d]=%f >= x[%d]=%f", i, x[i], i+1, x[i+1])
		}
	}
}

// TestSiLU verifies SiLU activation
func TestSiLU(t *testing.T) {
	cases := []struct {
		x        float32
		expected float32
	}{
		{0, 0},
		{1, 0.7310586}, // 1 / (1 + e^-1) ≈ 0.731
		{-1, -0.2689414},
	}

	for _, c := range cases {
		got := yent.SiLU(c.x)
		if math.Abs(float64(got-c.expected)) > 1e-4 {
			t.Errorf("SiLU(%f): got %f, expected %f", c.x, got, c.expected)
		}
	}
}

// TestDequantQ4_0Block verifies Q4_0 block dequantization
func TestDequantQ4_0Block(t *testing.T) {
	// Create a test block: scale = 1.0, all values = 8 (which becomes 0 after -8)
	block := make([]byte, 18)
	// fp16 for 1.0: 0x3C00
	block[0] = 0x00
	block[1] = 0x3C
	// All nibbles = 8 (0x88 bytes)
	for i := 2; i < 18; i++ {
		block[i] = 0x88
	}

	out := make([]float32, 32)
	yent.DequantQ4_0Block(block, out)

	// All values should be 0 (since (8-8) * 1.0 = 0)
	for i, v := range out {
		if v != 0 {
			t.Errorf("DequantQ4_0Block[%d]: got %f, expected 0", i, v)
		}
	}
}

// TestMatMulF32 verifies F32 matrix multiplication
func TestMatMulF32(t *testing.T) {
	// Simple 2x3 @ 3 -> 2
	w := []float32{
		1, 2, 3, // row 0
		4, 5, 6, // row 1
	}
	x := []float32{1, 1, 1}
	out := make([]float32, 2)

	yent.MatMulF32(out, w, x, 2, 3)

	expected := []float32{6, 15} // 1+2+3, 4+5+6
	for i := range out {
		if math.Abs(float64(out[i]-expected[i])) > 1e-5 {
			t.Errorf("MatMulF32[%d]: got %f, expected %f", i, out[i], expected[i])
		}
	}
}
