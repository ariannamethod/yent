package yent

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

func testSamplerYent(logits []float32, vocab int) *Yent {
	if vocab == 0 {
		vocab = len(logits)
	}
	return &Yent{
		model: &LlamaModel{
			Config: LlamaConfig{VocabSize: vocab},
			State:  LlamaState{Logits: logits},
		},
		rng: rand.New(rand.NewSource(1)),
	}
}

func TestSampleNextTokenRejectsInvalidSamplerSurface(t *testing.T) {
	tests := []struct {
		name string
		y    *Yent
		temp float32
		topP float32
		want string
	}{
		{
			name: "short logits",
			y:    testSamplerYent([]float32{1, 2}, 3),
			temp: 0.7,
			topP: 1,
			want: "logit buffer too small",
		},
		{
			name: "no finite logits",
			y:    testSamplerYent([]float32{float32(math.NaN()), float32(math.Inf(-1))}, 0),
			temp: 0.7,
			topP: 1,
			want: "no finite logits",
		},
		{
			name: "nan temperature",
			y:    testSamplerYent([]float32{1}, 0),
			temp: float32(math.NaN()),
			topP: 1,
			want: "temperature must be finite",
		},
		{
			name: "bad top p",
			y:    testSamplerYent([]float32{1}, 0),
			temp: 0.7,
			topP: 0,
			want: "top_p must be finite and positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.y.sampleNextToken(tt.temp, tt.topP, 5)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("sampleNextToken token=%d error=%v want %q", got, err, tt.want)
			}
		})
	}
}

func TestSampleNextTokenSkipsNonFiniteLogits(t *testing.T) {
	y := testSamplerYent([]float32{
		float32(math.NaN()),
		2,
		float32(math.Inf(1)),
		1,
	}, 0)

	for _, topP := range []float32{1, 0.9} {
		t.Run("top_p", func(t *testing.T) {
			got, err := y.sampleNextToken(0.7, topP, 4)
			if err != nil {
				t.Fatalf("sampleNextToken topP=%g: %v", topP, err)
			}
			if got != 1 && got != 3 {
				t.Fatalf("sampleNextToken topP=%g picked non-finite token %d", topP, got)
			}
		})
	}
}

func TestSampleTopKBadInputsFallBackToFiniteArgmax(t *testing.T) {
	y := testSamplerYent([]float32{1, float32(math.NaN()), 3}, 0)
	if got := y.sampleTopK(0.7, 0); got != 2 {
		t.Fatalf("sampleTopK topK=0 = %d want finite argmax 2", got)
	}
	if got := y.sampleTopK(float32(math.NaN()), 2); got != 2 {
		t.Fatalf("sampleTopK NaN temp = %d want finite argmax 2", got)
	}
}

func TestSampleTopPAllNonFiniteReturnsInvalidToken(t *testing.T) {
	y := testSamplerYent([]float32{float32(math.NaN()), float32(math.Inf(1))}, 0)
	if got := y.sampleTopP(0.8, 0.9); got != -1 {
		t.Fatalf("sampleTopP all non-finite = %d want -1", got)
	}
}
