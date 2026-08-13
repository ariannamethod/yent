package yent

import (
	"math"
	"strings"
	"testing"
)

func TestRuntimeConfigLayoutAcceptsValidShape(t *testing.T) {
	cfg := testRuntimeConfig()
	layout, err := runtimeConfigLayoutFor(cfg)
	if err != nil {
		t.Fatalf("runtimeConfigLayoutFor valid config: %v", err)
	}
	if layout.QRows != 2 || layout.KVRows != 2 || layout.AttElems != 4 ||
		layout.KVCacheElems != 8 || layout.RopeElems != 4 {
		t.Fatalf("unexpected layout: %+v", layout)
	}
}

func TestRuntimeConfigLayoutRejectsUnsafeGeometry(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name   string
		mutate func(*LlamaConfig)
		want   string
	}{
		{
			name: "odd head dim",
			mutate: func(cfg *LlamaConfig) {
				cfg.HeadDim = 3
			},
			want: "head_dim must be even",
		},
		{
			name: "q rows overflow",
			mutate: func(cfg *LlamaConfig) {
				cfg.NumHeads = maxInt/4 + 1
				cfg.HeadDim = 8
			},
			want: "attention q rows overflow",
		},
		{
			name: "attention buffer overflow",
			mutate: func(cfg *LlamaConfig) {
				cfg.NumHeads = maxInt/4 + 1
				cfg.HeadDim = 2
				cfg.SeqLen = 8
			},
			want: "attention score buffer overflow",
		},
		{
			name: "kv cache overflow",
			mutate: func(cfg *LlamaConfig) {
				cfg.NumLayers = maxInt/4 + 1
				cfg.SeqLen = 8
			},
			want: "kv cache layer/sequence overflow",
		},
		{
			name: "invalid rope theta",
			mutate: func(cfg *LlamaConfig) {
				cfg.RopeTheta = float32(math.Inf(1))
			},
			want: "rope_theta must be finite and positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testRuntimeConfig()
			tt.mutate(cfg)
			_, err := runtimeConfigLayoutFor(cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("runtimeConfigLayoutFor error = %v want containing %q", err, tt.want)
			}
		})
	}
}

func TestLoadLlamaModelRejectsRuntimeGeometryBeforeWeights(t *testing.T) {
	gguf := &GGUFFile{
		Meta: GGUFMetadata{
			Architecture: "llama",
			NumLayers:    1,
			EmbedDim:     3,
			NumHeads:     1,
			NumKVHeads:   1,
			HeadDim:      3,
			VocabSize:    2,
			SeqLen:       4,
			IntermSize:   4,
			RMSNormEps:   1e-5,
			RopeTheta:    10000,
		},
		Tensors: make(map[string]*GGUFTensorInfo),
	}
	_, err := LoadLlamaModel(gguf)
	if err == nil || !strings.Contains(err.Error(), "runtime config: head_dim must be even") {
		t.Fatalf("LoadLlamaModel error = %v want runtime config rejection before weight lookup", err)
	}
}

func testRuntimeConfig() *LlamaConfig {
	return &LlamaConfig{
		Architecture: "llama",
		NumLayers:    1,
		EmbedDim:     2,
		NumHeads:     1,
		NumKVHeads:   1,
		HeadDim:      2,
		VocabSize:    3,
		SeqLen:       4,
		IntermSize:   4,
		RMSNormEps:   1e-5,
		RopeTheta:    10000,
	}
}
