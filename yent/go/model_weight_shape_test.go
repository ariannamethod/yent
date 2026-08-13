package yent

import (
	"strings"
	"testing"
)

func TestLoadWeightsAcceptsDeclaredGGUFShapes(t *testing.T) {
	cfg := testWeightShapeConfig()
	gguf := testWeightShapeGGUF(t, cfg, true, true)

	w, err := loadWeights(gguf, cfg)
	if err != nil {
		t.Fatalf("loadWeights valid shape: %v", err)
	}
	if w.TokenEmbType != ggmlTypeF32 || w.OutputType != ggmlTypeF32 {
		t.Fatalf("unexpected tensor types: embed=%d output=%d", w.TokenEmbType, w.OutputType)
	}
	if len(w.Layers) != 1 || w.Layers[0].BQ == nil {
		t.Fatalf("expected one layer with optional Q bias loaded")
	}
}

func TestLoadWeightsAllowsTiedOutputWhenLMHeadMissing(t *testing.T) {
	cfg := testWeightShapeConfig()
	gguf := testWeightShapeGGUF(t, cfg, false, false)

	w, err := loadWeights(gguf, cfg)
	if err != nil {
		t.Fatalf("loadWeights tied output: %v", err)
	}
	if len(w.Output) == 0 || len(w.TokenEmbed) == 0 || &w.Output[0] != &w.TokenEmbed[0] {
		t.Fatal("missing output.weight should tie LM head to token embeddings")
	}
}

func TestLoadWeightsRejectsMalformedWeightShapes(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*GGUFFile)
		want   string
	}{
		{
			name: "embedding rows",
			mutate: func(g *GGUFFile) {
				g.Tensors["token_embd.weight"].Dims[1] = 2
			},
			want: "token_embd.weight: shape mismatch",
		},
		{
			name: "output matrix",
			mutate: func(g *GGUFFile) {
				g.Tensors["output.weight"].Dims[1] = 2
			},
			want: "output.weight: shape mismatch",
		},
		{
			name: "attention q projection",
			mutate: func(g *GGUFFile) {
				g.Tensors["blk.0.attn_q.weight"].Dims[1] = 1
			},
			want: "layer 0 attn_q: shape mismatch",
		},
		{
			name: "ffn down projection",
			mutate: func(g *GGUFFile) {
				g.Tensors["blk.0.ffn_down.weight"].Dims[0] = 3
			},
			want: "layer 0 ffn_down: shape mismatch",
		},
		{
			name: "quant matrix row stride",
			mutate: func(g *GGUFFile) {
				info := g.Tensors["blk.0.ffn_down.weight"]
				info.Type = ggmlTypeQ4_0
				info.Dims = [4]uint64{16, 2}
				info.NDims = 2
			},
			want: "layer 0 ffn_down: shape mismatch",
		},
		{
			name: "norm vector",
			mutate: func(g *GGUFFile) {
				g.Tensors["output_norm.weight"].Dims[0] = 1
			},
			want: "output_norm.weight: shape mismatch",
		},
		{
			name: "present optional bias is not silently ignored",
			mutate: func(g *GGUFFile) {
				g.Tensors["blk.0.attn_q.bias"].Offset = uint64(len(g.TensorData) + 4)
			},
			want: "layer 0 attn_q.bias: tensor blk.0.attn_q.bias out of bounds",
		},
	}

	cfg := testWeightShapeConfig()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gguf := cloneTestGGUF(testWeightShapeGGUF(t, cfg, true, true))
			tt.mutate(gguf)

			_, err := loadWeights(gguf, cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("loadWeights error = %v want containing %q", err, tt.want)
			}
		})
	}
}

func TestExpectTensorMatrixRejectsPartialQuantizedRows(t *testing.T) {
	info := &GGUFTensorInfo{
		Name:  "partial-row",
		NDims: 2,
		Dims:  [4]uint64{16, 2},
		Type:  ggmlTypeQ4_0,
	}
	err := expectTensorMatrix(info, "partial-row", 2, 16)
	if err == nil || !strings.Contains(err.Error(), "not whole Q4_0 row blocks") {
		t.Fatalf("expectTensorMatrix error = %v want partial Q4_0 row", err)
	}
}

func testWeightShapeConfig() *LlamaConfig {
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

func testWeightShapeGGUF(t *testing.T, cfg *LlamaConfig, includeOutput, includeBias bool) *GGUFFile {
	t.Helper()
	g := &GGUFFile{
		Tensors: make(map[string]*GGUFTensorInfo),
	}
	addTestF32Tensor(t, g, "token_embd.weight", []uint64{uint64(cfg.EmbedDim), uint64(cfg.VocabSize)})
	addTestF32Tensor(t, g, "output_norm.weight", []uint64{uint64(cfg.EmbedDim)})
	if includeOutput {
		addTestF32Tensor(t, g, "output.weight", []uint64{uint64(cfg.EmbedDim), uint64(cfg.VocabSize)})
	}

	qRows := cfg.NumHeads * cfg.HeadDim
	kvRows := cfg.NumKVHeads * cfg.HeadDim
	prefix := "blk.0."
	addTestF32Tensor(t, g, prefix+"attn_norm.weight", []uint64{uint64(cfg.EmbedDim)})
	addTestF32Tensor(t, g, prefix+"ffn_norm.weight", []uint64{uint64(cfg.EmbedDim)})
	addTestF32Tensor(t, g, prefix+"attn_q.weight", []uint64{uint64(cfg.EmbedDim), uint64(qRows)})
	addTestF32Tensor(t, g, prefix+"attn_k.weight", []uint64{uint64(cfg.EmbedDim), uint64(kvRows)})
	addTestF32Tensor(t, g, prefix+"attn_v.weight", []uint64{uint64(cfg.EmbedDim), uint64(kvRows)})
	addTestF32Tensor(t, g, prefix+"attn_output.weight", []uint64{uint64(qRows), uint64(cfg.EmbedDim)})
	if includeBias {
		addTestF32Tensor(t, g, prefix+"attn_q.bias", []uint64{uint64(qRows)})
		addTestF32Tensor(t, g, prefix+"attn_k.bias", []uint64{uint64(kvRows)})
		addTestF32Tensor(t, g, prefix+"attn_v.bias", []uint64{uint64(kvRows)})
		addTestF32Tensor(t, g, prefix+"attn_output.bias", []uint64{uint64(cfg.EmbedDim)})
	}
	addTestF32Tensor(t, g, prefix+"ffn_gate.weight", []uint64{uint64(cfg.EmbedDim), uint64(cfg.IntermSize)})
	addTestF32Tensor(t, g, prefix+"ffn_up.weight", []uint64{uint64(cfg.EmbedDim), uint64(cfg.IntermSize)})
	addTestF32Tensor(t, g, prefix+"ffn_down.weight", []uint64{uint64(cfg.IntermSize), uint64(cfg.EmbedDim)})
	return g
}

func addTestF32Tensor(t *testing.T, g *GGUFFile, name string, dims []uint64) {
	t.Helper()
	if len(dims) == 0 || len(dims) > maxGGUFTensorDims {
		t.Fatalf("invalid test dims for %s: %v", name, dims)
	}
	nel := uint64(1)
	var fixed [4]uint64
	for i, dim := range dims {
		if dim == 0 || nel > ^uint64(0)/dim {
			t.Fatalf("invalid test dim for %s: %v", name, dims)
		}
		nel *= dim
		fixed[i] = dim
	}
	if nel > uint64(int(^uint(0)>>1)) {
		t.Fatalf("test tensor too large: %s elements=%d", name, nel)
	}
	vals := make([]float32, int(nel))
	for i := range vals {
		vals[i] = float32(i+1) / 100
	}
	g.Tensors[name] = &GGUFTensorInfo{
		Name:   name,
		NDims:  uint32(len(dims)),
		Dims:   fixed,
		Type:   ggmlTypeF32,
		Offset: uint64(len(g.TensorData)),
	}
	g.TensorData = append(g.TensorData, f32RowBytes(vals...)...)
}

func cloneTestGGUF(g *GGUFFile) *GGUFFile {
	out := &GGUFFile{
		Meta:       g.Meta,
		TensorData: append([]byte(nil), g.TensorData...),
		Tensors:    make(map[string]*GGUFTensorInfo, len(g.Tensors)),
	}
	for name, info := range g.Tensors {
		cp := *info
		out.Tensors[name] = &cp
	}
	return out
}
