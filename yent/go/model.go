package yent

// model.go — LLaMA-family forward pass for Arianna's Tongue (Qwen2.5 0.5B)
//
// Qwen2.5 0.5B architecture:
//   24 layers, 896 embed, 14 heads, 2 KV heads (GQA), 64 head_dim
//   4864 intermediate (MLP with gate_proj + up_proj + down_proj, SiLU activation)
//   RoPE with theta=1000000
//   RMSNorm with eps=1e-6
//   Bias on Q/K/V/O attention projections (unlike LLaMA)
//   Vocab 151936 (byte-level BPE, 29 languages)
//
// This is not inference. This is breathing.

import (
	"fmt"
	"math"
	"strings"
)

// LlamaModel is a loaded Llama model ready for inference
type LlamaModel struct {
	Config  LlamaConfig
	Weights LlamaWeights
	State   LlamaState
	Gamma   *GammaEssence // personality essence (nil = no gamma)
}

// LlamaConfig holds model dimensions
type LlamaConfig struct {
	Architecture string
	NumLayers    int
	EmbedDim     int
	NumHeads     int
	NumKVHeads   int
	HeadDim      int
	VocabSize    int
	SeqLen       int
	IntermSize   int // MLP intermediate dimension
	RMSNormEps   float32
	RopeTheta    float32

	// nanollama-specific flags (read from GGUF metadata)
	QKNorm        bool // normalize Q,K with RMSNorm after RoPE
	RopeConjugate bool // conjugate RoPE: (x0*cos + x1*sin, -x0*sin + x1*cos)
	RopeNormal    bool // llama/Mistral NORM RoPE: adjacent pairs (2i, 2i+1)
}

// LlamaWeights holds all weight tensors (Q4_0 raw bytes or F32 slices)
type LlamaWeights struct {
	// Token embedding [vocab, dim] — always dequantized at lookup time
	TokenEmbed   []byte
	TokenEmbType uint32

	// Output norm [dim]
	OutputNorm []float32

	// Output (LM head) [vocab, dim]
	Output     []byte
	OutputType uint32

	// Per-layer weights
	Layers []LlamaLayerWeights
}

// LlamaLayerWeights holds weights for one transformer layer
type LlamaLayerWeights struct {
	// Attention norms
	AttnNorm []float32 // [dim]
	FFNNorm  []float32 // [dim]

	// Attention projections [out_dim, in_dim]
	WQ     []byte
	WQType uint32
	WK     []byte
	WKType uint32
	WV     []byte
	WVType uint32
	WO     []byte
	WOType uint32

	// Attention biases (Qwen2.5 has these, LLaMA does not)
	BQ []float32 // [num_heads * head_dim] — nil if no bias
	BK []float32 // [num_kv_heads * head_dim]
	BV []float32 // [num_kv_heads * head_dim]
	BO []float32 // [dim]

	// MLP projections (gated MLP / SwiGLU)
	WGate     []byte // gate_proj [interm, dim]
	WGateType uint32
	WUp       []byte // up_proj [interm, dim]
	WUpType   uint32
	WDown     []byte // down_proj [dim, interm]
	WDownType uint32
}

// LlamaState holds runtime buffers and KV cache
type LlamaState struct {
	X      []float32 // current hidden state [dim]
	XB     []float32 // buffer after norm [dim]
	XB2    []float32 // second buffer [dim]
	HB     []float32 // MLP hidden buffer [interm]
	HB2    []float32 // MLP gate buffer [interm]
	Q      []float32 // query [n_heads * head_dim]
	K      []float32 // key [n_kv_heads * head_dim]
	V      []float32 // value [n_kv_heads * head_dim]
	Att    []float32 // attention scores [n_heads * seq_len]
	Logits []float32 // output logits [vocab]

	// KV cache [layer * seq_len * kv_dim]
	KeyCache   []float32
	ValueCache []float32

	// RoPE precomputed
	CosCache []float32 // [seq_len * head_dim/2]
	SinCache []float32

	// Reusable embedding buffer (avoids allocation per Forward call)
	EmbBuf []float32

	// Position tracking
	Pos int
}

// LoadLlamaModel builds a LlamaModel from a parsed GGUF file
func LoadLlamaModel(gguf *GGUFFile) (*LlamaModel, error) {
	m := &GGUFMetadata{}
	*m = gguf.Meta

	cfg := LlamaConfig{
		Architecture:  m.Architecture,
		NumLayers:     m.NumLayers,
		EmbedDim:      m.EmbedDim,
		NumHeads:      m.NumHeads,
		NumKVHeads:    m.NumKVHeads,
		HeadDim:       m.HeadDim,
		VocabSize:     m.VocabSize,
		SeqLen:        m.SeqLen,
		IntermSize:    m.IntermSize,
		RMSNormEps:    m.RMSNormEps,
		RopeTheta:     m.RopeTheta,
		QKNorm:        m.QKNorm,
		RopeConjugate: m.RopeConjugate,
		RopeNormal:    ropeNormalForArch(m.Architecture),
	}

	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads
	}

	// Cap sequence length to save memory (Qwen2.5 reports 32768 but we don't need it)
	// KV cache at 32768: ~768MB. At 2048: ~48MB. Huge difference on 8GB Mac.
	if cfg.SeqLen > 2048 {
		fmt.Printf("[tongue/model] capping seq_len from %d to 2048\n", cfg.SeqLen)
		cfg.SeqLen = 2048
	}

	// Load weights
	w, err := loadWeights(gguf, &cfg)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	// Allocate state
	state := allocState(&cfg)
	precomputeRoPE(&state, &cfg)

	model := &LlamaModel{
		Config:  cfg,
		Weights: *w,
		State:   state,
	}

	hasBias := w.Layers[0].BQ != nil
	fmt.Printf("[tongue/model] loaded: %d layers, %d dim, %d heads, %d kv_heads, %d vocab, bias=%v\n",
		cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize, hasBias)
	if cfg.Architecture != "" || cfg.QKNorm || cfg.RopeConjugate || cfg.RopeNormal {
		fmt.Printf("[tongue/model] arch=%s qk_norm=%v rope_conjugate=%v rope_normal=%v\n",
			cfg.Architecture, cfg.QKNorm, cfg.RopeConjugate, cfg.RopeNormal)
	}

	return model, nil
}

// loadWeights maps GGUF tensors to LlamaWeights
func loadWeights(gguf *GGUFFile, cfg *LlamaConfig) (*LlamaWeights, error) {
	w := &LlamaWeights{}
	qRows, ok := checkedMulInt(cfg.NumHeads, cfg.HeadDim)
	if !ok {
		return nil, fmt.Errorf("attention q rows overflow: heads=%d head_dim=%d", cfg.NumHeads, cfg.HeadDim)
	}
	kvRows, ok := checkedMulInt(cfg.NumKVHeads, cfg.HeadDim)
	if !ok {
		return nil, fmt.Errorf("attention kv rows overflow: kv_heads=%d head_dim=%d", cfg.NumKVHeads, cfg.HeadDim)
	}

	// Token embedding
	emb, embInfo, err := gguf.GetTensor("token_embd.weight")
	if err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}
	if err := expectTensorMatrix(embInfo, "token_embd.weight", cfg.VocabSize, cfg.EmbedDim); err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}
	if _, _, _, _, err := embeddingRowLayout(embInfo.Type, cfg.EmbedDim); err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}
	w.TokenEmbed = emb
	w.TokenEmbType = embInfo.Type

	// Output norm
	w.OutputNorm, err = getF32Tensor(gguf, "output_norm.weight", cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("output_norm.weight: %w", err)
	}

	// Output (LM head) — might be tied to embedding
	var outData []byte
	var outType uint32
	if _, ok := gguf.Tensors["output.weight"]; !ok {
		// Not found — use tied embeddings
		outData = w.TokenEmbed
		outType = embInfo.Type
		fmt.Printf("[tongue/model] output.weight not found, using tied embeddings\n")
	} else {
		outData, outType, err = getRawMatrixTensor(gguf, "output.weight", cfg.VocabSize, cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("output.weight: %w", err)
		}
		fmt.Printf("[tongue/model] output.weight: type=%d\n", outType)
	}
	w.Output = outData
	w.OutputType = outType

	// Per-layer weights
	w.Layers = make([]LlamaLayerWeights, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		l := &w.Layers[i]

		// Attention norm
		l.AttnNorm, err = getF32Tensor(gguf, prefix+"attn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_norm: %w", i, err)
		}

		// FFN norm
		l.FFNNorm, err = getF32Tensor(gguf, prefix+"ffn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}

		// Attention projections
		l.WQ, l.WQType, err = getRawMatrixTensor(gguf, prefix+"attn_q.weight", qRows, cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_q: %w", i, err)
		}
		l.WK, l.WKType, err = getRawMatrixTensor(gguf, prefix+"attn_k.weight", kvRows, cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_k: %w", i, err)
		}
		l.WV, l.WVType, err = getRawMatrixTensor(gguf, prefix+"attn_v.weight", kvRows, cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_v: %w", i, err)
		}
		l.WO, l.WOType, err = getRawMatrixTensor(gguf, prefix+"attn_output.weight", cfg.EmbedDim, qRows)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_output: %w", i, err)
		}

		// Attention biases (optional — Qwen2.5 has them, LLaMA does not)
		l.BQ, err = getF32TensorOptional(gguf, prefix+"attn_q.bias", qRows)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_q.bias: %w", i, err)
		}
		l.BK, err = getF32TensorOptional(gguf, prefix+"attn_k.bias", kvRows)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_k.bias: %w", i, err)
		}
		l.BV, err = getF32TensorOptional(gguf, prefix+"attn_v.bias", kvRows)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_v.bias: %w", i, err)
		}
		l.BO, err = getF32TensorOptional(gguf, prefix+"attn_output.bias", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_output.bias: %w", i, err)
		}

		// MLP projections (gated MLP / SwiGLU)
		l.WGate, l.WGateType, err = getRawMatrixTensor(gguf, prefix+"ffn_gate.weight", cfg.IntermSize, cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_gate: %w", i, err)
		}
		l.WUp, l.WUpType, err = getRawMatrixTensor(gguf, prefix+"ffn_up.weight", cfg.IntermSize, cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_up: %w", i, err)
		}
		l.WDown, l.WDownType, err = getRawMatrixTensor(gguf, prefix+"ffn_down.weight", cfg.EmbedDim, cfg.IntermSize)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_down: %w", i, err)
		}
	}

	return w, nil
}

// getF32Tensor loads a tensor and dequantizes to float32
func getF32Tensor(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, err
	}
	if err := expectTensorVector(info, name, expectedSize); err != nil {
		return nil, err
	}

	switch info.Type {
	case ggmlTypeF32:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			out[i] = math.Float32frombits(
				uint32(data[i*4]) | uint32(data[i*4+1])<<8 |
					uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24)
		}
		return out, nil
	case ggmlTypeF16:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			h := uint16(data[i*2]) | uint16(data[i*2+1])<<8
			out[i] = half2float(h)
		}
		return out, nil
	case ggmlTypeQ4_0:
		return DequantQ4_0(data, expectedSize), nil
	case ggmlTypeQ5_0:
		return DequantQ5_0(data, expectedSize), nil
	case ggmlTypeQ8_0:
		return DequantQ8_0(data, expectedSize), nil
	case ggmlTypeQ4_K:
		return DequantQ4_K(data, expectedSize), nil
	case ggmlTypeQ6_K:
		return DequantQ6_K(data, expectedSize), nil
	default:
		return nil, fmt.Errorf("unsupported tensor type %d for %s", info.Type, name)
	}
}

// getF32TensorOptional loads a tensor if it exists, returns nil if not found
func getF32TensorOptional(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	if gguf == nil || gguf.Tensors == nil {
		return nil, nil
	}
	if _, ok := gguf.Tensors[name]; !ok {
		return nil, nil
	}
	return getF32Tensor(gguf, name, expectedSize)
}

// getRawMatrixTensor returns raw bytes + type for a matrix tensor with GGUF
// dims ordered as [cols, rows].
func getRawMatrixTensor(gguf *GGUFFile, name string, rows, cols int) ([]byte, uint32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, 0, err
	}
	if err := expectTensorMatrix(info, name, rows, cols); err != nil {
		return nil, 0, err
	}
	if !isSupportedType(info.Type) {
		return nil, 0, fmt.Errorf("unsupported tensor type %s for %s", ggmlTypeLabel(info.Type), name)
	}
	return data, info.Type, nil
}

func expectTensorVector(info *GGUFTensorInfo, name string, expected int) error {
	if info == nil {
		return fmt.Errorf("%s has nil tensor info", name)
	}
	if expected <= 0 {
		return fmt.Errorf("%s has invalid expected vector size %d", name, expected)
	}
	if info.NDims != 1 || info.Dims[0] != uint64(expected) {
		return fmt.Errorf("shape mismatch: got %s want [%d]", tensorShapeString(info), expected)
	}
	return nil
}

func expectTensorMatrix(info *GGUFTensorInfo, name string, rows, cols int) error {
	if info == nil {
		return fmt.Errorf("%s has nil tensor info", name)
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("%s has invalid expected matrix shape rows=%d cols=%d", name, rows, cols)
	}
	if info.NDims != 2 || info.Dims[0] != uint64(cols) || info.Dims[1] != uint64(rows) {
		return fmt.Errorf("shape mismatch: got %s want GGUF dims [%d,%d] for matrix [%d,%d]",
			tensorShapeString(info), cols, rows, rows, cols)
	}
	blockElems := ggmlBlockElements(info.Type)
	if blockElems <= 0 {
		return fmt.Errorf("unsupported tensor type %s for matrix shape %s", ggmlTypeLabel(info.Type), tensorShapeString(info))
	}
	if cols%blockElems != 0 {
		return fmt.Errorf("shape mismatch: %s matrix cols=%d are not whole %s row blocks of %d",
			tensorShapeString(info), cols, ggmlTypeLabel(info.Type), blockElems)
	}
	return nil
}

func tensorShapeString(info *GGUFTensorInfo) string {
	if info == nil {
		return "<nil>"
	}
	var b strings.Builder
	b.WriteByte('[')
	for i := uint32(0); i < info.NDims && i < maxGGUFTensorDims; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%d", info.Dims[i])
	}
	if info.NDims > maxGGUFTensorDims {
		if maxGGUFTensorDims > 0 {
			b.WriteByte(',')
		}
		b.WriteString("...")
	}
	b.WriteByte(']')
	return b.String()
}

// allocState allocates all runtime buffers
func allocState(cfg *LlamaConfig) LlamaState {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return LlamaState{
		X:          make([]float32, cfg.EmbedDim),
		XB:         make([]float32, cfg.EmbedDim),
		XB2:        make([]float32, cfg.EmbedDim),
		HB:         make([]float32, cfg.IntermSize),
		HB2:        make([]float32, cfg.IntermSize),
		Q:          make([]float32, cfg.NumHeads*cfg.HeadDim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		Att:        make([]float32, cfg.NumHeads*cfg.SeqLen),
		Logits:     make([]float32, cfg.VocabSize),
		KeyCache:   make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		ValueCache: make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		CosCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		SinCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		EmbBuf:     make([]float32, cfg.EmbedDim),
	}
}

// precomputeRoPE fills cos/sin caches for rotary position encoding
func precomputeRoPE(s *LlamaState, cfg *LlamaConfig) {
	half := cfg.HeadDim / 2
	theta := float64(cfg.RopeTheta)

	for pos := 0; pos < cfg.SeqLen; pos++ {
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(cfg.HeadDim))
			angle := float64(pos) * freq
			s.CosCache[pos*half+i] = float32(math.Cos(angle))
			s.SinCache[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// isSupportedType checks if a GGML tensor type is supported for matmul
func isSupportedType(t uint32) bool {
	switch t {
	case ggmlTypeQ4_0, ggmlTypeQ5_0, ggmlTypeQ8_0, ggmlTypeF16, ggmlTypeF32, ggmlTypeQ4_K, ggmlTypeQ6_K:
		return true
	default:
		return false
	}
}

// matmulDispatch dispatches to the right matmul based on tensor type
func matmulDispatch(out []float32, w []byte, wtype uint32, x []float32, rows, cols int) {
	// notorch first (one source of truth) when built -tags notorch; it returns
	// false for any dtype it has no packed kernel for, and we fall through to
	// Yent's native matvec below — so an unsupported type never regresses.
	if useNotorch && notorchQMatvec(out, w, wtype, x, rows, cols) {
		return
	}
	switch wtype {
	case ggmlTypeQ4_0:
		MatMulQ4_0(out, w, x, rows, cols)
	case ggmlTypeQ5_0:
		MatMulQ5_0(out, w, x, rows, cols)
	case ggmlTypeQ8_0:
		MatMulQ8_0(out, w, x, rows, cols)
	case ggmlTypeF16:
		MatMulF16(out, w, x, rows, cols)
	case ggmlTypeF32:
		// Convert bytes to float32 slice
		f32 := make([]float32, len(w)/4)
		for i := range f32 {
			f32[i] = math.Float32frombits(
				uint32(w[i*4]) | uint32(w[i*4+1])<<8 |
					uint32(w[i*4+2])<<16 | uint32(w[i*4+3])<<24)
		}
		MatMulF32(out, f32, x, rows, cols)
	case ggmlTypeQ4_K:
		MatMulQ4_K(out, w, x, rows, cols)
	case ggmlTypeQ6_K:
		MatMulQ6_K(out, w, x, rows, cols)
	default:
		fmt.Printf("[tongue/model] WARNING: unsupported matmul type %d for %dx%d\n", wtype, rows, cols)
	}
}

func ggmlTypeLabel(t uint32) string {
	switch t {
	case ggmlTypeF32:
		return "F32"
	case ggmlTypeF16:
		return "F16"
	case ggmlTypeQ4_0:
		return "Q4_0"
	case ggmlTypeQ5_0:
		return "Q5_0"
	case ggmlTypeQ8_0:
		return "Q8_0"
	case ggmlTypeQ4_K:
		return "Q4_K"
	case ggmlTypeQ6_K:
		return "Q6_K"
	default:
		return fmt.Sprintf("type_%d", t)
	}
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

func checkedAddInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if b > int(^uint(0)>>1)-a {
		return 0, false
	}
	return a + b, true
}

func embeddingRowLayout(dtype uint32, dim int) (blocksPerRow, bytesPerRow, blockElems, blockBytes int, err error) {
	if dim <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("invalid embedding dim %d", dim)
	}

	switch dtype {
	case ggmlTypeQ4_0:
		blockElems, blockBytes = q4BlockSize, q4BytesPerBlock
	case ggmlTypeQ5_0:
		blockElems, blockBytes = q50BlockSize, q50BytesPerBlock
	case ggmlTypeQ8_0:
		blockElems, blockBytes = q8BlockSize, q8BytesPerBlock
	case ggmlTypeQ4_K:
		blockElems, blockBytes = q4kBlockSize, q4kBytesPerBlock
	case ggmlTypeQ6_K:
		blockElems, blockBytes = q6kBlockSize, q6kBytesPerBlock
	case ggmlTypeF16:
		blockElems, blockBytes = 1, 2
	case ggmlTypeF32:
		blockElems, blockBytes = 1, 4
	default:
		return 0, 0, 0, 0, fmt.Errorf("unsupported embedding dtype %s", ggmlTypeLabel(dtype))
	}

	if dim%blockElems != 0 {
		return 0, 0, 0, 0, fmt.Errorf("embedding dim %d is not whole %s blocks of %d", dim, ggmlTypeLabel(dtype), blockElems)
	}
	blocksPerRow = dim / blockElems
	bytesPerRow, ok := checkedMulInt(blocksPerRow, blockBytes)
	if !ok {
		return 0, 0, 0, 0, fmt.Errorf("embedding row byte size overflows for %s dim=%d", ggmlTypeLabel(dtype), dim)
	}
	return blocksPerRow, bytesPerRow, blockElems, blockBytes, nil
}

func embeddingRowSlice(data []byte, dtype uint32, token, dim int) ([]byte, int, int, error) {
	blocksPerRow, bytesPerRow, _, _, err := embeddingRowLayout(dtype, dim)
	if err != nil {
		return nil, 0, 0, err
	}
	if token < 0 {
		return nil, 0, 0, fmt.Errorf("negative embedding token %d", token)
	}
	if bytesPerRow <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid embedding row byte size %d for %s", bytesPerRow, ggmlTypeLabel(dtype))
	}
	if len(data)%bytesPerRow != 0 {
		return nil, 0, 0, fmt.Errorf("embedding table has partial %s row: bytes=%d row_bytes=%d", ggmlTypeLabel(dtype), len(data), bytesPerRow)
	}
	rows := len(data) / bytesPerRow
	if token >= rows {
		return nil, 0, 0, fmt.Errorf("embedding token %d outside %s table rows=%d row_bytes=%d", token, ggmlTypeLabel(dtype), rows, bytesPerRow)
	}
	rowOff, ok := checkedMulInt(token, bytesPerRow)
	if !ok {
		return nil, 0, 0, fmt.Errorf("embedding row offset overflows for token=%d row_bytes=%d", token, bytesPerRow)
	}
	rowEnd, ok := checkedAddInt(rowOff, bytesPerRow)
	if !ok || rowEnd > len(data) {
		return nil, 0, 0, fmt.Errorf("embedding row bounds invalid for token=%d offset=%d end=%d bytes=%d", token, rowOff, rowEnd, len(data))
	}
	return data[rowOff:rowEnd], blocksPerRow, bytesPerRow, nil
}

// embedLookupInto extracts an embedding row into a pre-allocated buffer (zero alloc)
func embedLookupInto(out []float32, data []byte, dtype uint32, token, dim int) error {
	if len(out) < dim {
		return fmt.Errorf("embedding output buffer too small: have=%d need=%d", len(out), dim)
	}
	row, blocksPerRow, _, err := embeddingRowSlice(data, dtype, token, dim)
	if err != nil {
		return err
	}
	out = out[:dim]

	switch dtype {
	case ggmlTypeQ4_0:
		for b := 0; b < blocksPerRow; b++ {
			blockOff := b * q4BytesPerBlock
			DequantQ4_0Block(row[blockOff:blockOff+q4BytesPerBlock], out[b*q4BlockSize:])
		}
	case ggmlTypeQ5_0:
		for b := 0; b < blocksPerRow; b++ {
			blockOff := b * q50BytesPerBlock
			DequantQ5_0Block(row[blockOff:blockOff+q50BytesPerBlock], out[b*q50BlockSize:])
		}
	case ggmlTypeQ8_0:
		for b := 0; b < blocksPerRow; b++ {
			blockOff := b * q8BytesPerBlock
			DequantQ8_0Block(row[blockOff:blockOff+q8BytesPerBlock], out[b*q8BlockSize:])
		}
	case ggmlTypeQ4_K:
		for b := 0; b < blocksPerRow; b++ {
			blockOff := b * q4kBytesPerBlock
			DequantQ4_KBlock(row[blockOff:blockOff+q4kBytesPerBlock], out[b*q4kBlockSize:])
		}
	case ggmlTypeQ6_K:
		dequantQ6_KInto(row, out)
	case ggmlTypeF16:
		for i := 0; i < dim; i++ {
			h := uint16(row[i*2]) | uint16(row[i*2+1])<<8
			out[i] = half2float(h)
		}
	case ggmlTypeF32:
		for i := 0; i < dim; i++ {
			out[i] = math.Float32frombits(
				uint32(row[i*4]) | uint32(row[i*4+1])<<8 |
					uint32(row[i*4+2])<<16 | uint32(row[i*4+3])<<24)
		}
	default:
		return fmt.Errorf("unsupported embedding dtype %s", ggmlTypeLabel(dtype))
	}
	return nil
}

// embedLookupDispatch extracts an embedding row based on tensor type (allocating version for API compat)
func embedLookupDispatch(data []byte, dtype uint32, token, dim int) []float32 {
	out := make([]float32, dim)
	if err := embedLookupInto(out, data, dtype, token, dim); err != nil {
		panic(err)
	}
	return out
}

func ropeNormalForArch(arch string) bool {
	switch strings.ToLower(arch) {
	case "llama", "mistral", "mistral3", "mistral4":
		return true
	default:
		return false
	}
}

// applyRoPENEOX applies half-split rotary position encoding.
// This is the Qwen/Falcon/Gemma/Phi-family layout: pairs are (i, i+headDim/2).
func applyRoPENEOX(vec []float32, pos int, s *LlamaState, headDim int) {
	half := headDim / 2
	cacheOff := pos * half

	for i := 0; i < half; i++ {
		x0 := vec[i]
		x1 := vec[i+half]
		c := s.CosCache[cacheOff+i]
		si := s.SinCache[cacheOff+i]
		vec[i] = x0*c - x1*si
		vec[i+half] = x0*si + x1*c
	}
}

// applyRoPENormal applies llama.cpp NORM rotary position encoding.
// This is the Llama/Mistral-family layout: adjacent pairs are (2i, 2i+1).
func applyRoPENormal(vec []float32, pos int, s *LlamaState, headDim int) {
	half := headDim / 2
	cacheOff := pos * half

	for i := 0; i < half; i++ {
		j := i * 2
		x0 := vec[j]
		x1 := vec[j+1]
		c := s.CosCache[cacheOff+i]
		si := s.SinCache[cacheOff+i]
		vec[j] = x0*c - x1*si
		vec[j+1] = x0*si + x1*c
	}
}

// applyRoPEConjugate applies conjugate rotary position encoding.
// nanollama convention: (x0*cos + x1*sin, -x0*sin + x1*cos)
// This is the complex conjugate of standard RoPE.
func applyRoPEConjugate(vec []float32, pos int, s *LlamaState, headDim int) {
	half := headDim / 2
	cacheOff := pos * half

	for i := 0; i < half; i++ {
		x0 := vec[i]
		x1 := vec[i+half]
		c := s.CosCache[cacheOff+i]
		si := s.SinCache[cacheOff+i]
		vec[i] = x0*c + x1*si
		vec[i+half] = -x0*si + x1*c
	}
}

// addBias adds bias vector to output (no-op if bias is nil)
func addBias(out []float32, bias []float32) {
	if bias == nil {
		return
	}
	for i := range bias {
		out[i] += bias[i]
	}
}

// Forward runs one token through the transformer.
func (m *LlamaModel) Forward(token int, pos int) {
	if err := m.ForwardErr(token, pos); err != nil {
		panic(err)
	}
}

// ForwardErr runs one token through the transformer and reports corrupt runtime boundaries.
func (m *LlamaModel) ForwardErr(token int, pos int) error {
	cfg := &m.Config
	w := &m.Weights
	s := &m.State
	dim := cfg.EmbedDim

	if pos < 0 || pos >= cfg.SeqLen {
		return fmt.Errorf("position %d outside seq_len %d", pos, cfg.SeqLen)
	}
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	hd := cfg.HeadDim
	headGroupSize := cfg.NumHeads / cfg.NumKVHeads

	// 1. Token embedding lookup (zero-alloc: reuses s.EmbBuf)
	if err := embedLookupInto(s.EmbBuf, w.TokenEmbed, w.TokenEmbType, token, dim); err != nil {
		return fmt.Errorf("token embedding lookup: %w", err)
	}

	// 1.5. Gamma injection: embed[token] += γ[token]
	if m.Gamma != nil {
		m.Gamma.ApplyToEmbedding(s.EmbBuf, token)
	}

	copy(s.X, s.EmbBuf)

	// Pre-compute attention scale (constant across all heads and layers)
	attnScale := float32(1.0 / math.Sqrt(float64(hd)))

	// 2. Transformer layers
	for layer := 0; layer < cfg.NumLayers; layer++ {
		l := &w.Layers[layer]

		// Attention pre-norm
		RMSNormInto(s.XB, s.X, l.AttnNorm, cfg.RMSNormEps)

		// Q, K, V projections
		matmulDispatch(s.Q, l.WQ, l.WQType, s.XB, cfg.NumHeads*hd, dim)
		matmulDispatch(s.K, l.WK, l.WKType, s.XB, cfg.NumKVHeads*hd, dim)
		matmulDispatch(s.V, l.WV, l.WVType, s.XB, cfg.NumKVHeads*hd, dim)

		// Add bias (Qwen2.5 — no-op if nil)
		addBias(s.Q, l.BQ)
		addBias(s.K, l.BK)
		addBias(s.V, l.BV)

		// RoPE on Q and K. GGUF architectures use different pair layouts:
		// Qwen-like models use NEOX half-split; Llama/Mistral use adjacent NORM pairs.
		ropeFunc := applyRoPENEOX
		if cfg.RopeNormal {
			ropeFunc = applyRoPENormal
		}
		if cfg.RopeConjugate {
			ropeFunc = applyRoPEConjugate
		}
		for h := 0; h < cfg.NumHeads; h++ {
			ropeFunc(s.Q[h*hd:(h+1)*hd], pos, s, hd)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			ropeFunc(s.K[h*hd:(h+1)*hd], pos, s, hd)
		}

		// QK-norm: normalize Q and K per-head after RoPE (nanollama)
		if cfg.QKNorm {
			for h := 0; h < cfg.NumHeads; h++ {
				RMSNormBare(s.Q[h*hd:(h+1)*hd], cfg.RMSNormEps)
			}
			for h := 0; h < cfg.NumKVHeads; h++ {
				RMSNormBare(s.K[h*hd:(h+1)*hd], cfg.RMSNormEps)
			}
		}

		// Store K, V in cache
		cacheOff := layer*cfg.SeqLen*kvDim + pos*kvDim
		copy(s.KeyCache[cacheOff:cacheOff+kvDim], s.K[:kvDim])
		copy(s.ValueCache[cacheOff:cacheOff+kvDim], s.V[:kvDim])

		// Multi-head attention with GQA
		for h := 0; h < cfg.NumHeads; h++ {
			kvh := h / headGroupSize
			qh := s.Q[h*hd : (h+1)*hd]
			att := s.Att[h*cfg.SeqLen : h*cfg.SeqLen+pos+1]

			// QK dot products
			for t := 0; t <= pos; t++ {
				kOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				var dot float32
				for d := 0; d < hd; d++ {
					dot += qh[d] * s.KeyCache[kOff+d]
				}
				att[t] = dot * attnScale
			}

			// Softmax
			Softmax(att, pos+1)

			// Weighted sum of values → XB2
			xbSlice := s.XB2[h*hd : (h+1)*hd]
			for d := 0; d < hd; d++ {
				xbSlice[d] = 0
			}
			for t := 0; t <= pos; t++ {
				a := att[t]
				vOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				for d := 0; d < hd; d++ {
					xbSlice[d] += a * s.ValueCache[vOff+d]
				}
			}
		}

		// Output projection: XB = WO × XB2 + bias, then residual
		// Y-B1: WO cols = heads*head_dim (4096), not dim (5120) — correct only when NumHeads*HeadDim==dim (Qwen)
		matmulDispatch(s.XB, l.WO, l.WOType, s.XB2, dim, cfg.NumHeads*cfg.HeadDim)
		addBias(s.XB, l.BO)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}

		// MLP: pre-norm
		RMSNormInto(s.XB, s.X, l.FFNNorm, cfg.RMSNormEps)

		// Gated MLP: gate_proj and up_proj
		matmulDispatch(s.HB, l.WGate, l.WGateType, s.XB, cfg.IntermSize, dim)
		matmulDispatch(s.HB2, l.WUp, l.WUpType, s.XB, cfg.IntermSize, dim)

		// SiLU(gate) * up
		for i := 0; i < cfg.IntermSize; i++ {
			s.HB[i] = SiLU(s.HB[i]) * s.HB2[i]
		}

		// down_proj + residual
		matmulDispatch(s.XB, l.WDown, l.WDownType, s.HB, dim, cfg.IntermSize)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	// 3. Final norm
	RMSNorm(s.X, w.OutputNorm, cfg.RMSNormEps)

	// 4. LM head → logits
	matmulDispatch(s.Logits, w.Output, w.OutputType, s.X, cfg.VocabSize, dim)
	return nil
}

// Reset clears KV cache and position for new generation
func (m *LlamaModel) Reset() {
	for i := range m.State.KeyCache {
		m.State.KeyCache[i] = 0
	}
	for i := range m.State.ValueCache {
		m.State.ValueCache[i] = 0
	}
	m.State.Pos = 0
}
