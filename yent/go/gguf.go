package yent

// gguf.go — GGUF file parser for Arianna's Tongue (Qwen2.5 0.5B)
//
// GGUF is llama.cpp's binary format. Structure:
//   Header: magic + version + tensor_count + metadata_count
//   Metadata: key-value pairs (vocab, config, etc.)
//   Tensor info: name + dims + type + offset
//   Alignment padding
//   Tensor data blob
//
// We support Q4_0 quantization (the format our GGUF uses):
//   Block of 32 values = 2 bytes (fp16 scale) + 16 bytes (4-bit pairs) = 18 bytes

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// GGUF constants
const (
	ggufMagic   = 0x46554747 // "GGUF" in LE
	ggufVersion = 3

	maxGGUFStringBytes   = 1 << 24
	maxGGUFArrayElements = 1 << 22
	maxGGUFArrayDepth    = 4
	maxGGUFMetadataCount = 1 << 20
	maxGGUFTensorCount   = 1 << 20
	maxGGUFTensorDims    = 4

	maxInt32Value = int64(1<<31 - 1)
	minInt32Value = int64(-1 << 31)

	// GGUF value types
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12

	// GGML tensor types
	ggmlTypeF32  = 0
	ggmlTypeF16  = 1
	ggmlTypeQ4_0 = 2
	ggmlTypeQ4_1 = 3
	ggmlTypeQ5_0 = 6
	ggmlTypeQ5_1 = 7
	ggmlTypeQ8_0 = 8
	ggmlTypeQ8_1 = 9
	ggmlTypeQ2_K = 10
	ggmlTypeQ3_K = 11
	ggmlTypeQ4_K = 12
	ggmlTypeQ5_K = 13
	ggmlTypeQ6_K = 14
)

// GGUFMetadata holds parsed metadata
type GGUFMetadata struct {
	// Model architecture
	Architecture string
	NumLayers    int
	EmbedDim     int
	NumHeads     int
	NumKVHeads   int
	HeadDim      int
	VocabSize    int
	SeqLen       int
	IntermSize   int // MLP intermediate size
	RMSNormEps   float32
	RopeTheta    float32
	RopeFreqBase float32

	// nanollama-specific flags
	QKNorm        bool // normalize Q,K with RMSNorm after RoPE (parameterless)
	RopeConjugate bool // conjugate RoPE convention: (x0*cos+x1*sin, -x0*sin+x1*cos)

	// Tokenizer
	TokenList      []string
	TokenScores    []float32
	TokenTypes     []int32
	TokenMerges    []string // GPT-2 BPE merge rules (empty for SentencePiece)
	TokenizerModel string   // "llama" (SentencePiece) or "gpt2" (byte-level BPE)
	BosID          int
	EosID          int
	AddSpacePrefix bool

	// Raw KV store
	KV map[string]interface{}
}

// GGUFTensorInfo describes a tensor in the file
type GGUFTensorInfo struct {
	Name   string
	NDims  uint32
	Dims   [4]uint64
	Type   uint32
	Offset uint64
}

// GGUFFile is a parsed GGUF file
type GGUFFile struct {
	Meta       GGUFMetadata
	Tensors    map[string]*GGUFTensorInfo
	TensorData []byte // mmap'd or read tensor data blob
	DataOffset int64  // offset where tensor data starts in file
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > maxGGUFStringBytes { // 16MB sanity limit
		return "", fmt.Errorf("string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readValue(r io.Reader, vtype uint32) (interface{}, error) {
	return readValueDepth(r, vtype, 0)
}

func readValueDepth(r io.Reader, vtype uint32, depth int) (interface{}, error) {
	switch vtype {
	case ggufTypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case ggufTypeString:
		return readString(r)
	case ggufTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeArray:
		if depth >= maxGGUFArrayDepth {
			return nil, fmt.Errorf("array nesting too deep: %d > %d", depth+1, maxGGUFArrayDepth)
		}
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		if count > maxGGUFArrayElements {
			return nil, fmt.Errorf("array too large: %d > %d", count, maxGGUFArrayElements)
		}
		arr := make([]interface{}, count)
		for i := uint64(0); i < count; i++ {
			v, err := readValueDepth(r, elemType, depth+1)
			if err != nil {
				return nil, fmt.Errorf("array[%d]: %w", i, err)
			}
			arr[i] = v
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown GGUF type: %d", vtype)
	}
}

// ggmlTypeSize returns bytes per element for a tensor type (for blocked types, per block)
func ggmlBlockSize(t uint32) int {
	switch t {
	case ggmlTypeF32:
		return 4
	case ggmlTypeF16:
		return 2
	case ggmlTypeQ4_0:
		return 18 // 2 (fp16 scale) + 16 (32 x 4-bit values)
	case ggmlTypeQ4_1:
		return 20 // 2 (min) + 2 (scale) + 16 data
	case ggmlTypeQ8_0:
		return 34 // 2 (fp16 scale) + 32 (32 x 8-bit)
	case ggmlTypeQ5_0:
		return 22 // 2 (scale) + 4 (qh) + 16 (qs) per 32 elements
	case ggmlTypeQ6_K:
		return 210 // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) per 256 elements
	case ggmlTypeQ4_K:
		return 144 // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) per 256 elements
	default:
		return 0
	}
}

// ggmlBlockElements returns the number of elements per block
func ggmlBlockElements(t uint32) int {
	switch t {
	case ggmlTypeF32, ggmlTypeF16:
		return 1
	case ggmlTypeQ4_0, ggmlTypeQ4_1, ggmlTypeQ5_0, ggmlTypeQ8_0:
		return 32
	case ggmlTypeQ4_K, ggmlTypeQ6_K:
		return 256 // k-quant super block
	default:
		return 0
	}
}

// tensorBytes returns total bytes for a tensor
func tensorBytes(info *GGUFTensorInfo) (uint64, error) {
	if info == nil {
		return 0, fmt.Errorf("nil tensor info")
	}
	if info.NDims == 0 || info.NDims > maxGGUFTensorDims {
		return 0, fmt.Errorf("invalid ndim %d", info.NDims)
	}

	nel := uint64(1)
	for i := uint32(0); i < info.NDims; i++ {
		if info.Dims[i] == 0 {
			return 0, fmt.Errorf("zero dimension at axis %d", i)
		}
		if nel > ^uint64(0)/info.Dims[i] {
			return 0, fmt.Errorf("element count overflows at axis %d (%d * %d)", i, nel, info.Dims[i])
		}
		nel *= info.Dims[i]
	}
	bs := uint64(ggmlBlockSize(info.Type))
	be := uint64(ggmlBlockElements(info.Type))
	if bs == 0 || be == 0 {
		return 0, fmt.Errorf("unsupported tensor type %d", info.Type)
	}
	if nel%be != 0 {
		return 0, fmt.Errorf("tensor %s has %d elements, not whole blocks of %d for type %d",
			info.Name, nel, be, info.Type)
	}
	blocks := nel / be
	if blocks > ^uint64(0)/bs {
		return 0, fmt.Errorf("byte size overflows for %d blocks of %d bytes", blocks, bs)
	}
	return blocks * bs, nil
}

func validateTensorRanges(tensors map[string]*GGUFTensorInfo, dataSize uint64) error {
	for name, info := range tensors {
		size, err := tensorBytes(info)
		if err != nil {
			return fmt.Errorf("tensor %s invalid layout: %w", name, err)
		}
		if info.Offset > dataSize {
			return fmt.Errorf("tensor %s out of tensor data bounds: offset %d > data size %d",
				name, info.Offset, dataSize)
		}
		if size > dataSize-info.Offset {
			return fmt.Errorf("tensor %s out of tensor data bounds: offset %d + size %d > data size %d",
				name, info.Offset, size, dataSize)
		}
	}
	return nil
}

// LoadGGUF loads a GGUF file
func LoadGGUF(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}
	defer f.Close()

	// Read header
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("bad magic: 0x%08X (expected 0x%08X)", magic, ggufMagic)
	}

	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	var tensorCount, metadataCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &metadataCount); err != nil {
		return nil, err
	}
	if tensorCount > maxGGUFTensorCount {
		return nil, fmt.Errorf("tensor count too large: %d > %d", tensorCount, maxGGUFTensorCount)
	}
	if metadataCount > maxGGUFMetadataCount {
		return nil, fmt.Errorf("metadata count too large: %d > %d", metadataCount, maxGGUFMetadataCount)
	}

	fmt.Printf("[tongue/gguf] version=%d tensors=%d metadata=%d\n", version, tensorCount, metadataCount)

	// Read metadata
	kv := make(map[string]interface{})
	for i := uint64(0); i < metadataCount; i++ {
		key, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("read metadata key %d: %w", i, err)
		}
		var vtype uint32
		if err := binary.Read(f, binary.LittleEndian, &vtype); err != nil {
			return nil, fmt.Errorf("read metadata type %d: %w", i, err)
		}
		val, err := readValue(f, vtype)
		if err != nil {
			return nil, fmt.Errorf("read metadata value '%s': %w", key, err)
		}
		kv[key] = val
	}

	// Read tensor infos
	tensors := make(map[string]*GGUFTensorInfo, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("read tensor name %d: %w", i, err)
		}
		var ndims uint32
		if err := binary.Read(f, binary.LittleEndian, &ndims); err != nil {
			return nil, fmt.Errorf("read tensor ndim %d (%s): %w", i, name, err)
		}
		if ndims == 0 || ndims > maxGGUFTensorDims {
			return nil, fmt.Errorf("tensor %s invalid ndim %d", name, ndims)
		}
		var dims [4]uint64
		for d := uint32(0); d < ndims; d++ {
			if err := binary.Read(f, binary.LittleEndian, &dims[d]); err != nil {
				return nil, fmt.Errorf("read tensor dim %d for %s: %w", d, name, err)
			}
		}
		var ttype uint32
		if err := binary.Read(f, binary.LittleEndian, &ttype); err != nil {
			return nil, fmt.Errorf("read tensor type %d (%s): %w", i, name, err)
		}
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("read tensor offset %d (%s): %w", i, name, err)
		}
		if _, exists := tensors[name]; exists {
			return nil, fmt.Errorf("duplicate tensor name: %s", name)
		}
		info := &GGUFTensorInfo{
			Name:   name,
			NDims:  ndims,
			Dims:   dims,
			Type:   ttype,
			Offset: offset,
		}
		if _, err := tensorBytes(info); err != nil {
			return nil, fmt.Errorf("tensor %s invalid layout: %w", name, err)
		}
		tensors[name] = info
	}

	// Current position = end of header/metadata/tensor_info
	headerEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	// GGUF alignment = 32 bytes
	alignment := int64(32)
	dataOffset := ((headerEnd + alignment - 1) / alignment) * alignment

	// Read all tensor data
	fileInfo, err := f.Stat()
	if err != nil {
		return nil, err
	}
	dataSize := fileInfo.Size() - dataOffset
	if dataSize <= 0 {
		return nil, fmt.Errorf("no tensor data (dataOffset=%d, fileSize=%d)", dataOffset, fileInfo.Size())
	}
	if err := validateTensorRanges(tensors, uint64(dataSize)); err != nil {
		return nil, err
	}

	fmt.Printf("[tongue/gguf] data offset=%d size=%.1f MB\n", dataOffset, float64(dataSize)/1024/1024)

	if _, err := f.Seek(dataOffset, io.SeekStart); err != nil {
		return nil, err
	}
	tensorData := make([]byte, dataSize)
	if _, err := io.ReadFull(f, tensorData); err != nil {
		return nil, fmt.Errorf("read tensor data: %w", err)
	}

	// Parse metadata into structured form
	meta, err := parseMetadata(kv)
	if err != nil {
		return nil, fmt.Errorf("parse metadata: %w", err)
	}

	return &GGUFFile{
		Meta:       meta,
		Tensors:    tensors,
		TensorData: tensorData,
		DataOffset: dataOffset,
	}, nil
}

func metadataStringArray(kv map[string]interface{}, key string) ([]string, bool, error) {
	v, ok := kv[key]
	if !ok {
		return nil, false, nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, true, fmt.Errorf("%s is %T, want array", key, v)
	}
	out := make([]string, len(arr))
	for i, item := range arr {
		s, ok := item.(string)
		if !ok {
			return nil, true, fmt.Errorf("%s[%d] is %T, want string", key, i, item)
		}
		out[i] = s
	}
	return out, true, nil
}

func metadataStringValue(kv map[string]interface{}, key string) (string, bool, error) {
	v, ok := kv[key]
	if !ok {
		return "", false, nil
	}
	s, ok := v.(string)
	if !ok {
		return "", true, fmt.Errorf("%s is %T, want string", key, v)
	}
	if strings.TrimSpace(s) == "" {
		return "", true, fmt.Errorf("%s is empty", key)
	}
	return s, true, nil
}

func metadataFloat32Value(v interface{}) (float32, bool) {
	switch x := v.(type) {
	case float32:
		return x, true
	case float64:
		return float32(x), true
	case uint8:
		return float32(x), true
	case int8:
		return float32(x), true
	case uint16:
		return float32(x), true
	case int16:
		return float32(x), true
	case uint32:
		return float32(x), true
	case int32:
		return float32(x), true
	case uint64:
		return float32(x), true
	case int64:
		return float32(x), true
	default:
		return 0, false
	}
}

func metadataFloat32Array(kv map[string]interface{}, key string) ([]float32, bool, error) {
	v, ok := kv[key]
	if !ok {
		return nil, false, nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, true, fmt.Errorf("%s is %T, want array", key, v)
	}
	out := make([]float32, len(arr))
	for i, item := range arr {
		val, ok := metadataFloat32Value(item)
		if !ok {
			return nil, true, fmt.Errorf("%s[%d] is %T, want numeric", key, i, item)
		}
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			return nil, true, fmt.Errorf("%s[%d] is non-finite", key, i)
		}
		out[i] = val
	}
	return out, true, nil
}

func metadataPositiveFloat32(kv map[string]interface{}, key string) (float32, bool, error) {
	v, ok := kv[key]
	if !ok {
		return 0, false, nil
	}
	val, valid := metadataFloat32Value(v)
	if !valid {
		return 0, true, fmt.Errorf("%s is %T, want numeric", key, v)
	}
	if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
		return 0, true, fmt.Errorf("%s is non-finite", key)
	}
	if val <= 0 {
		return 0, true, fmt.Errorf("%s must be positive, got %g", key, val)
	}
	return val, true, nil
}

func metadataInt32Value(v interface{}) (int32, bool) {
	switch x := v.(type) {
	case uint8:
		return int32(x), true
	case int8:
		return int32(x), true
	case uint16:
		return int32(x), true
	case int16:
		return int32(x), true
	case uint32:
		if uint64(x) > uint64(maxInt32Value) {
			return 0, false
		}
		return int32(x), true
	case int32:
		return x, true
	case uint64:
		if x > uint64(maxInt32Value) {
			return 0, false
		}
		return int32(x), true
	case int64:
		if x < minInt32Value || x > maxInt32Value {
			return 0, false
		}
		return int32(x), true
	case int:
		if int64(x) < minInt32Value || int64(x) > maxInt32Value {
			return 0, false
		}
		return int32(x), true
	default:
		return 0, false
	}
}

func metadataInt32Array(kv map[string]interface{}, key string) ([]int32, bool, error) {
	v, ok := kv[key]
	if !ok {
		return nil, false, nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, true, fmt.Errorf("%s is %T, want array", key, v)
	}
	out := make([]int32, len(arr))
	for i, item := range arr {
		val, ok := metadataInt32Value(item)
		if !ok {
			return nil, true, fmt.Errorf("%s[%d] is %T, want int32-compatible integer", key, i, item)
		}
		out[i] = val
	}
	return out, true, nil
}

func metadataPositiveInt(kv map[string]interface{}, key string) (int, bool, error) {
	v, ok := kv[key]
	if !ok {
		return 0, false, nil
	}
	val, valid := metadataInt32Value(v)
	if !valid {
		return 0, true, fmt.Errorf("%s is %T, want int32-compatible integer", key, v)
	}
	if val <= 0 {
		return 0, true, fmt.Errorf("%s must be positive, got %d", key, val)
	}
	return int(val), true, nil
}

func metadataTokenID(kv map[string]interface{}, key string) (int, bool, error) {
	v, ok := kv[key]
	if !ok {
		return 0, false, nil
	}
	id, valid := metadataInt32Value(v)
	if !valid {
		return 0, true, fmt.Errorf("%s is %T, want int32-compatible integer", key, v)
	}
	if id < 0 {
		return 0, true, fmt.Errorf("%s is negative: %d", key, id)
	}
	return int(id), true, nil
}

func metadataBool(kv map[string]interface{}, key string) (bool, bool, error) {
	v, ok := kv[key]
	if !ok {
		return false, false, nil
	}
	b, ok := v.(bool)
	if !ok {
		return false, true, fmt.Errorf("%s is %T, want bool", key, v)
	}
	return b, true, nil
}

func validateModelMetadata(meta *GGUFMetadata, hasHeadDim bool) error {
	if meta == nil {
		return nil
	}
	if meta.NumHeads > 0 && meta.NumKVHeads > 0 && meta.NumKVHeads > meta.NumHeads {
		return fmt.Errorf("%s.attention.head_count_kv %d > %s.attention.head_count %d",
			meta.Architecture, meta.NumKVHeads, meta.Architecture, meta.NumHeads)
	}
	if meta.NumHeads > 0 && meta.NumKVHeads > 0 && meta.NumHeads%meta.NumKVHeads != 0 {
		return fmt.Errorf("%s.attention.head_count %d is not divisible by %s.attention.head_count_kv %d",
			meta.Architecture, meta.NumHeads, meta.Architecture, meta.NumKVHeads)
	}
	if meta.NumHeads > 0 && meta.EmbedDim > 0 && !hasHeadDim {
		if meta.EmbedDim%meta.NumHeads != 0 {
			return fmt.Errorf("%s.embedding_length %d is not divisible by %s.attention.head_count %d",
				meta.Architecture, meta.EmbedDim, meta.Architecture, meta.NumHeads)
		}
	}
	return nil
}

func validateTokenizerMetadata(meta *GGUFMetadata, hasBosID, hasEosID bool) error {
	if meta == nil || len(meta.TokenList) == 0 {
		return nil
	}
	seen := make(map[string]int, len(meta.TokenList))
	for i, tok := range meta.TokenList {
		if prev, ok := seen[tok]; ok {
			return fmt.Errorf("tokenizer.ggml.tokens[%d] duplicates tokenizer.ggml.tokens[%d]", i, prev)
		}
		seen[tok] = i
	}
	if len(meta.TokenScores) > 0 && len(meta.TokenScores) != len(meta.TokenList) {
		return fmt.Errorf("tokenizer.ggml.scores length %d != tokenizer.ggml.tokens length %d",
			len(meta.TokenScores), len(meta.TokenList))
	}
	if len(meta.TokenTypes) > 0 && len(meta.TokenTypes) != len(meta.TokenList) {
		return fmt.Errorf("tokenizer.ggml.token_type length %d != tokenizer.ggml.tokens length %d",
			len(meta.TokenTypes), len(meta.TokenList))
	}
	if hasBosID && (meta.BosID < 0 || meta.BosID >= len(meta.TokenList)) {
		return fmt.Errorf("tokenizer.ggml.bos_token_id %d out of vocab range 0..%d", meta.BosID, len(meta.TokenList)-1)
	}
	if hasEosID && (meta.EosID < 0 || meta.EosID >= len(meta.TokenList)) {
		return fmt.Errorf("tokenizer.ggml.eos_token_id %d out of vocab range 0..%d", meta.EosID, len(meta.TokenList)-1)
	}
	return nil
}

// parseMetadata extracts model config from GGUF KV pairs
func parseMetadata(kv map[string]interface{}) (GGUFMetadata, error) {
	meta := GGUFMetadata{
		KV:         kv,
		RMSNormEps: 1e-5,
		RopeTheta:  10000.0,
		BosID:      1,
		EosID:      2,
	}

	arch := "llama"
	if s, ok, err := metadataStringValue(kv, "general.architecture"); err != nil {
		return meta, err
	} else if ok {
		arch = s
	}
	meta.Architecture = arch

	// Model dimensions
	if n, ok, err := metadataPositiveInt(kv, arch+".block_count"); err != nil {
		return meta, err
	} else if ok {
		meta.NumLayers = n
	}
	if n, ok, err := metadataPositiveInt(kv, arch+".embedding_length"); err != nil {
		return meta, err
	} else if ok {
		meta.EmbedDim = n
	}
	if n, ok, err := metadataPositiveInt(kv, arch+".attention.head_count"); err != nil {
		return meta, err
	} else if ok {
		meta.NumHeads = n
	}
	if n, ok, err := metadataPositiveInt(kv, arch+".attention.head_count_kv"); err != nil {
		return meta, err
	} else if ok {
		meta.NumKVHeads = n
	}
	if n, ok, err := metadataPositiveInt(kv, arch+".feed_forward_length"); err != nil {
		return meta, err
	} else if ok {
		meta.IntermSize = n
	}
	if n, ok, err := metadataPositiveInt(kv, arch+".context_length"); err != nil {
		return meta, err
	} else if ok {
		meta.SeqLen = n
	}
	if f, ok, err := metadataPositiveFloat32(kv, arch+".attention.layer_norm_rms_epsilon"); err != nil {
		return meta, err
	} else if ok {
		meta.RMSNormEps = f
	}
	if f, ok, err := metadataPositiveFloat32(kv, arch+".rope.freq_base"); err != nil {
		return meta, err
	} else if ok {
		meta.RopeTheta = f
	}

	// Derived (Y-B1: prefer attention.key_length from header — Mistral hd=128 != 5120/32=160; Qwen-neutral)
	hasHeadDim := false
	if n, ok, err := metadataPositiveInt(kv, arch+".attention.key_length"); err != nil {
		return meta, err
	} else if ok {
		hasHeadDim = true
		meta.HeadDim = n
	} else if meta.NumHeads > 0 && meta.EmbedDim > 0 {
		meta.HeadDim = meta.EmbedDim / meta.NumHeads
	}
	if meta.NumKVHeads == 0 {
		meta.NumKVHeads = meta.NumHeads // MHA fallback
	}
	if err := validateModelMetadata(&meta, hasHeadDim); err != nil {
		return meta, err
	}

	// nanollama-specific flags
	if b, ok, err := metadataBool(kv, "nanollama.qk_norm"); err != nil {
		return meta, err
	} else if ok {
		meta.QKNorm = b
	}
	if b, ok, err := metadataBool(kv, "nanollama.rope_conjugate"); err != nil {
		return meta, err
	} else if ok {
		meta.RopeConjugate = b
	}

	// Tokenizer model type
	meta.TokenizerModel = "llama" // default: SentencePiece
	if s, ok, err := metadataStringValue(kv, "tokenizer.ggml.model"); err != nil {
		return meta, err
	} else if ok {
		meta.TokenizerModel = s
	}

	// Tokenizer
	hasTokens := false
	if tokens, ok, err := metadataStringArray(kv, "tokenizer.ggml.tokens"); err != nil {
		return meta, err
	} else if ok {
		hasTokens = true
		meta.TokenList = tokens
		meta.VocabSize = len(meta.TokenList)
	}
	hasScores := false
	if scores, ok, err := metadataFloat32Array(kv, "tokenizer.ggml.scores"); err != nil {
		return meta, err
	} else if ok {
		hasScores = true
		meta.TokenScores = scores
	}
	hasTypes := false
	if types, ok, err := metadataInt32Array(kv, "tokenizer.ggml.token_type"); err != nil {
		return meta, err
	} else if ok {
		hasTypes = true
		meta.TokenTypes = types
	}
	bosID, hasBosID, err := metadataTokenID(kv, "tokenizer.ggml.bos_token_id")
	if err != nil {
		return meta, err
	}
	if hasBosID {
		meta.BosID = bosID
	}
	eosID, hasEosID, err := metadataTokenID(kv, "tokenizer.ggml.eos_token_id")
	if err != nil {
		return meta, err
	}
	if hasEosID {
		meta.EosID = eosID
	}
	// BPE merges (GPT-2 style tokenizers)
	hasMerges := false
	if merges, ok, err := metadataStringArray(kv, "tokenizer.ggml.merges"); err != nil {
		return meta, err
	} else if ok {
		hasMerges = true
		meta.TokenMerges = merges
	}
	if !hasTokens && (hasScores || hasTypes || hasMerges) {
		return meta, fmt.Errorf("tokenizer side metadata present without tokenizer.ggml.tokens")
	}
	if hasTokens && len(meta.TokenList) == 0 {
		return meta, fmt.Errorf("tokenizer.ggml.tokens is empty")
	}
	for i, merge := range meta.TokenMerges {
		if strings.TrimSpace(merge) == "" {
			return meta, fmt.Errorf("tokenizer.ggml.merges[%d] is empty", i)
		}
	}
	if err := validateTokenizerMetadata(&meta, hasBosID, hasEosID); err != nil {
		return meta, err
	}

	// Default: add space prefix (standard SentencePiece behavior)
	meta.AddSpacePrefix = true
	if v, ok := kv["tokenizer.ggml.add_space_prefix"]; ok {
		switch val := v.(type) {
		case bool:
			meta.AddSpacePrefix = val
		case uint8:
			meta.AddSpacePrefix = val != 0
		case int:
			meta.AddSpacePrefix = val != 0
		case uint32:
			meta.AddSpacePrefix = val != 0
		}
	}

	fmt.Printf("[tongue/gguf] arch=%s layers=%d dim=%d heads=%d kv_heads=%d head_dim=%d\n",
		arch, meta.NumLayers, meta.EmbedDim, meta.NumHeads, meta.NumKVHeads, meta.HeadDim)
	fmt.Printf("[tongue/gguf] vocab=%d seq_len=%d ffn=%d rope_theta=%.1f tokenizer=%s\n",
		meta.VocabSize, meta.SeqLen, meta.IntermSize, meta.RopeTheta, meta.TokenizerModel)
	if len(meta.TokenMerges) > 0 {
		fmt.Printf("[tongue/gguf] BPE merges=%d\n", len(meta.TokenMerges))
	}

	return meta, nil
}

// GetTensor returns raw bytes for a named tensor
func (g *GGUFFile) GetTensor(name string) ([]byte, *GGUFTensorInfo, error) {
	info, ok := g.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor not found: %s", name)
	}
	size, err := tensorBytes(info)
	if err != nil {
		return nil, nil, fmt.Errorf("tensor %s invalid layout: %w", name, err)
	}
	start := info.Offset
	if start > uint64(len(g.TensorData)) {
		return nil, nil, fmt.Errorf("tensor %s out of bounds: %d > %d",
			name, start, len(g.TensorData))
	}
	if size > uint64(len(g.TensorData))-start {
		return nil, nil, fmt.Errorf("tensor %s out of bounds: %d + %d > %d",
			name, start, size, len(g.TensorData))
	}
	end := start + size
	return g.TensorData[start:end], info, nil
}

// FindTensor searches for a tensor by substring match
func (g *GGUFFile) FindTensor(substr string) (*GGUFTensorInfo, bool) {
	for name, info := range g.Tensors {
		if strings.Contains(name, substr) {
			return info, true
		}
	}
	return nil, false
}

// ListTensors prints all tensors (debug)
func (g *GGUFFile) ListTensors() {
	for name, info := range g.Tensors {
		fmt.Printf("  %-50s  type=%d  dims=[", name, info.Type)
		for d := uint32(0); d < info.NDims; d++ {
			if d > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%d", info.Dims[d])
		}
		size, err := tensorBytes(info)
		if err != nil {
			fmt.Printf("]  invalid: %v\n", err)
			continue
		}
		fmt.Printf("]  %.2f MB\n", float64(size)/1024/1024)
	}
}

// half2floatLUT is a precomputed lookup table for all 65536 fp16 values.
// 256KB, fits in L2 cache. Eliminates branching in the hottest matmul paths.
var half2floatLUT [65536]float32

func init() {
	for h := 0; h < 65536; h++ {
		sign := uint32(h>>15) & 1
		exp := uint32(h>>10) & 0x1F
		mant := uint32(h & 0x3FF)

		var f uint32
		if exp == 0 {
			if mant == 0 {
				f = sign << 31
			} else {
				e := uint32(1)
				for mant&0x400 == 0 {
					mant <<= 1
					e--
				}
				mant &= 0x3FF
				f = (sign << 31) | ((e + 127 - 15) << 23) | (mant << 13)
			}
		} else if exp == 0x1F {
			f = (sign << 31) | 0x7F800000 | (mant << 13)
		} else {
			f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
		}
		half2floatLUT[h] = math.Float32frombits(f)
	}
}

// half2float converts IEEE 754 binary16 to float32 via lookup table
func half2float(h uint16) float32 {
	return half2floatLUT[h]
}
