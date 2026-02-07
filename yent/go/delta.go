package yent

// delta.go — Delta Voice: multilingual recovery via DSL-controlled delta injection
//
// "from ariannamethod import Destiny"
//
// Architecture:
//   delta = base_qwen_lm_head - yent_lm_head (precomputed, stored as SVD factors)
//   logits += alpha * A @ (B @ hidden_state)
//
//   alpha = 0.0 → pure Yent English
//   alpha = 0.5 → Yent + multilingual (29 languages)
//   alpha = 1.0 → base Qwen distribution (no personality)
//
// The delta is stored as NPZ (numpy compressed) with float16 A and B matrices.
// A: [vocab_size, rank]   — output projection
// B: [rank, hidden_dim]   — input projection
//
// Cost per token: rank × (vocab + hidden) FMA ops ≈ 10M for rank=64
// This is ~2% of a full forward pass. Negligible.

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strings"
)

// DeltaVoice holds the low-rank delta for multilingual recovery
type DeltaVoice struct {
	VocabSize int
	HiddenDim int
	Rank      int

	// A: [VocabSize × Rank] stored as float32 (converted from float16 on load)
	A []float32
	// B: [Rank × HiddenDim] stored as float32
	B []float32

	// Scratch buffer for B @ x computation
	Bx []float32 // [Rank]
}

// LoadDelta loads a delta voice file from NPZ format
// Expected entries: A.npy, B.npy (float16, C-order)
func LoadDelta(path string) (*DeltaVoice, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open delta npz: %w", err)
	}
	defer r.Close()

	var aData, bData []float32
	var aShape, bShape [2]int

	for _, f := range r.File {
		name := f.Name
		if !strings.HasSuffix(name, ".npy") {
			continue
		}

		// Only load A.npy and B.npy — skip scalar metadata (rank, vocab_size, etc.)
		isA := name == "A.npy"
		isB := name == "B.npy"
		if !isA && !isB {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("open %s: %w", name, err)
		}

		data, shape, err := readNpy(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", name, err)
		}

		if isA {
			aData = data
			aShape = shape
		} else {
			bData = data
			bShape = shape
		}
	}

	if aData == nil || bData == nil {
		return nil, fmt.Errorf("delta npz missing A.npy or B.npy")
	}

	// Validate shapes
	vocabSize := aShape[0]
	rank := aShape[1]
	if bShape[0] != rank {
		return nil, fmt.Errorf("rank mismatch: A has rank %d, B has %d", rank, bShape[0])
	}
	hiddenDim := bShape[1]

	fmt.Printf("[delta-voice] loaded: vocab=%d, hidden=%d, rank=%d\n", vocabSize, hiddenDim, rank)
	fmt.Printf("[delta-voice] A: %d×%d (%.1f MB), B: %d×%d (%.1f MB)\n",
		vocabSize, rank, float64(len(aData)*4)/1024/1024,
		rank, hiddenDim, float64(len(bData)*4)/1024/1024)

	return &DeltaVoice{
		VocabSize: vocabSize,
		HiddenDim: hiddenDim,
		Rank:      rank,
		A:         aData,
		B:         bData,
		Bx:        make([]float32, rank),
	}, nil
}

// ApplyToLogits adds alpha * A @ (B @ x) to logits
// logits: [VocabSize], x: [HiddenDim], alpha: blend factor
func (d *DeltaVoice) ApplyToLogits(logits []float32, x []float32, alpha float32) {
	if alpha == 0 || d == nil {
		return
	}

	rank := d.Rank
	hiddenDim := d.HiddenDim
	vocabSize := d.VocabSize

	// Step 1: Bx = B @ x → [rank]
	// B is [rank, hiddenDim], x is [hiddenDim]
	for r := 0; r < rank; r++ {
		var sum float32
		off := r * hiddenDim
		for j := 0; j < hiddenDim; j++ {
			sum += d.B[off+j] * x[j]
		}
		d.Bx[r] = sum
	}

	// Step 2: logits += alpha * A @ Bx
	// A is [vocabSize, rank], Bx is [rank]
	for i := 0; i < vocabSize; i++ {
		var sum float32
		off := i * rank
		for r := 0; r < rank; r++ {
			sum += d.A[off+r] * d.Bx[r]
		}
		logits[i] += alpha * sum
	}
}

// readNpy reads a numpy .npy file and returns float32 data + 2D shape
// Supports float16 and float32 dtypes
func readNpy(r io.Reader) ([]float32, [2]int, error) {
	// Magic: \x93NUMPY
	magic := make([]byte, 6)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, [2]int{}, fmt.Errorf("read magic: %w", err)
	}
	if magic[0] != 0x93 || string(magic[1:6]) != "NUMPY" {
		return nil, [2]int{}, fmt.Errorf("not a npy file")
	}

	// Version
	ver := make([]byte, 2)
	if _, err := io.ReadFull(r, ver); err != nil {
		return nil, [2]int{}, fmt.Errorf("read version: %w", err)
	}

	// Header length
	var headerLen int
	if ver[0] == 1 {
		hl := make([]byte, 2)
		if _, err := io.ReadFull(r, hl); err != nil {
			return nil, [2]int{}, fmt.Errorf("read header len: %w", err)
		}
		headerLen = int(binary.LittleEndian.Uint16(hl))
	} else {
		hl := make([]byte, 4)
		if _, err := io.ReadFull(r, hl); err != nil {
			return nil, [2]int{}, fmt.Errorf("read header len v2: %w", err)
		}
		headerLen = int(binary.LittleEndian.Uint32(hl))
	}

	// Header string (Python dict)
	header := make([]byte, headerLen)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, [2]int{}, fmt.Errorf("read header: %w", err)
	}
	hstr := string(header)

	// Parse dtype
	isFloat16 := strings.Contains(hstr, "'<f2'") || strings.Contains(hstr, "float16")
	isFloat32 := strings.Contains(hstr, "'<f4'") || strings.Contains(hstr, "float32")
	if !isFloat16 && !isFloat32 {
		return nil, [2]int{}, fmt.Errorf("unsupported dtype in header: %s", hstr)
	}

	// Parse shape — find (N, M) in header
	shape := parseShape(hstr)
	if shape[0] == 0 || shape[1] == 0 {
		return nil, [2]int{}, fmt.Errorf("could not parse shape from header: %s", hstr)
	}

	totalElements := shape[0] * shape[1]

	// Read raw data
	var data []float32
	if isFloat16 {
		raw := make([]byte, totalElements*2)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, [2]int{}, fmt.Errorf("read float16 data: %w", err)
		}
		data = make([]float32, totalElements)
		for i := 0; i < totalElements; i++ {
			h := uint16(raw[i*2]) | uint16(raw[i*2+1])<<8
			data[i] = half2float(h)
		}
	} else {
		raw := make([]byte, totalElements*4)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, [2]int{}, fmt.Errorf("read float32 data: %w", err)
		}
		data = make([]float32, totalElements)
		for i := 0; i < totalElements; i++ {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}

	return data, shape, nil
}

// parseShape extracts (rows, cols) from npy header string
// Header looks like: {'descr': '<f2', 'fortran_order': False, 'shape': (151936, 64), }
func parseShape(header string) [2]int {
	idx := strings.Index(header, "shape")
	if idx < 0 {
		return [2]int{}
	}

	// Find opening paren
	start := strings.Index(header[idx:], "(")
	if start < 0 {
		return [2]int{}
	}
	start += idx + 1

	// Find closing paren
	end := strings.Index(header[start:], ")")
	if end < 0 {
		return [2]int{}
	}

	shapeStr := header[start : start+end]
	shapeStr = strings.TrimSpace(shapeStr)

	// Parse "N, M"
	parts := strings.Split(shapeStr, ",")
	if len(parts) < 2 {
		return [2]int{}
	}

	var shape [2]int
	fmt.Sscanf(strings.TrimSpace(parts[0]), "%d", &shape[0])
	fmt.Sscanf(strings.TrimSpace(parts[1]), "%d", &shape[1])
	return shape
}
