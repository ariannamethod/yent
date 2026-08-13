package yent

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTensorBytesValidLayouts(t *testing.T) {
	tests := []struct {
		name string
		info GGUFTensorInfo
		want uint64
	}{
		{
			name: "f32 scalars",
			info: GGUFTensorInfo{Name: "f32", NDims: 1, Dims: [4]uint64{3}, Type: ggmlTypeF32},
			want: 12,
		},
		{
			name: "q4_0 whole blocks",
			info: GGUFTensorInfo{Name: "q4", NDims: 1, Dims: [4]uint64{64}, Type: ggmlTypeQ4_0},
			want: 36,
		},
		{
			name: "q6_k whole rows",
			info: GGUFTensorInfo{Name: "q6k", NDims: 2, Dims: [4]uint64{256, 2}, Type: ggmlTypeQ6_K},
			want: 420,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tensorBytes(&tt.info)
			if err != nil {
				t.Fatalf("tensorBytes error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("tensorBytes = %d want %d", got, tt.want)
			}
		})
	}
}

func TestTensorBytesRejectsMalformedLayouts(t *testing.T) {
	tests := []struct {
		name string
		info GGUFTensorInfo
		want string
	}{
		{
			name: "zero dimension",
			info: GGUFTensorInfo{Name: "zero", NDims: 1, Dims: [4]uint64{0}, Type: ggmlTypeF32},
			want: "zero dimension",
		},
		{
			name: "partial q4 block",
			info: GGUFTensorInfo{Name: "partial", NDims: 1, Dims: [4]uint64{33}, Type: ggmlTypeQ4_0},
			want: "not whole blocks",
		},
		{
			name: "unsupported type",
			info: GGUFTensorInfo{Name: "unknown", NDims: 1, Dims: [4]uint64{256}, Type: ggmlTypeQ5_K},
			want: "unsupported tensor type",
		},
		{
			name: "element overflow",
			info: GGUFTensorInfo{Name: "overflow", NDims: 2, Dims: [4]uint64{^uint64(0), 2}, Type: ggmlTypeF32},
			want: "element count overflows",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tensorBytes(&tt.info)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("tensorBytes error = %v want containing %q", err, tt.want)
			}
		})
	}
}

func TestGGUFGetTensorRejectsInvalidLayoutBeforeSlicing(t *testing.T) {
	g := &GGUFFile{
		TensorData: make([]byte, q4BytesPerBlock*2),
		Tensors: map[string]*GGUFTensorInfo{
			"bad": {
				Name:  "bad",
				NDims: 1,
				Dims:  [4]uint64{33},
				Type:  ggmlTypeQ4_0,
			},
		},
	}

	data, info, err := g.GetTensor("bad")
	if err == nil || !strings.Contains(err.Error(), "invalid layout") ||
		!strings.Contains(err.Error(), "not whole blocks") {
		t.Fatalf("GetTensor error = %v", err)
	}
	if data != nil || info != nil {
		t.Fatalf("invalid tensor returned data/info: data=%v info=%+v", data, info)
	}
}

func TestGGUFGetTensorRejectsOutOfBoundsWithoutOffsetWrap(t *testing.T) {
	g := &GGUFFile{
		TensorData: make([]byte, q4BytesPerBlock),
		Tensors: map[string]*GGUFTensorInfo{
			"past": {
				Name:   "past",
				NDims:  1,
				Dims:   [4]uint64{32},
				Type:   ggmlTypeQ4_0,
				Offset: uint64(q4BytesPerBlock + 1),
			},
			"short": {
				Name:   "short",
				NDims:  1,
				Dims:   [4]uint64{64},
				Type:   ggmlTypeQ4_0,
				Offset: 0,
			},
		},
	}

	for _, name := range []string{"past", "short"} {
		data, info, err := g.GetTensor(name)
		if err == nil || !strings.Contains(err.Error(), "out of bounds") {
			t.Fatalf("GetTensor(%q) error = %v want out of bounds", name, err)
		}
		if data != nil || info != nil {
			t.Fatalf("out-of-bounds tensor %q returned data/info: data=%v info=%+v", name, data, info)
		}
	}
}

func TestLoadGGUFRejectsInvalidTensorInfo(t *testing.T) {
	tests := []struct {
		name        string
		tensors     []ggufTestTensor
		want        string
		wantNoPanic bool
	}{
		{
			name:        "ndims over four",
			tensors:     []ggufTestTensor{{name: "bad.ndims", ndims: 5, dims: []uint64{1, 1, 1, 1, 1}, typ: ggmlTypeF32}},
			want:        "invalid ndim 5",
			wantNoPanic: true,
		},
		{
			name:    "duplicate tensor",
			tensors: []ggufTestTensor{{name: "dup", dims: []uint64{1}, typ: ggmlTypeF32}, {name: "dup", dims: []uint64{1}, typ: ggmlTypeF32}},
			want:    "duplicate tensor name",
		},
		{
			name:    "partial q4 tensor",
			tensors: []ggufTestTensor{{name: "partial", dims: []uint64{33}, typ: ggmlTypeQ4_0}},
			want:    "not whole blocks",
		},
		{
			name:    "unsupported tensor type",
			tensors: []ggufTestTensor{{name: "unsupported", dims: []uint64{256}, typ: ggmlTypeQ5_K}},
			want:    "unsupported tensor type",
		},
		{
			name:    "offset beyond tensor data",
			tensors: []ggufTestTensor{{name: "past", dims: []uint64{1}, typ: ggmlTypeF32, offset: 8}},
			want:    "offset 8 > data size 4",
		},
		{
			name:    "tensor range beyond tensor data",
			tensors: []ggufTestTensor{{name: "short", dims: []uint64{2}, typ: ggmlTypeF32, offset: 0}},
			want:    "offset 0 + size 8 > data size 4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataBytes := 64
			if strings.Contains(tt.name, "tensor data") {
				dataBytes = 4
			}
			path := writeGGUFTestFile(t, tt.tensors, dataBytes)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("LoadGGUF panicked: %v", r)
				}
			}()
			_, err := LoadGGUF(path)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("LoadGGUF error = %v want containing %q", err, tt.want)
			}
		})
	}
}

func TestLoadGGUFRejectsAbsurdHeaderCounts(t *testing.T) {
	tests := []struct {
		name          string
		tensorCount   uint64
		metadataCount uint64
		want          string
	}{
		{
			name:        "tensor count",
			tensorCount: maxGGUFTensorCount + 1,
			want:        "tensor count too large",
		},
		{
			name:          "metadata count",
			metadataCount: maxGGUFMetadataCount + 1,
			want:          "metadata count too large",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeGGUFHeaderOnlyTestFile(t, tt.tensorCount, tt.metadataCount)
			_, err := LoadGGUF(path)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("LoadGGUF error = %v want containing %q", err, tt.want)
			}
		})
	}
}

type ggufTestTensor struct {
	name   string
	ndims  uint32
	dims   []uint64
	typ    uint32
	offset uint64
}

func writeGGUFTestFile(t *testing.T, tensors []ggufTestTensor, dataBytes int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	writeTestLE(t, f, uint32(ggufMagic))
	writeTestLE(t, f, uint32(ggufVersion))
	writeTestLE(t, f, uint64(len(tensors)))
	writeTestLE(t, f, uint64(0))

	for _, tensor := range tensors {
		writeTestGGUFString(t, f, tensor.name)
		ndims := tensor.ndims
		if ndims == 0 && len(tensor.dims) > 0 {
			ndims = uint32(len(tensor.dims))
		}
		writeTestLE(t, f, ndims)
		for _, dim := range tensor.dims {
			writeTestLE(t, f, dim)
		}
		writeTestLE(t, f, tensor.typ)
		writeTestLE(t, f, tensor.offset)
	}

	if pos, err := f.Seek(0, 1); err != nil {
		t.Fatal(err)
	} else {
		pad := (32 - (pos % 32)) % 32
		if pad > 0 {
			if _, err := f.Write(make([]byte, pad)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if dataBytes > 0 {
		if _, err := f.Write(make([]byte, dataBytes)); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func writeGGUFHeaderOnlyTestFile(t *testing.T, tensorCount, metadataCount uint64) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "header.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	writeTestLE(t, f, uint32(ggufMagic))
	writeTestLE(t, f, uint32(ggufVersion))
	writeTestLE(t, f, tensorCount)
	writeTestLE(t, f, metadataCount)
	return path
}

func writeTestGGUFString(t *testing.T, f *os.File, s string) {
	t.Helper()
	writeTestLE(t, f, uint64(len(s)))
	if _, err := f.WriteString(s); err != nil {
		t.Fatal(err)
	}
}

func writeTestLE(t *testing.T, f *os.File, v interface{}) {
	t.Helper()
	if err := binary.Write(f, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
}
