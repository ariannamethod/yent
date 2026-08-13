package yent

import (
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
