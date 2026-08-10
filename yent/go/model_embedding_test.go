package yent

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func f32RowBytes(vals ...float32) []byte {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:i*4+4], math.Float32bits(v))
	}
	return data
}

func filledQ6KRow(seed byte) []byte {
	row := make([]byte, q6kBytesPerBlock)
	for i := range row {
		row[i] = byte((int(seed) + i*17) & 0xFF)
	}
	binary.LittleEndian.PutUint16(row[208:210], float2half(1.0))
	return row
}

func TestEmbeddingRowLookupF32ChecksRows(t *testing.T) {
	data := f32RowBytes(
		1, 2, 3,
		4, 5, 6,
	)
	out := []float32{-1, -1, -1}

	if err := embedLookupInto(out, data, ggmlTypeF32, 1, 3); err != nil {
		t.Fatalf("lookup row 1: %v", err)
	}
	want := []float32{4, 5, 6}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%v want %v", i, out[i], want[i])
		}
	}

	before := append([]float32(nil), out...)
	err := embedLookupInto(out, data, ggmlTypeF32, 2, 3)
	if err == nil || !strings.Contains(err.Error(), "outside F32 table") {
		t.Fatalf("out-of-range error = %v", err)
	}
	for i := range out {
		if out[i] != before[i] {
			t.Fatalf("failed lookup changed out[%d]: got %v want %v", i, out[i], before[i])
		}
	}
}

func TestEmbeddingRowLookupRejectsMalformedTables(t *testing.T) {
	out := make([]float32, q4BlockSize)
	if err := embedLookupInto(out, make([]byte, q4BytesPerBlock), ggmlTypeQ4_0, 0, q4BlockSize-1); err == nil {
		t.Fatal("expected non-whole Q4_0 row dim to fail")
	}
	if err := embedLookupInto(out, make([]byte, q4BytesPerBlock-1), ggmlTypeQ4_0, 0, q4BlockSize); err == nil {
		t.Fatal("expected partial Q4_0 table to fail")
	}
	if err := embedLookupInto(out, make([]byte, q4BytesPerBlock), ggmlTypeQ5_K, 0, q4BlockSize); err == nil {
		t.Fatal("expected unsupported embedding dtype to fail")
	}
}

func TestEmbeddingRowLookupQ6KUsesCheckedRowWithoutAllocation(t *testing.T) {
	row0 := filledQ6KRow(3)
	row1 := filledQ6KRow(19)
	data := append(append([]byte{}, row0...), row1...)
	out := make([]float32, q6kBlockSize)

	if err := embedLookupInto(out, data, ggmlTypeQ6_K, 1, q6kBlockSize); err != nil {
		t.Fatalf("lookup Q6_K row: %v", err)
	}
	want := DequantQ6_K(row1, q6kBlockSize)
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%v want %v", i, out[i], want[i])
		}
	}

	allocs := testing.AllocsPerRun(100, func() {
		if err := embedLookupInto(out, data, ggmlTypeQ6_K, 1, q6kBlockSize); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Q6_K embedding lookup allocated: got %.1f want 0", allocs)
	}
}

func TestForwardErrRejectsInvalidPosition(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{
			EmbedDim:   1,
			NumHeads:   1,
			NumKVHeads: 1,
			HeadDim:    1,
			SeqLen:     1,
		},
	}
	err := m.ForwardErr(0, 1)
	if err == nil || !strings.Contains(err.Error(), "outside seq_len") {
		t.Fatalf("position error = %v", err)
	}
}
