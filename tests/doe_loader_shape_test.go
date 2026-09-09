package tests

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	yent "github.com/ariannamethod/yent/yent/go"
)

type doeFixtureTensor struct {
	name   string
	dims   []uint64
	offset uint64
}

type doeLoaderFaultCase struct {
	name       string
	tensors    []doeFixtureTensor
	want       string
	wantNoText string
}

func TestDOEHostLoaderRejectsShapeAndCompletenessFaults(t *testing.T) {
	exe := buildDOEForLoaderTest(t)
	runDOEHostLoaderFaultCases(t, exe, nil)
}

func TestDOEHostLoaderRejectsShapeAndCompletenessFaultsUnderASan(t *testing.T) {
	exe := buildDOEForLoaderTestWithFlags(t, "asan", []string{
		"-O1",
		"-g",
		"-fsanitize=address",
		"-fno-omit-frame-pointer",
	})
	runDOEHostLoaderFaultCases(t, exe, []string{
		"ASAN_OPTIONS=halt_on_error=1:abort_on_error=1",
	})
}

func TestDOEHostLoaderExercisesNonzeroParliamentElection(t *testing.T) {
	exe := buildDOEForLoaderTest(t)
	modelPath := filepath.Join(t.TempDir(), "parliament.gguf")
	if err := writeMinimalDOEHostGGUFWithTokenizer(modelPath, baseDOELoaderTensors()); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe,
		"--model", modelPath,
		"--max-new", "1",
		"--train", "0",
		"--field-gain", "0",
		"--lora-alpha", "0.1",
		"--temp", "0",
		"--top-k", "1",
		"--no-load-spore",
		"--no-save-spore")
	cmd.Dir = t.TempDir()
	cmd.Stdin = strings.NewReader("hi\nstatus\nexit\n")
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("doe_field timed out; output:\n%s", string(out))
	}
	if err != nil {
		t.Fatalf("doe_field nonzero parliament smoke failed: %v\n%s", err, string(out))
	}
	text := string(out)
	if !strings.Contains(text, "alpha=0.10") {
		t.Fatalf("nonzero LoRA alpha was not acknowledged; output:\n%s", text)
	}
	elections, ok := extractDOEStatusElections(text)
	if !ok {
		t.Fatalf("status output did not include parliament elections; output:\n%s", text)
	}
	if elections <= 0 {
		t.Fatalf("nonzero LoRA path did not run a parliament election; output:\n%s", text)
	}
}

func TestDOEBodyCapturesRealDOEForcedSlotFailureDiagnostics(t *testing.T) {
	exe := buildDOEForLoaderTestWithFlags(t, "testing", []string{"-O0", "-DDOE_TESTING"})
	modelPath := filepath.Join(t.TempDir(), "diagnostics.gguf")
	if err := writeMinimalDOEHostGGUFWithTokenizer(modelPath, baseDOELoaderTensors()); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	body, err := yent.NewDOEBody(yent.DOEBodyConfig{
		Name:      "nemo12",
		BinPath:   exe,
		ModelPath: modelPath,
		WorkDir:   t.TempDir(),
		Args: []string{
			"--max-new", "1",
			"--train", "0",
			"--field-gain", "0",
			"--lora-alpha", "0.1",
			"--temp", "0",
			"--top-k", "1",
			"--no-load-spore",
			"--no-save-spore",
		},
		Env:          []string{"DOE_TEST_FORCE_SLOT_FAIL=nt_metal_slot_download(SLOT_X)"},
		Timeout:      5 * time.Second,
		PrimeTimeout: 5 * time.Second,
		DisableLease: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()

	out, err := body.Generate("hi", "")
	if err != nil {
		t.Fatal(err)
	}
	if out.ExecutionPath != "doe_resident" {
		t.Fatalf("execution path = %q, want doe_resident", out.ExecutionPath)
	}
	joined := strings.Join(out.Diagnostics, "\n")
	for _, want := range []string{
		"Metal slot path failed",
		"test-forced slot path",
		"nt_metal_slot_download(SLOT_X)",
		"rc=77",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("real DOE diagnostics missing %q:\n%v", want, out.Diagnostics)
		}
	}
}

func doeLoaderFaultCases() []doeLoaderFaultCase {
	return []doeLoaderFaultCase{
		{
			name:       "bad_token_embedding_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "token_embd.weight", []uint64{65, 8}),
			want:       "tensor shape mismatch for token_embd.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_output_vocab_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "output.weight", []uint64{64, 9}),
			want:       "tensor vocab mismatch for output.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_output_norm_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "output_norm.weight", []uint64{63}),
			want:       "tensor shape mismatch for output_norm.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_attention_q_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.attn_q.weight", []uint64{64, 65}),
			want:       "tensor shape mismatch for blk.0.attn_q.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_attention_k_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.attn_k.weight", []uint64{64, 65}),
			want:       "tensor shape mismatch for blk.0.attn_k.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_attention_v_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.attn_v.weight", []uint64{64, 65}),
			want:       "tensor shape mismatch for blk.0.attn_v.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_attention_output_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.attn_output.weight", []uint64{65, 64}),
			want:       "tensor shape mismatch for blk.0.attn_output.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_attention_norm_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.attn_norm.weight", []uint64{63}),
			want:       "tensor shape mismatch for blk.0.attn_norm.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_ffn_gate_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.ffn_gate.weight", []uint64{64, 127}),
			want:       "tensor shape mismatch for blk.0.ffn_gate.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_ffn_up_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.ffn_up.weight", []uint64{64, 129}),
			want:       "tensor shape mismatch for blk.0.ffn_up.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_ffn_down_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.ffn_down.weight", []uint64{127, 64}),
			want:       "tensor shape mismatch for blk.0.ffn_down.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "bad_ffn_norm_shape",
			tensors:    mutateDOETensorDims(baseDOELoaderTensors(), "blk.0.ffn_norm.weight", []uint64{63}),
			want:       "tensor shape mismatch for blk.0.ffn_norm.weight",
			wantNoText: "[sonar] profiling",
		},
		{
			name:       "missing_attention_k",
			tensors:    dropDOETensor(baseDOELoaderTensors(), "blk.0.attn_k.weight"),
			want:       "host layer 0 incomplete",
			wantNoText: "[sonar] profiling",
		},
	}
}

func runDOEHostLoaderFaultCases(t *testing.T, exe string, env []string) {
	t.Helper()
	outDir := t.TempDir()
	for _, tc := range doeLoaderFaultCases() {
		t.Run(tc.name, func(t *testing.T) {
			modelPath := filepath.Join(outDir, tc.name+".gguf")
			if err := writeMinimalDOEHostGGUF(modelPath, tc.tensors); err != nil {
				t.Fatalf("write fixture: %v", err)
			}
			out, err := runDOEExpectingLoadFailure(t, exe, modelPath, env)
			if err == nil {
				t.Fatalf("doe_field accepted malformed GGUF; output:\n%s", out)
			}
			if strings.Contains(out, "ERROR: AddressSanitizer") {
				t.Fatalf("ASan reported a loader memory error:\n%s", out)
			}
			if !strings.Contains(out, tc.want) {
				t.Fatalf("missing diagnostic %q in output:\n%s", tc.want, out)
			}
			if tc.wantNoText != "" && strings.Contains(out, tc.wantNoText) {
				t.Fatalf("loader reached forbidden phase %q; output:\n%s", tc.wantNoText, out)
			}
		})
	}
}

func buildDOEForLoaderTest(t *testing.T) string {
	t.Helper()
	return buildDOEForLoaderTestWithFlags(t, "native", nil)
}

func buildDOEForLoaderTestWithFlags(t *testing.T, suffix string, extraFlags []string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("doe.c loader smoke is POSIX-only")
	}
	if _, err := exec.LookPath("cc"); err != nil {
		t.Skipf("cc not found: %v", err)
	}
	root := repoRootForTest(t)
	dir := t.TempDir()
	stubPath := filepath.Join(dir, "pv_stub.c")
	stub := `#include <stdlib.h>
float *pv_encode_image(const char *img_path, const char *mmproj_path, int *o_ntok, int *o_dim) {
    (void)img_path; (void)mmproj_path; (void)o_ntok; (void)o_dim;
    return NULL;
}
`
	if err := os.WriteFile(stubPath, []byte(stub), 0o600); err != nil {
		t.Fatalf("write pixtral stub: %v", err)
	}
	exe := filepath.Join(dir, "doe_field_loader_test_"+suffix)
	args := []string{"-O0"}
	if len(extraFlags) > 0 {
		args = append([]string{}, extraFlags...)
	}
	args = append(args, "-Wall", "-Wextra",
		filepath.Join(root, "DoE", "doe.c"),
		stubPath,
		"-lm", "-lpthread", "-o", exe)
	cmd := exec.Command("cc", args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		if len(extraFlags) > 0 && sanitizerUnavailable(string(out)) {
			t.Skipf("C sanitizer unavailable: %v\n%s", err, string(out))
		}
		t.Fatalf("compile doe loader test binary: %v\n%s", err, string(out))
	}
	return exe
}

func sanitizerUnavailable(out string) bool {
	lower := strings.ToLower(out)
	return strings.Contains(lower, "unsupported") ||
		strings.Contains(lower, "unrecognized") ||
		strings.Contains(lower, "unknown argument") ||
		strings.Contains(lower, "invalid argument") ||
		strings.Contains(lower, "fsanitize=address")
}

func repoRootForTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		next := filepath.Dir(dir)
		if next == dir {
			t.Fatalf("could not find repo root from %s", dir)
		}
		dir = next
	}
}

func runDOEExpectingLoadFailure(t *testing.T, exe, modelPath string, env []string) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe,
		"--model", modelPath,
		"--once",
		"--max-new", "1",
		"--train", "0",
		"--field-gain", "0",
		"--lora-alpha", "0",
		"--no-load-spore",
		"--no-save-spore")
	cmd.Dir = t.TempDir()
	if len(env) > 0 {
		cmd.Env = append(os.Environ(), env...)
	}
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("doe_field timed out; output:\n%s", string(out))
	}
	return string(out), err
}

func baseDOELoaderTensors() []doeFixtureTensor {
	return []doeFixtureTensor{
		{name: "token_embd.weight", dims: []uint64{64, 8}},
		{name: "output_norm.weight", dims: []uint64{64}},
		{name: "output.weight", dims: []uint64{64, 8}},
		{name: "blk.0.attn_norm.weight", dims: []uint64{64}},
		{name: "blk.0.ffn_norm.weight", dims: []uint64{64}},
		{name: "blk.0.attn_q.weight", dims: []uint64{64, 64}},
		{name: "blk.0.attn_k.weight", dims: []uint64{64, 64}},
		{name: "blk.0.attn_v.weight", dims: []uint64{64, 64}},
		{name: "blk.0.attn_output.weight", dims: []uint64{64, 64}},
		{name: "blk.0.ffn_gate.weight", dims: []uint64{64, 128}},
		{name: "blk.0.ffn_up.weight", dims: []uint64{64, 128}},
		{name: "blk.0.ffn_down.weight", dims: []uint64{128, 64}},
	}
}

func mutateDOETensorDims(tensors []doeFixtureTensor, name string, dims []uint64) []doeFixtureTensor {
	out := cloneDOETensors(tensors)
	for i := range out {
		if out[i].name == name {
			out[i].dims = append([]uint64(nil), dims...)
			return out
		}
	}
	panic("unknown tensor " + name)
}

func dropDOETensor(tensors []doeFixtureTensor, name string) []doeFixtureTensor {
	out := make([]doeFixtureTensor, 0, len(tensors)-1)
	for _, tensor := range tensors {
		if tensor.name == name {
			continue
		}
		out = append(out, doeFixtureTensor{
			name: tensor.name,
			dims: append([]uint64(nil), tensor.dims...),
		})
	}
	if len(out) == len(tensors) {
		panic("unknown tensor " + name)
	}
	return out
}

func cloneDOETensors(tensors []doeFixtureTensor) []doeFixtureTensor {
	out := make([]doeFixtureTensor, len(tensors))
	for i, tensor := range tensors {
		out[i] = doeFixtureTensor{
			name: tensor.name,
			dims: append([]uint64(nil), tensor.dims...),
		}
	}
	return out
}

func writeMinimalDOEHostGGUF(path string, tensors []doeFixtureTensor) error {
	return writeMinimalDOEHostGGUFConfig(path, tensors, nil)
}

func writeMinimalDOEHostGGUFWithTokenizer(path string, tensors []doeFixtureTensor) error {
	return writeMinimalDOEHostGGUFConfig(path, tensors, []string{
		"<unk>", "h", "i", " ", "a", "e", "n", "t",
	})
}

func writeMinimalDOEHostGGUFConfig(path string, tensors []doeFixtureTensor, tokens []string) error {
	tensors = cloneDOETensors(tensors)
	var dataBytes uint64
	for i := range tensors {
		tensors[i].offset = dataBytes
		n, err := doeTensorF32Bytes(tensors[i].dims)
		if err != nil {
			return fmt.Errorf("%s: %w", tensors[i].name, err)
		}
		dataBytes += n
	}

	var buf bytes.Buffer
	writeLE(&buf, uint32(0x46554747)) // GGUF
	writeLE(&buf, uint32(3))
	writeLE(&buf, uint64(len(tensors)))
	nKV := uint64(10)
	if len(tokens) > 0 {
		nKV++
	}
	writeLE(&buf, nKV)
	writeGGUFStringKV(&buf, "general.architecture", "llama")
	writeGGUFUint32KV(&buf, "llama.embedding_length", 64)
	writeGGUFUint32KV(&buf, "llama.block_count", 1)
	writeGGUFUint32KV(&buf, "llama.attention.head_count", 1)
	writeGGUFUint32KV(&buf, "llama.attention.head_count_kv", 1)
	writeGGUFUint32KV(&buf, "llama.attention.key_length", 64)
	writeGGUFUint32KV(&buf, "llama.feed_forward_length", 128)
	writeGGUFUint32KV(&buf, "llama.vocab_size", 8)
	writeGGUFFloat32KV(&buf, "llama.rope.freq_base", 10000)
	writeGGUFFloat32KV(&buf, "llama.attention.layer_norm_rms_epsilon", 1e-5)
	if len(tokens) > 0 {
		writeGGUFStringArrayKV(&buf, "tokenizer.ggml.tokens", tokens)
	}

	for _, tensor := range tensors {
		writeGGUFString(&buf, tensor.name)
		writeLE(&buf, uint32(len(tensor.dims)))
		for _, d := range tensor.dims {
			writeLE(&buf, d)
		}
		writeLE(&buf, uint32(0)) // F32
		writeLE(&buf, tensor.offset)
	}

	if pad := (32 - (buf.Len() % 32)) % 32; pad > 0 {
		buf.Write(make([]byte, pad))
	}
	if dataBytes > uint64(1<<31-1) {
		return fmt.Errorf("fixture data too large: %d bytes", dataBytes)
	}
	buf.Write(make([]byte, int(dataBytes)))
	return os.WriteFile(path, buf.Bytes(), 0o600)
}

func extractDOEStatusElections(out string) (int, bool) {
	idx := strings.LastIndex(out, "elections=")
	if idx < 0 {
		return 0, false
	}
	var elections int
	if _, err := fmt.Sscanf(out[idx:], "elections=%d", &elections); err != nil {
		return 0, false
	}
	return elections, true
}

func doeTensorF32Bytes(dims []uint64) (uint64, error) {
	if len(dims) == 0 || len(dims) > 4 {
		return 0, fmt.Errorf("invalid ndim %d", len(dims))
	}
	n := uint64(1)
	for _, d := range dims {
		if d == 0 || n > ^uint64(0)/d {
			return 0, fmt.Errorf("invalid dimension product")
		}
		n *= d
	}
	if n > ^uint64(0)/4 {
		return 0, fmt.Errorf("byte size overflow")
	}
	return n * 4, nil
}

func writeGGUFStringKV(buf *bytes.Buffer, key, value string) {
	writeGGUFString(buf, key)
	writeLE(buf, uint32(8))
	writeGGUFString(buf, value)
}

func writeGGUFUint32KV(buf *bytes.Buffer, key string, value uint32) {
	writeGGUFString(buf, key)
	writeLE(buf, uint32(4))
	writeLE(buf, value)
}

func writeGGUFFloat32KV(buf *bytes.Buffer, key string, value float32) {
	writeGGUFString(buf, key)
	writeLE(buf, uint32(6))
	writeLE(buf, math.Float32bits(value))
}

func writeGGUFStringArrayKV(buf *bytes.Buffer, key string, values []string) {
	writeGGUFString(buf, key)
	writeLE(buf, uint32(9))
	writeLE(buf, uint32(8))
	writeLE(buf, uint64(len(values)))
	for _, value := range values {
		writeGGUFString(buf, value)
	}
}

func writeGGUFString(buf *bytes.Buffer, s string) {
	writeLE(buf, uint64(len(s)))
	buf.WriteString(s)
}

func writeLE(buf *bytes.Buffer, v any) {
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		panic(err)
	}
}
