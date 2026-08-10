package tests

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestDOEQMatvecInt8FastPathContracts(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("doe.c qmatvec harness is POSIX-only")
	}
	if _, err := exec.LookPath("cc"); err != nil {
		t.Skipf("cc not found: %v", err)
	}
	root := repoRootForTest(t)
	doePath := strings.ReplaceAll(filepath.Join(root, "DoE", "doe.c"), `\`, `\\`)
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "qmatvec_i8_harness.c")
	exe := filepath.Join(dir, "qmatvec_i8_harness")
	src := fmt.Sprintf(`#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float *pv_encode_image(const char *img_path, const char *mmproj_path, int *o_ntok, int *o_dim) {
    (void)img_path; (void)mmproj_path; (void)o_ntok; (void)o_dim;
    return NULL;
}

#define main doe_embedded_main
#include "%s"
#undef main

static void put_f16(uint8_t *p, uint16_t h) {
    p[0] = (uint8_t)(h & 0xffu);
    p[1] = (uint8_t)(h >> 8);
}

static void fill_x(float *x, int cols, int salt) {
    for (int i = 0; i < cols; i++) {
        int v = ((i * 17 + salt * 13) %% 41) - 20;
        x[i] = (float)v * 0.03125f;
    }
}

static size_t block_bytes(int dt) {
    if (dt == 2) return 18;
    if (dt == 6) return 22;
    if (dt == 8) return 34;
    if (dt == 12) return 144;
    if (dt == 14) return 210;
    return 0;
}

static int blocks_for(int dt, int cols) {
    return (dt == 12 || dt == 14) ? cols / 256 : cols / 32;
}

static void fill_q4_0(uint8_t *w, int rows, int cols) {
    int nb = cols / 32;
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) {
        uint8_t *blk = w + ((size_t)r * nb + b) * 18;
        put_f16(blk, 0x3c00);
        for (int i = 0; i < 16; i++) {
            uint8_t lo = (uint8_t)((r + b * 3 + i * 5) & 15);
            uint8_t hi = (uint8_t)((r * 7 + b + i * 11) & 15);
            blk[2 + i] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static void fill_q5_0(uint8_t *w, int rows, int cols) {
    int nb = cols / 32;
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) {
        uint8_t *blk = w + ((size_t)r * nb + b) * 22;
        uint32_t qh = 0;
        put_f16(blk, 0x3c00);
        for (int i = 0; i < 16; i++) {
            uint8_t q0 = (uint8_t)((r * 5 + b * 3 + i * 7) & 31);
            uint8_t q1 = (uint8_t)((r * 11 + b * 13 + i * 17) & 31);
            if (q0 & 16) qh |= (uint32_t)1u << i;
            if (q1 & 16) qh |= (uint32_t)1u << (i + 16);
            blk[6 + i] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
        }
        blk[2] = (uint8_t)(qh & 0xffu);
        blk[3] = (uint8_t)((qh >> 8) & 0xffu);
        blk[4] = (uint8_t)((qh >> 16) & 0xffu);
        blk[5] = (uint8_t)((qh >> 24) & 0xffu);
    }
}

static void fill_q8_0(uint8_t *w, int rows, int cols) {
    int nb = cols / 32;
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) {
        uint8_t *blk = w + ((size_t)r * nb + b) * 34;
        put_f16(blk, 0x3c00);
        for (int i = 0; i < 32; i++) {
            int q = ((r * 3 + b * 5 + i * 7) %% 31) - 15;
            blk[2 + i] = (uint8_t)(int8_t)q;
        }
    }
}

static void fill_q4_k_scales(uint8_t *sc) {
    memset(sc, 0, 12);
    for (int j = 0; j < 4; j++) sc[j] = 1;
    for (int j = 4; j < 8; j++) sc[j + 4] = (uint8_t)((sc[j + 4] & 0xf0u) | 1u);
}

static void fill_q4_k(uint8_t *w, int rows, int cols) {
    int nb = cols / 256;
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) {
        uint8_t *blk = w + ((size_t)r * nb + b) * 144;
        put_f16(blk, 0x3c00);
        put_f16(blk + 2, 0x0000);
        fill_q4_k_scales(blk + 4);
        for (int i = 0; i < 128; i++) {
            uint8_t lo = (uint8_t)((r + b + i) & 15);
            uint8_t hi = (uint8_t)((r * 5 + b * 3 + i * 7) & 15);
            blk[16 + i] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static void fill_q6_k(uint8_t *w, int rows, int cols) {
    int nb = cols / 256;
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) {
        uint8_t *blk = w + ((size_t)r * nb + b) * 210;
        for (int i = 0; i < 128; i++) blk[i] = (uint8_t)(r * 11 + b * 17 + i * 3);
        for (int i = 0; i < 64; i++) blk[128 + i] = (uint8_t)(r * 5 + b * 7 + i * 13);
        for (int i = 0; i < 16; i++) blk[192 + i] = (uint8_t)(int8_t)((i %% 5) - 2);
        put_f16(blk + 208, 0x3c00);
    }
}

static void fill_weight(uint8_t *w, int dt, int rows, int cols) {
    if (dt == 2) fill_q4_0(w, rows, cols);
    else if (dt == 6) fill_q5_0(w, rows, cols);
    else if (dt == 8) fill_q8_0(w, rows, cols);
    else if (dt == 12) fill_q4_k(w, rows, cols);
    else if (dt == 14) fill_q6_k(w, rows, cols);
}

static int outputs_same(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) if (memcmp(a + i, b + i, sizeof(float)) != 0) return 0;
    return 1;
}

static int check_threaded_i8(int dt) {
    const int rows = 1300;
    const int cols = 1024;
    int nb = blocks_for(dt, cols);
    size_t bytes = (size_t)rows * (size_t)nb * block_bytes(dt);
    uint8_t *w = (uint8_t *)calloc(bytes, 1);
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *ref = (float *)calloc((size_t)rows, sizeof(float));
    float *got = (float *)calloc((size_t)rows, sizeof(float));
    if (!w || !x || !ref || !got) return 10 + dt;
    fill_weight(w, dt, rows, cols);
    fill_x(x, cols, dt);

    g_n_threads = 1;
    if (doe_qmatvec_i8(ref, w, dt, x, rows, cols) != 0) return 20 + dt;
    g_n_threads = 8;
    if (doe_qmatvec_i8(got, w, dt, x, rows, cols) != 0) return 30 + dt;
    if (!outputs_same(ref, got, rows)) {
        fprintf(stderr, "dt %%d single/threaded mismatch\n", dt);
        return 40 + dt;
    }

    free(got); free(ref); free(x); free(w);
    return 0;
}

static int check_cache_not_stale(void) {
    const int rows = 9;
    const int cols = 64;
    int nb = cols / 32;
    uint8_t *w = (uint8_t *)calloc((size_t)rows * nb * 34, 1);
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float out1[9], out2[9];
    if (!w || !x) return 90;
    fill_q8_0(w, rows, cols);
    fill_x(x, cols, 8);
    g_n_threads = 1;
    if (doe_qmatvec_i8(out1, w, 8, x, rows, cols) != 0) return 91;
    for (int i = 0; i < cols; i += 7) x[i] += 3.0f;
    if (doe_qmatvec_i8(out2, w, 8, x, rows, cols) != 0) return 92;
    if (outputs_same(out1, out2, rows)) {
        fprintf(stderr, "activation cache reused stale same-pointer contents\n");
        return 93;
    }
    free(x); free(w);
    return 0;
}

int main(void) {
    int dts[] = {2, 6, 8, 12, 14};
    for (int i = 0; i < 5; i++) {
        int rc = check_threaded_i8(dts[i]);
        if (rc != 0) return rc;
    }
    if (doe_qmatvec_i8(NULL, NULL, 1, NULL, 1, 32) != -1) return 80;
    if (doe_qmatvec_i8(NULL, NULL, 6, NULL, 1, 33) != -1) return 81;
    return check_cache_not_stale();
}
`, doePath)
	if err := os.WriteFile(srcPath, []byte(src), 0o600); err != nil {
		t.Fatalf("write harness: %v", err)
	}
	cmd := exec.Command("cc", "-O2", "-Wall", "-Wextra", srcPath, "-lm", "-lpthread", "-o", exe)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile qmatvec i8 harness: %v\n%s", err, string(out))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	run := exec.CommandContext(ctx, exe)
	out, err = run.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("qmatvec i8 harness timed out:\n%s", string(out))
	}
	if err != nil {
		t.Fatalf("qmatvec i8 harness failed: %v\n%s", err, string(out))
	}
}
