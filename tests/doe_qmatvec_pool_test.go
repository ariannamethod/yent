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

func TestDOEQMatvecPoolMatchesSingleThreadQ80(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("doe.c qmatvec harness is POSIX-only")
	}
	if _, err := exec.LookPath("cc"); err != nil {
		t.Skipf("cc not found: %v", err)
	}
	root := repoRootForTest(t)
	doePath := strings.ReplaceAll(filepath.Join(root, "DoE", "doe.c"), `\`, `\\`)
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "qmatvec_pool_harness.c")
	exe := filepath.Join(dir, "qmatvec_pool_harness")
	src := fmt.Sprintf(`#include <stdint.h>
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

static void put_f16_one(uint8_t *p) {
    p[0] = 0x00;
    p[1] = 0x3c;
}

int main(void) {
    const int rows = 1300;
    const int cols = 1024;
    const int nb = cols / 32;
    uint8_t *w = (uint8_t *)calloc((size_t)rows * nb * 34, 1);
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *ref = (float *)calloc((size_t)rows, sizeof(float));
    float *got = (float *)calloc((size_t)rows, sizeof(float));
    if (!w || !x || !ref || !got) return 2;

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            uint8_t *blk = w + ((size_t)r * nb + b) * 34;
            put_f16_one(blk);
            for (int i = 0; i < 32; i++) {
                int q = (r * 3 + b * 5 + i * 7) %% 31;
                blk[2 + i] = (uint8_t)(int8_t)(q - 15);
            }
        }
    }
    for (int i = 0; i < cols; i++) x[i] = (float)((i %% 23) - 11) * 0.03125f;

    g_n_threads = 1;
    if (doe_qmatvec(ref, w, 8, x, rows, cols) != 0) return 3;

    g_n_threads = 8;
    if (doe_qmatvec(got, w, 8, x, rows, cols) != 0) return 4;

    for (int r = 0; r < rows; r++) {
        if (memcmp(&ref[r], &got[r], sizeof(float)) != 0) {
            fprintf(stderr, "row %%d mismatch: ref=%%.9g got=%%.9g\n", r, ref[r], got[r]);
            return 5;
        }
    }
    free(got); free(ref); free(x); free(w);
    return 0;
}
`, doePath)
	if err := os.WriteFile(srcPath, []byte(src), 0o600); err != nil {
		t.Fatalf("write harness: %v", err)
	}
	cmd := exec.Command("cc", "-O2", "-Wall", "-Wextra", srcPath, "-lm", "-lpthread", "-o", exe)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile qmatvec pool harness: %v\n%s", err, string(out))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	run := exec.CommandContext(ctx, exe)
	out, err = run.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("qmatvec pool harness timed out:\n%s", string(out))
	}
	if err != nil {
		t.Fatalf("qmatvec pool harness failed: %v\n%s", err, string(out))
	}
}
