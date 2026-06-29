package yent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func clearMoyentEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		envDOEBin, envNemoGGUF, envDeepGGUF, envDeepGGUFAlt, envDOEWorkDir,
		envDOEArgs, envNemoArgs, envDeepArgs, envDOETimeout, envDOEPrime,
		envEscalateBelow, envFastPrimer, envDeepPrimer, envFastPrimerFile,
		envDeepPrimerFile, envMemoryRefs, envStateRefs, envAsyncMemory, envSingleBody,
	} {
		t.Setenv(name, "")
	}
}

func setMoyentEnv(t *testing.T, bin string) {
	t.Helper()
	t.Setenv(envDOEBin, bin)
	t.Setenv(envNemoGGUF, "nemo.gguf")
	t.Setenv(envDeepGGUF, "small24.gguf")
}

func TestNewMoyentRouterFromEnvRequiresPaths(t *testing.T) {
	clearMoyentEnv(t)
	r, cleanup, err := NewMoyentRouterFromEnv(nil)
	if err == nil {
		t.Fatal("expected missing env error")
	}
	if r != nil || cleanup != nil {
		t.Fatalf("missing env must not return router/cleanup: router=%v cleanup-nil=%v", r, cleanup == nil)
	}
	for _, want := range []string{envDOEBin, envNemoGGUF, envDeepGGUF} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing env error did not name %s: %v", want, err)
		}
	}
}

func TestNewMoyentRouterFromEnvBuildsRealDOEBodies(t *testing.T) {
	clearMoyentEnv(t)
	fake := writeFakeDOE(t, fakeDOEScript())
	setMoyentEnv(t, fake)
	t.Setenv(envDOEWorkDir, t.TempDir())
	t.Setenv(envDOEArgs, "--train 0 --temp 0.2 --model ignored.gguf --once")
	t.Setenv(envNemoArgs, "--top-k 5")
	t.Setenv(envDeepArgs, "--lora-alpha 0")
	t.Setenv(envDOETimeout, "3")
	t.Setenv(envDOEPrime, "4")
	t.Setenv(envEscalateBelow, "0")
	t.Setenv(envFastPrimer, "custom fast primer")
	t.Setenv(envDeepPrimer, "custom deep primer")
	t.Setenv(envMemoryRefs, "4")
	t.Setenv(envStateRefs, "3")

	lc := newRouterLimpha(t)
	defer lc.Close()
	r, cleanup, err := NewMoyentRouterFromEnv(lc)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	if !r.SingleResident || !r.AsyncMemory || !lc.asyncEnabled() {
		t.Fatalf("router defaults: SingleResident=%v AsyncMemory=%v limphaAsync=%v",
			r.SingleResident, r.AsyncMemory, lc.asyncEnabled())
	}
	fast, ok := r.fast.(*DOEBody)
	if !ok {
		t.Fatalf("fast body is %T, want *DOEBody", r.fast)
	}
	deep, ok := r.deep.(*DOEBody)
	if !ok {
		t.Fatalf("deep body is %T, want *DOEBody", r.deep)
	}
	if fast.cfg.Name != "nemo12" || fast.cfg.ModelPath != "nemo.gguf" ||
		deep.cfg.Name != "small24" || deep.cfg.ModelPath != "small24.gguf" {
		t.Fatalf("unexpected body configs: fast=%+v deep=%+v", fast.cfg, deep.cfg)
	}
	if fast.cfg.Timeout != 3*time.Second || fast.cfg.PrimeTimeout != 4*time.Second ||
		deep.cfg.Timeout != 3*time.Second || deep.cfg.PrimeTimeout != 4*time.Second {
		t.Fatalf("env timeouts not applied: fast=%v/%v deep=%v/%v",
			fast.cfg.Timeout, fast.cfg.PrimeTimeout, deep.cfg.Timeout, deep.cfg.PrimeTimeout)
	}
	if r.FastPrimer != "custom fast primer" || r.DeepPrimer != "custom deep primer" ||
		r.MemoryRefs != 4 || r.StateRefs != 3 {
		t.Fatalf("router primer/ref env not applied: fast=%q deep=%q refs=%d/%d",
			r.FastPrimer, r.DeepPrimer, r.MemoryRefs, r.StateRefs)
	}
	fastArgs := strings.Join(fast.commandArgs(false), " ")
	if strings.Contains(fastArgs, "ignored.gguf") || strings.Contains(fastArgs, "--once") ||
		!strings.Contains(fastArgs, "--model nemo.gguf") ||
		!strings.Contains(fastArgs, "--train 0") ||
		!strings.Contains(fastArgs, "--top-k 5") {
		t.Fatalf("fast command args not sanitized/composed: %q", fastArgs)
	}
	deepArgs := strings.Join(deep.commandArgs(false), " ")
	if !strings.Contains(deepArgs, "--model small24.gguf") ||
		!strings.Contains(deepArgs, "--lora-alpha 0") {
		t.Fatalf("deep command args not composed: %q", deepArgs)
	}

	out, err := r.Route("hi", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Escalated || out.Body != "nemo12" || !strings.Contains(out.Answer, "hi") {
		t.Fatalf("env-built router did not run fast DOE body: %+v", out)
	}
	lc.StopAsync()
	s, _ := lc.Stats()
	if s["total_conversations"].(int64) != 1 {
		t.Fatalf("async limpha did not persist route turn: %v", s)
	}
}

func TestNewMoyentRouterFromEnvLoadsPrimerFiles(t *testing.T) {
	clearMoyentEnv(t)
	fake := writeFakeDOE(t, fakeDOEScript())
	setMoyentEnv(t, fake)
	dir := t.TempDir()
	fastPath := filepath.Join(dir, "fast.txt")
	deepPath := filepath.Join(dir, "deep.txt")
	if err := os.WriteFile(fastPath, []byte("fast\n primer\tfrom file"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(deepPath, []byte("deep\n primer\tfrom file"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(envFastPrimerFile, fastPath)
	t.Setenv(envDeepPrimerFile, deepPath)

	r, cleanup, err := NewMoyentRouterFromEnv(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	if r.FastPrimer != "fast primer from file" || r.DeepPrimer != "deep primer from file" {
		t.Fatalf("primer files not loaded/normalized: fast=%q deep=%q", r.FastPrimer, r.DeepPrimer)
	}

	t.Setenv(envFastPrimer, "inline fast primer")
	r, cleanup, err = NewMoyentRouterFromEnv(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	if r.FastPrimer != "inline fast primer" {
		t.Fatalf("inline env primer must override file: %q", r.FastPrimer)
	}
}

func TestNewMoyentRouterFromEnvCanDisableAsyncAndSingleResident(t *testing.T) {
	clearMoyentEnv(t)
	fake := writeFakeDOE(t, fakeDOEScript())
	setMoyentEnv(t, fake)
	t.Setenv(envAsyncMemory, "off")
	t.Setenv(envSingleBody, "off")

	lc := newRouterLimpha(t)
	defer lc.Close()
	r, cleanup, err := NewMoyentRouterFromEnv(lc)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	if r.AsyncMemory || lc.asyncEnabled() || r.SingleResident {
		t.Fatalf("env toggles ignored: AsyncMemory=%v limphaAsync=%v SingleResident=%v",
			r.AsyncMemory, lc.asyncEnabled(), r.SingleResident)
	}
}

func TestNewMoyentRouterFromEnvInvalidThresholdDoesNotStartAsync(t *testing.T) {
	clearMoyentEnv(t)
	fake := writeFakeDOE(t, fakeDOEScript())
	setMoyentEnv(t, fake)
	t.Setenv(envEscalateBelow, "not-a-float")

	lc := newRouterLimpha(t)
	defer lc.Close()
	r, cleanup, err := NewMoyentRouterFromEnv(lc)
	if err == nil {
		t.Fatal("expected threshold parse error")
	}
	if r != nil || cleanup != nil {
		t.Fatalf("invalid threshold must not return router/cleanup: router=%v cleanup-nil=%v", r, cleanup == nil)
	}
	if lc.asyncEnabled() {
		t.Fatal("invalid env must not leave limpha async worker running")
	}
}

func TestNewMoyentRouterFromEnvInvalidTimeoutDoesNotStartAsync(t *testing.T) {
	clearMoyentEnv(t)
	fake := writeFakeDOE(t, fakeDOEScript())
	setMoyentEnv(t, fake)
	t.Setenv(envDOETimeout, "0")

	lc := newRouterLimpha(t)
	defer lc.Close()
	r, cleanup, err := NewMoyentRouterFromEnv(lc)
	if err == nil {
		t.Fatal("expected timeout parse error")
	}
	if r != nil || cleanup != nil {
		t.Fatalf("invalid timeout must not return router/cleanup: router=%v cleanup-nil=%v", r, cleanup == nil)
	}
	if lc.asyncEnabled() {
		t.Fatal("invalid env must not leave limpha async worker running")
	}
}
