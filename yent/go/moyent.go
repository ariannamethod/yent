package yent

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	envDOEBin         = "YENT_DOE_BIN"
	envNemoGGUF       = "YENT_NEMO_GGUF"
	envDeepGGUF       = "YENT_24B_GGUF"
	envDeepGGUFAlt    = "YENT_DEEP_GGUF"
	envDOEWorkDir     = "YENT_DOE_WORKDIR"
	envDOEArgs        = "YENT_DOE_ARGS"
	envNemoArgs       = "YENT_NEMO_ARGS"
	envDeepArgs       = "YENT_24B_ARGS"
	envDOETimeout     = "YENT_DOE_TIMEOUT_SEC"
	envDOEPrime       = "YENT_DOE_PRIME_TIMEOUT_SEC"
	envEscalateBelow  = "YENT_ESCALATE_BELOW"
	envFastPrimer     = "YENT_FAST_PRIMER"
	envDeepPrimer     = "YENT_DEEP_PRIMER"
	envFastPrimerFile = "YENT_FAST_PRIMER_FILE"
	envDeepPrimerFile = "YENT_DEEP_PRIMER_FILE"
	envMemoryRefs     = "YENT_MEMORY_REFS"
	envStateRefs      = "YENT_STATE_REFS"
	envAsyncMemory    = "YENT_ASYNC_MEMORY"
	envSingleBody     = "YENT_SINGLE_RESIDENT"
)

const (
	defaultFastPrimerFile = "prompts/nemo12_fast_v1.txt"
	defaultDeepPrimerFile = "prompts/small24_deep_v1.txt"
)

// NewDOERouter wires two real doe-backed bodies into one shared limpha brain.
// It does not validate model quality; it only constructs the process plumbing.
func NewDOERouter(fastCfg, deepCfg DOEBodyConfig, limpha *LimphaClient) (*Router, func() error, error) {
	fast, err := NewDOEBody(fastCfg)
	if err != nil {
		return nil, nil, fmt.Errorf("fast body: %w", err)
	}
	deep, err := NewDOEBody(deepCfg)
	if err != nil {
		_ = fast.Close()
		return nil, nil, fmt.Errorf("deep body: %w", err)
	}
	cleanup := func() error {
		return errors.Join(fast.Close(), deep.Close())
	}
	router := NewRouter(fast, deep, limpha)
	router.AsyncMemory = limpha != nil && limpha.asyncEnabled()
	router.SingleResident = true
	return router, cleanup, nil
}

// NewMoyentRouterFromEnv builds the real two-body router from environment paths.
// Required:
//
//	YENT_DOE_BIN   path to doe_field
//	YENT_NEMO_GGUF fast-body GGUF
//	YENT_24B_GGUF  deep-body GGUF (or YENT_DEEP_GGUF)
//
// Optional:
//
//	YENT_DOE_ARGS, YENT_NEMO_ARGS, YENT_24B_ARGS are whitespace-split args
//	appended after --model <path>. Use simple flags here, not shell quoting.
func NewMoyentRouterFromEnv(limpha *LimphaClient) (*Router, func() error, error) {
	bin := strings.TrimSpace(os.Getenv(envDOEBin))
	fastModel := strings.TrimSpace(os.Getenv(envNemoGGUF))
	deepModel := firstEnv(envDeepGGUF, envDeepGGUFAlt)
	var missing []string
	if bin == "" {
		missing = append(missing, envDOEBin)
	}
	if fastModel == "" {
		missing = append(missing, envNemoGGUF)
	}
	if deepModel == "" {
		missing = append(missing, envDeepGGUF)
	}
	if len(missing) > 0 {
		return nil, nil, fmt.Errorf("moyent env missing: %s", strings.Join(missing, ", "))
	}

	commonArgs := splitEnvArgs(os.Getenv(envDOEArgs))
	workDir := strings.TrimSpace(os.Getenv(envDOEWorkDir))
	threshold, hasThreshold, err := floatEnv(envEscalateBelow)
	if err != nil {
		return nil, nil, err
	}
	timeout, err := durationEnvSeconds(envDOETimeout)
	if err != nil {
		return nil, nil, err
	}
	primeTimeout, err := durationEnvSeconds(envDOEPrime)
	if err != nil {
		return nil, nil, err
	}
	fastCfg := DOEBodyConfig{
		Name:         "nemo12",
		BinPath:      bin,
		ModelPath:    fastModel,
		WorkDir:      workDir,
		Args:         appendArgs(commonArgs, splitEnvArgs(os.Getenv(envNemoArgs))),
		Timeout:      timeout,
		PrimeTimeout: primeTimeout,
	}
	deepCfg := DOEBodyConfig{
		Name:         "small24",
		BinPath:      bin,
		ModelPath:    deepModel,
		WorkDir:      workDir,
		Args:         appendArgs(commonArgs, splitEnvArgs(os.Getenv(envDeepArgs))),
		Timeout:      timeout,
		PrimeTimeout: primeTimeout,
	}
	if limpha != nil && boolEnv(envAsyncMemory, true) {
		limpha.StartAsync(256)
	}
	router, cleanup, err := NewDOERouter(fastCfg, deepCfg, limpha)
	if err != nil {
		if limpha != nil {
			limpha.StopAsync()
		}
		return nil, nil, err
	}
	if hasThreshold {
		router.EscalateBelow = threshold
	}
	if primer, err := resolvePrimer(envFastPrimer, envFastPrimerFile, defaultFastPrimerFile, router.FastPrimer); err != nil {
		if limpha != nil {
			limpha.StopAsync()
		}
		_ = cleanup()
		return nil, nil, err
	} else {
		router.FastPrimer = primer
	}
	if primer, err := resolvePrimer(envDeepPrimer, envDeepPrimerFile, defaultDeepPrimerFile, router.DeepPrimer); err != nil {
		if limpha != nil {
			limpha.StopAsync()
		}
		_ = cleanup()
		return nil, nil, err
	} else {
		router.DeepPrimer = primer
	}
	if refs, ok, err := intEnv(envMemoryRefs); err != nil {
		if limpha != nil {
			limpha.StopAsync()
		}
		_ = cleanup()
		return nil, nil, err
	} else if ok {
		router.MemoryRefs = refs
	}
	if refs, ok, err := intEnv(envStateRefs); err != nil {
		if limpha != nil {
			limpha.StopAsync()
		}
		_ = cleanup()
		return nil, nil, err
	} else if ok {
		router.StateRefs = refs
	}
	router.AsyncMemory = limpha != nil && boolEnv(envAsyncMemory, true)
	router.SingleResident = boolEnv(envSingleBody, true)
	return router, cleanup, nil
}

func splitEnvArgs(raw string) []string {
	return strings.Fields(raw)
}

func resolvePrimer(valueEnv, fileEnv, defaultFile, fallback string) (string, error) {
	if primer := normalizePrimer(os.Getenv(valueEnv)); primer != "" {
		return primer, nil
	}
	path := strings.TrimSpace(os.Getenv(fileEnv))
	explicit := path != ""
	if path == "" {
		path = defaultFile
	}
	if path != "" {
		data, err := os.ReadFile(path)
		if err == nil {
			if primer := normalizePrimer(string(data)); primer != "" {
				return primer, nil
			}
			if explicit {
				return "", fmt.Errorf("%s points to an empty primer file: %s", fileEnv, path)
			}
		} else if explicit || !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("%s: read %s: %w", fileEnv, path, err)
		}
	}
	return normalizePrimer(fallback), nil
}

func normalizePrimer(raw string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(raw)), " ")
}

func appendArgs(common, extra []string) []string {
	out := make([]string, 0, len(common)+len(extra))
	out = append(out, common...)
	out = append(out, extra...)
	return out
}

func firstEnv(names ...string) string {
	for _, name := range names {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	return ""
}

func boolEnv(name string, def bool) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "":
		return def
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}

func floatEnv(name string) (float64, bool, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, false, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, true, fmt.Errorf("%s: %w", name, err)
	}
	return clamp01(v), true, nil
}

func intEnv(name string) (int, bool, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, false, nil
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, true, fmt.Errorf("%s: %w", name, err)
	}
	if v <= 0 {
		return 0, true, fmt.Errorf("%s must be positive", name)
	}
	return v, true, nil
}

func durationEnvSeconds(name string) (time.Duration, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", name, err)
	}
	if v <= 0 {
		return 0, fmt.Errorf("%s must be positive seconds", name)
	}
	return time.Duration(v * float64(time.Second)), nil
}
