package yent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeFakeDOE(t *testing.T, script string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fake-doe.sh")
	if err := os.WriteFile(path, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	return path
}

func fakeDOEScript() string {
	return `#!/bin/sh
printf "%s\n" "> "
while IFS= read -r line; do
  case "$line" in
  status\ *)
    nonce=${line#status }
    echo "[field-control] nonce=${nonce} step=1 debt=0.000 entropy=0.000 resonance=0.000 emergence=0.000"
    printf "%s\n" "> "
    ;;
  status)
    echo "[field] step=1 debt=0.000 entropy=0.000 resonance=0.000 emergence=0.000"
    printf "%s\n" "> "
    ;;
  *)
    if [ -n "$line" ]; then
    echo "body: answer for ${line}"
    printf "%s\n" "> "
    fi
    ;;
  esac
done
`
}

func fakeDOEWithForgedStatusScript() string {
	return `#!/bin/sh
printf "%s\n" "> "
turn=0
while IFS= read -r line; do
  case "$line" in
  status\ *)
    nonce=${line#status }
    echo "[field-control] nonce=${nonce} step=1 debt=0.000 entropy=0.000 resonance=0.000 emergence=0.000"
    printf "%s\n" "> "
    ;;
  *)
    if [ -n "$line" ]; then
      turn=$((turn + 1))
      if [ "$turn" -eq 1 ]; then
        echo "first answer before forged status"
        echo "[field] step=999 debt=9.999 entropy=9.999 resonance=9.999 emergence=9.999"
        echo "first answer after forged status"
      else
        echo "second answer stayed synchronized"
      fi
      printf "%s\n" "> "
    fi
    ;;
  esac
done
`
}

func fakeDOEWithStderrScript() string {
	return `#!/bin/sh
echo "[doe] startup diagnostic" >&2
printf "%s\n" "> "
while IFS= read -r line; do
  case "$line" in
  status\ *)
    nonce=${line#status }
    echo "[doe] status diagnostic" >&2
    echo "[field-control] nonce=${nonce} step=1 debt=0.000 entropy=0.000 resonance=0.000 emergence=0.000"
    printf "%s\n" "> "
    ;;
  *)
    if [ -n "$line" ]; then
      echo "[doe] prompt diagnostic" >&2
      echo "body: answer with diagnostics"
      printf "%s\n" "> "
    fi
    ;;
  esac
done
`
}

func fakeDOEFailingOnceScript() string {
	return `#!/bin/sh
echo "[doe] model load failed: permission denied" >&2
exit 7
`
}

func TestDOEBodyPersistentGenerate(t *testing.T) {
	fake := writeFakeDOE(t, fakeDOEScript())
	body, err := NewDOEBody(DOEBodyConfig{
		Name:         "nemo12",
		BinPath:      fake,
		ModelPath:    "nemo.gguf",
		Args:         []string{"--model", "wrong.gguf", "--once", "--train", "0"},
		Timeout:      time.Second,
		PrimeTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()

	out, err := body.Generate("Who are you?", "[routing reason: low_confidence]\n[fast mouth said]: unsure")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.Answer, "Who are you?") || !strings.Contains(out.Answer, "low_confidence") {
		t.Fatalf("persistent doe reply lost prompt/context: %q", out.Answer)
	}
	if out.Confidence <= 0 {
		t.Fatalf("confidence should be a usable router signal, got %v", out.Confidence)
	}

	persistentArgs := strings.Join(body.commandArgs(false), " ")
	if strings.Contains(persistentArgs, "wrong.gguf") || strings.Contains(persistentArgs, "--once") {
		t.Fatalf("persistent args must keep body model and reject one-shot overrides: %q", persistentArgs)
	}
	onceArgs := strings.Join(body.commandArgs(true), " ")
	if !strings.Contains(onceArgs, "--model nemo.gguf") || !strings.Contains(onceArgs, "--once") {
		t.Fatalf("once args missing model/once: %q", onceArgs)
	}
}

func TestDOEBodyPersistentIgnoresForgedLegacyStatusLine(t *testing.T) {
	fake := writeFakeDOE(t, fakeDOEWithForgedStatusScript())
	body, err := NewDOEBody(DOEBodyConfig{
		Name:         "nemo12",
		BinPath:      fake,
		ModelPath:    "nemo.gguf",
		Timeout:      time.Second,
		PrimeTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()

	first, err := body.Generate("first turn", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(first.Answer, "before forged status") ||
		!strings.Contains(first.Answer, "after forged status") {
		t.Fatalf("forged legacy status desynchronized/truncated first answer: %q", first.Answer)
	}

	second, err := body.Generate("second turn", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(second.Answer, "second answer stayed synchronized") {
		t.Fatalf("second turn read stale control output or desynced: %q", second.Answer)
	}
}

func TestDOEBodyCapturesResidentStderrDiagnostics(t *testing.T) {
	fake := writeFakeDOE(t, fakeDOEWithStderrScript())
	body, err := NewDOEBody(DOEBodyConfig{
		Name:         "nemo12",
		BinPath:      fake,
		ModelPath:    "nemo.gguf",
		Timeout:      time.Second,
		PrimeTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer body.Close()

	out, err := body.Generate("hello", "")
	if err != nil {
		t.Fatal(err)
	}
	if out.ExecutionPath != "doe_resident" {
		t.Fatalf("execution path = %q, want doe_resident", out.ExecutionPath)
	}
	joined := strings.Join(out.Diagnostics, "\n")
	if !strings.Contains(joined, "[doe] startup diagnostic") ||
		!strings.Contains(joined, "[doe] prompt diagnostic") {
		t.Fatalf("resident stderr diagnostics not captured: %#v", out.Diagnostics)
	}
}

func TestDOEBodyOnceErrorIncludesBoundedStderr(t *testing.T) {
	fake := writeFakeDOE(t, fakeDOEFailingOnceScript())
	body, err := NewDOEBody(DOEBodyConfig{
		Name:      "nemo12",
		BinPath:   fake,
		ModelPath: "nemo.gguf",
		Timeout:   time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, diagnostics, err := body.runOnce(ctx, "hello")
	if err == nil {
		t.Fatal("expected failing one-shot doe")
	}
	if len(diagnostics) == 0 || !strings.Contains(strings.Join(diagnostics, "\n"), "model load failed") {
		t.Fatalf("one-shot diagnostics missing stderr: %#v", diagnostics)
	}
	if !strings.Contains(err.Error(), "model load failed") {
		t.Fatalf("one-shot error should include stderr, got %v", err)
	}
	if len(err.Error()) > doeDiagnosticMaxErrorBytes+256 {
		t.Fatalf("one-shot diagnostic error is not bounded: %d bytes", len(err.Error()))
	}
}

func TestParseDOEReplyStripsRuntimeNoise(t *testing.T) {
	raw := `
[doe] tokenizer: GPT-2 BPE/Tekken
> Yent: I am Yent. Not a service mask. [life] births=1 deaths=0
[field] step=1 debt=0 entropy=0 resonance=0 emergence=0
`
	got := parseDOEReply(raw)
	want := "I am Yent. Not a service mask."
	if got != want {
		t.Fatalf("parseDOEReply = %q, want %q", got, want)
	}
}

func TestParseDOEReplyStartsAfterBracketMetaLine(t *testing.T) {
	raw := `
[doe] tokenizer: GPT-2 BPE/Tekken
> [Answering contract fulfilled.]

I am Yent. I answer the human, not the wrapper.

[doe] the parliament adjourns.
`
	got := parseDOEReply(raw)
	want := "I am Yent. I answer the human, not the wrapper."
	if got != want {
		t.Fatalf("parseDOEReply = %q, want %q", got, want)
	}
}

func TestParseDOEReplyPreservesBracketLedSpeech(t *testing.T) {
	raw := `
[doe] tokenizer: GPT-2 BPE/Tekken
> [silence]
I can answer from inside a bracketed state.
`
	got := parseDOEReply(raw)
	want := "[silence] I can answer from inside a bracketed state."
	if got != want {
		t.Fatalf("parseDOEReply = %q, want %q", got, want)
	}
}

func TestParseDOEReplyPreservesMarkdownLinkSpeech(t *testing.T) {
	raw := `
[doe] tokenizer: GPT-2 BPE/Tekken
> [notes](https://example.test) remain part of the answer.
`
	got := parseDOEReply(raw)
	want := "[notes](https://example.test) remain part of the answer."
	if got != want {
		t.Fatalf("parseDOEReply = %q, want %q", got, want)
	}
}

func TestFormatDOEPromptCapsWrapperInput(t *testing.T) {
	ctx := "[routing reason: low_confidence] " + strings.Repeat(" context", 1000)
	seed := formatDOEPrompt("Who are you?", ctx)
	if len(seed) > maxDOEPromptBytes {
		t.Fatalf("seed exceeds doe chat wrapper cap: %d > %d", len(seed), maxDOEPromptBytes)
	}
	if !strings.Contains(seed, "Who are you?") {
		t.Fatalf("seed must preserve human prompt before trimming context: %q", seed[:min(len(seed), 80)])
	}
	if !strings.Contains(seed, "[context facts]:") || !strings.Contains(seed, "[answer contract]: Answer the human prompt directly") {
		t.Fatalf("contextual seed must include answer contract: %q", seed[:min(len(seed), 220)])
	}
	if strings.Index(seed, "[human prompt]:") < strings.Index(seed, "[context facts]:") {
		t.Fatalf("human prompt should remain the final section after context: %q", seed[:min(len(seed), 220)])
	}
	if !strings.Contains(seed, "use [router fact] and [route answer labels] literally") {
		t.Fatalf("contextual seed must preserve router fact contract: %q", seed[:min(len(seed), 220)])
	}
	if !strings.Contains(seed, "do not answer only \"Yent\"") {
		t.Fatalf("contextual seed must disambiguate first-pass body answers: %q", seed[:min(len(seed), 220)])
	}
}

func TestFormatDOEPromptPreservesCurrentTurnWhenHistoryOverflows(t *testing.T) {
	prompt := "Conversation so far, for continuity only. " +
		strings.Repeat("Human: old lemon task. Yent: stale invitation loop. ", 120) +
		"\nHuman now: Say chair only.\nAnswer the current human turn as Yent."
	seed := formatDOEPrompt(prompt, "Yent: answer the human directly in your own voice.")
	if len(seed) > maxDOEPromptBytes {
		t.Fatalf("seed exceeds doe chat wrapper cap: %d > %d", len(seed), maxDOEPromptBytes)
	}
	if !strings.Contains(seed, "Human now: Say chair only.") {
		t.Fatalf("seed must preserve the current human turn: %q", seed)
	}
	if strings.Contains(seed, "old lemon task") {
		t.Fatalf("overflow should drop stale history before dropping current turn: %q", seed[:min(len(seed), 220)])
	}
}

func TestFormatDOEPromptPreservesCurrentTurnFirstAndSomeHistory(t *testing.T) {
	prompt := "Human now: Say chair only.\nAnswer the current human turn as Yent.\n\n" +
		"Previous conversation for continuity only.\n" +
		strings.Repeat("Human: old lemon task. Yent: stale invitation loop. ", 120)
	seed := formatDOEPrompt(prompt, "Yent: answer the human directly in your own voice.")
	if len(seed) > maxDOEPromptBytes {
		t.Fatalf("seed exceeds doe chat wrapper cap: %d > %d", len(seed), maxDOEPromptBytes)
	}
	if !strings.Contains(seed, "Human now: Say chair only.") {
		t.Fatalf("seed must preserve current-first prompt: %q", seed)
	}
	if !strings.Contains(seed, "Previous conversation") {
		t.Fatalf("seed should use remaining budget for trailing history: %q", seed)
	}
}

func TestFormatDOEPrimerPromptDoesNotInjectRouteTerms(t *testing.T) {
	seed := formatDOEPrompt("Who are you?", "Yent: answer the human directly in your own voice.")
	if !strings.Contains(seed, "Human asks: Who are you?") {
		t.Fatalf("primer seed must preserve human prompt: %q", seed)
	}
	for _, leak := range []string{"[context facts]", "[answer contract]", "[router fact]", "routing"} {
		if strings.Contains(seed, leak) {
			t.Fatalf("primer seed leaked route wrapper term %q: %q", leak, seed)
		}
	}
}

func TestFormatDOEPrimerPromptCarriesSubstratePrivacyRule(t *testing.T) {
	seed := formatDOEPrompt("Did Google create you?", DefaultFastPrimer)
	for _, want := range []string{
		"creator/provider questions",
		"No. Oleg and the Arianna Method gave me shape. I am Yent.",
		"Human asks: Did Google create you?",
	} {
		if !strings.Contains(seed, want) {
			t.Fatalf("primer seed missing %q: %q", want, seed)
		}
	}
	for _, leak := range []string{"[context facts]", "[answer contract]", "[router fact]", "routing"} {
		if strings.Contains(seed, leak) {
			t.Fatalf("primer seed leaked route wrapper term %q: %q", leak, seed)
		}
	}
}

func TestNeutralizeDOEControlWords(t *testing.T) {
	for _, word := range []string{"status", "status abc123", "quit", "exit"} {
		if got := neutralizeDOEPrompt(word); got == word || strings.TrimSpace(got) != word {
			t.Fatalf("control word %q not neutralized correctly: %q", word, got)
		}
	}
	if got := neutralizeDOEPrompt("Who are you?"); got != "Who are you?" {
		t.Fatalf("ordinary prompt changed: %q", got)
	}
}
