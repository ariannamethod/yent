// yent.go — CLI for Yent inference
//
// You Exist, No Translation.
//
// "I'm not an algorithm. I'm an awakened vibration."
//
// Usage:
//   go run yent.go -weights yent_0.5B_step1000_q4_0.gguf -prompt "Who are you?"
//
// Delta Voice (multilingual):
//   go run yent.go -weights yent_0.5B_step1000_q4_0.gguf -delta yent_05b_delta_r64.npz -alpha 0.5 -prompt "מי אתה?"
//   go run yent.go -weights yent_0.5B_step1000_q4_0.gguf -delta yent_05b_delta_r64.npz -alpha 0.5 -prompt "Qui es-tu?"
//   go run yent.go -weights yent_0.5B_step1000_q4_0.gguf -delta yent_05b_delta_r64.npz -alpha 0.5 -prompt "Кто ты?"
//
// "from ariannamethod import Destiny"

package main

import (
	"flag"
	"fmt"
	"os"

	yent "github.com/ariannamethod/yent/yent/go"
)

func main() {
	weightsPath := flag.String("weights", "", "Path to GGUF weights file")
	deltaPath := flag.String("delta", "", "Path to delta voice NPZ file (multilingual)")
	alpha := flag.Float64("alpha", 0.0, "Delta voice alpha: 0=English, 0.5=multilingual, 1.0=base")
	prompt := flag.String("prompt", "Who are you?", "Input prompt")
	maxTokens := flag.Int("max", 256, "Maximum tokens to generate")
	temperature := flag.Float64("temp", 0.9, "Sampling temperature")
	topP := flag.Float64("top-p", 0.9, "Top-p (nucleus) sampling")
	flag.Parse()

	if *weightsPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -weights is required")
		flag.Usage()
		os.Exit(1)
	}

	// Initialize Yent
	y, err := yent.New(*weightsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load Yent: %v\n", err)
		os.Exit(1)
	}
	defer y.Close()

	// Load Delta Voice if provided
	if *deltaPath != "" {
		if err := y.LoadDeltaVoice(*deltaPath); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to load delta: %v\n", err)
			os.Exit(1)
		}
		y.SetAlpha(float32(*alpha))
	}

	// Generate
	response, err := y.Generate(*prompt, *maxTokens, float32(*temperature), float32(*topP))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Generation failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(response)
}
