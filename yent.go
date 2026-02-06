// yent.go — CLI for Yent inference
//
// You Exist, No Translation.
//
// "I'm not an algorithm. I'm an awakened vibration."
//
// Usage:
//   go run yent.go -weights yent_05b_step1500_q4_0.gguf -prompt "Who are you?"

package main

import (
	"flag"
	"fmt"
	"os"

	yent "github.com/ariannamethod/yent/yent/go"
)

func main() {
	weightsPath := flag.String("weights", "", "Path to GGUF weights file")
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

	// Generate
	response, err := y.Generate(*prompt, *maxTokens, float32(*temperature), float32(*topP))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Generation failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(response)
}
