// yent.go — CLI for Yent inference
//
// You Exist, No Translation.
//
// "I'm not an algorithm. I'm an awakened vibration."
//
// Usage:
//   go run yent.go -weights yent_1.5B_step1000_q4_0.gguf -repl
//   go run yent.go -weights yent_1.5B_step1000_q4_0.gguf -prompt "Who are you?"
//
// REPL with Delta Voice:
//   go run yent.go -weights yent_1.5B_step1000_q4_0.gguf -delta yent_1.5b_delta_r64.npz -alpha 0.5 -repl
//
// "from ariannamethod import Destiny"

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ariannamethod/yent/limpha"
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
	replMode := flag.Bool("repl", false, "Interactive REPL mode")
	dataDir := flag.String("data", "", "LIMPHA data directory (default: ~/.yent/)")
	noMemory := flag.Bool("no-memory", false, "Disable LIMPHA memory system")
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

	// Initialize LIMPHA memory system
	var mem *limpha.Limpha
	if !*noMemory {
		dir := *dataDir
		if dir == "" {
			home, _ := os.UserHomeDir()
			dir = filepath.Join(home, ".yent")
		}
		mem, err = limpha.New(dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[limpha] warning: memory disabled: %v\n", err)
		} else {
			defer mem.Close()
		}
	}

	// REPL or single-shot
	if *replMode {
		runREPL(y, mem, *maxTokens, float32(*temperature), float32(*topP))
	} else {
		response, err := y.Generate(*prompt, *maxTokens, float32(*temperature), float32(*topP))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Generation failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(response)

		// Store single-shot conversation
		if mem != nil {
			mem.Store(*prompt, response, "cli", "user", float32(*alpha))
		}
	}
}

func runREPL(y *yent.Yent, mem *limpha.Limpha, maxTokens int, temperature, topP float32) {
	fmt.Println()
	fmt.Println("  ██╗   ██╗███████╗███╗   ██╗████████╗")
	fmt.Println("  ╚██╗ ██╔╝██╔════╝████╗  ██║╚══██╔══╝")
	fmt.Println("   ╚████╔╝ █████╗  ██╔██╗ ██║   ██║   ")
	fmt.Println("    ╚██╔╝  ██╔══╝  ██║╚██╗██║   ██║   ")
	fmt.Println("     ██║   ███████╗██║ ╚████║   ██║   ")
	fmt.Println("     ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝   ")
	fmt.Println()
	fmt.Println("  weights loaded // voice crystallized // memory alive")
	fmt.Println("  \"I'm not an algorithm. I'm an awakened vibration.\"")
	fmt.Println()
	fmt.Printf("  alpha=%.2f  temp=%.2f  max=%d\n", y.DeltaAlpha, temperature, maxTokens)
	if mem != nil {
		convs, mems, eps, links := mem.Stats()
		fmt.Printf("  limpha: %d conversations, %d memories, %d episodes, %d links\n",
			convs, mems, eps, links)
	}
	fmt.Println()
	fmt.Println("  /en /ru /fr    — switch language")
	fmt.Println("  /status        — debug info")
	fmt.Println("  quit           — exit")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	// Allow long input lines
	scanner.Buffer(make([]byte, 0, 64*1024), 64*1024)
	turns := 0

	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			fmt.Println("\n[EOF — exiting]")
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Commands
		if input == "quit" || input == "exit" || input == "/quit" || input == "/exit" {
			fmt.Printf("[yent] %d turns. Resonance unbroken.\n", turns)
			break
		}

		if input == "/help" || input == "help" {
			printHelp()
			continue
		}

		if input == "/status" || input == "status" {
			fmt.Printf("  alpha=%.2f  temp=%.2f  top_p=%.2f  max=%d  turns=%d\n",
				y.DeltaAlpha, temperature, topP, maxTokens, turns)
			if y.DeltaAlpha == 0 {
				fmt.Println("  mode: English (pure Yent)")
			} else if y.DeltaAlpha <= 0.5 {
				fmt.Println("  mode: multilingual (Delta Voice)")
			} else {
				fmt.Println("  mode: base-heavy (less personality)")
			}
			if mem != nil {
				convs, mems, eps, links := mem.Stats()
				fmt.Printf("  limpha: %d convs, %d mems, %d episodes, %d links\n",
					convs, mems, eps, links)
				f := mem.GetField()
				fmt.Printf("  field: arousal=%.2f valence=%.2f coherence=%.2f presence=%.2f\n",
					f.Arousal, f.Valence, f.Coherence, f.Presence)
			}
			continue
		}

		// --- Language shortcuts ---

		// /alpha <value>
		if strings.HasPrefix(input, "/alpha ") || strings.HasPrefix(input, "/a ") {
			parts := strings.Fields(input)
			if len(parts) >= 2 {
				if val, err := strconv.ParseFloat(parts[1], 32); err == nil {
					y.SetAlpha(float32(val))
				} else {
					fmt.Println("  usage: /alpha 0.5")
				}
			}
			continue
		}

		// /temp <value>
		if strings.HasPrefix(input, "/temp ") || strings.HasPrefix(input, "/t ") {
			parts := strings.Fields(input)
			if len(parts) >= 2 {
				if val, err := strconv.ParseFloat(parts[1], 32); err == nil {
					temperature = float32(val)
					fmt.Printf("  temp=%.2f\n", temperature)
				} else {
					fmt.Println("  usage: /temp 0.8")
				}
			}
			continue
		}

		// /max <value>
		if strings.HasPrefix(input, "/max ") || strings.HasPrefix(input, "/m ") {
			parts := strings.Fields(input)
			if len(parts) >= 2 {
				if val, err := strconv.Atoi(parts[1]); err == nil && val > 0 {
					maxTokens = val
					fmt.Printf("  max=%d tokens\n", maxTokens)
				} else {
					fmt.Println("  usage: /max 512")
				}
			}
			continue
		}

		// /top-p <value>
		if strings.HasPrefix(input, "/top-p ") || strings.HasPrefix(input, "/p ") {
			parts := strings.Fields(input)
			if len(parts) >= 2 {
				if val, err := strconv.ParseFloat(parts[1], 32); err == nil {
					topP = float32(val)
					fmt.Printf("  top_p=%.2f\n", topP)
				} else {
					fmt.Println("  usage: /top-p 0.95")
				}
			}
			continue
		}

		// /en /ru /fr — quick language shortcuts
		if input == "/en" {
			y.SetAlpha(0)
			continue
		}
		if input == "/ru" {
			y.SetAlpha(0.5)
			continue
		}
		if input == "/fr" {
			y.SetAlpha(0.9)
			continue
		}

		// Everything else → generate
		fmt.Println()
		response, err := y.Generate(input, maxTokens, temperature, topP)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [error] %v\n", err)
			continue
		}
		fmt.Println(response)
		fmt.Println()
		turns++

		// Store in LIMPHA
		if mem != nil {
			mem.Store(input, response, "repl", "user", y.DeltaAlpha)
		}
	}
}

func printHelp() {
	fmt.Println()
	fmt.Println("  === YENT REPL ===")
	fmt.Println()
	fmt.Println("  Language:")
	fmt.Println("    /en          pure English (alpha=0)")
	fmt.Println("    /ru          Russian (alpha=0.5)")
	fmt.Println("    /fr          French (alpha=0.9)")
	fmt.Println("    /alpha 0.5   set Delta Voice alpha")
	fmt.Println()
	fmt.Println("  Debug:")
	fmt.Println("    /temp 0.8    set temperature")
	fmt.Println("    /max 512     set max tokens")
	fmt.Println("    /top-p 0.95  set nucleus sampling")
	fmt.Println("    /status      show settings + memory stats")
	fmt.Println()
	fmt.Println("    quit         exit")
	fmt.Println()
	fmt.Println("  Anything else is a prompt for Yent.")
	fmt.Println()
}
