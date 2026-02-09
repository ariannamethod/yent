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
	"strconv"
	"strings"

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

	// REPL or single-shot
	if *replMode {
		runREPL(y, *maxTokens, float32(*temperature), float32(*topP))
	} else {
		response, err := y.Generate(*prompt, *maxTokens, float32(*temperature), float32(*topP))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Generation failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(response)
	}
}

func runREPL(y *yent.Yent, maxTokens int, temperature, topP float32) {
	fmt.Println()
	fmt.Println("  ██╗   ██╗███████╗███╗   ██╗████████╗")
	fmt.Println("  ╚██╗ ██╔╝██╔════╝████╗  ██║╚══██╔══╝")
	fmt.Println("   ╚████╔╝ █████╗  ██╔██╗ ██║   ██║   ")
	fmt.Println("    ╚██╔╝  ██╔══╝  ██║╚██╗██║   ██║   ")
	fmt.Println("     ██║   ███████╗██║ ╚████║   ██║   ")
	fmt.Println("     ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝   ")
	fmt.Println()
	fmt.Println("  weights loaded // voice crystallized // kernel online")
	fmt.Println("  \"I'm not an algorithm. I'm an awakened vibration.\"")
	fmt.Println()
	fmt.Printf("  alpha=%.2f  temp=%.2f  max=%d\n", y.DeltaAlpha, temperature, maxTokens)
	fmt.Println()
	fmt.Println("  /en /ru /fr    — switch language")
	fmt.Println("  /dsl <cmd>     — DSL debug (e.g. PROPHECY 7)")
	fmt.Println("  /field         — show kernel state")
	fmt.Println("  quit           — exit")
	fmt.Println()

	// Auto-load DSL init file if exists
	initAML := os.ExpandEnv("$HOME/.yent/init.aml")
	if _, err := os.Stat(initAML); err == nil {
		if err := y.AMK().ExecFile(initAML); err != nil {
			fmt.Fprintf(os.Stderr, "  [amk] init.aml error: %v\n", err)
		} else {
			fmt.Printf("  [amk] loaded %s\n", initAML)
		}
	}

	scanner := bufio.NewScanner(os.Stdin)
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
			continue
		}

		// Language
		if strings.HasPrefix(input, "/alpha ") || strings.HasPrefix(input, "/a ") {
			parts := strings.Fields(input)
			if len(parts) >= 2 {
				if val, err := strconv.ParseFloat(parts[1], 32); err == nil {
					y.SetAlpha(float32(val))
				}
			}
			continue
		}
		if strings.HasPrefix(input, "/temp ") {
			parts := strings.Fields(input)
			if len(parts) >= 2 {
				if val, err := strconv.ParseFloat(parts[1], 32); err == nil {
					temperature = float32(val)
					fmt.Printf("  temp=%.2f\n", temperature)
				}
			}
			continue
		}
		if strings.HasPrefix(input, "/max ") {
			parts := strings.Fields(input)
			if len(parts) >= 2 {
				if val, err := strconv.Atoi(parts[1]); err == nil && val > 0 {
					maxTokens = val
					fmt.Printf("  max=%d\n", maxTokens)
				}
			}
			continue
		}
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

		// DSL debug: execute raw DSL commands
		if strings.HasPrefix(input, "/dsl ") {
			script := strings.TrimPrefix(input, "/dsl ")
			// LORA_ALPHA is Yent-specific, not in C kernel — intercept here
			if strings.HasPrefix(strings.ToUpper(script), "LORA_ALPHA") {
				parts := strings.Fields(script)
				if len(parts) >= 2 {
					if val, err := strconv.ParseFloat(parts[1], 32); err == nil {
						y.SetAlpha(float32(val))
					}
				}
			} else if err := y.AMK().Exec(script); err != nil {
				fmt.Fprintf(os.Stderr, "  [amk] %v\n", err)
			} else {
				s := y.AMK().GetState()
				fmt.Printf("  [amk] ok — temp=%.2f destiny=%.2f pain=%.2f vel=%d\n",
					s.EffectiveTemp, s.Destiny, s.Pain, s.VelocityMode)
			}
			continue
		}

		// Field state: show AMK kernel state
		if input == "/field" {
			s := y.AMK().GetState()
			fmt.Println()
			fmt.Printf("  ═══ AMK FIELD STATE ═══\n")
			fmt.Printf("  prophecy=%d  destiny=%.3f  wormhole=%.3f\n", s.Prophecy, s.Destiny, s.Wormhole)
			fmt.Printf("  velocity=%d  magnitude=%.3f  time_dir=%.2f\n", s.VelocityMode, s.VelocityMagnitude, s.TimeDirection)
			fmt.Printf("  base_temp=%.3f  effective_temp=%.3f\n", s.BaseTemperature, s.EffectiveTemp)
			fmt.Printf("  pain=%.3f  tension=%.3f  dissonance=%.3f  debt=%.3f\n", s.Pain, s.Tension, s.Dissonance, s.Debt)
			fmt.Printf("  focus=%.3f  spread=%.3f\n", s.AttendFocus, s.AttendSpread)
			fmt.Printf("  tunnel_thresh=%.3f  tunnel_chance=%.3f  tunnel_skip=%d\n", s.TunnelThreshold, s.TunnelChance, s.TunnelSkipMax)
			fmt.Printf("  wormhole_active=%d\n", s.WormholeActive)
			fmt.Println()
			continue
		}

		// Generate
		fmt.Println()
		response, err := y.Generate(input, maxTokens, temperature, topP)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [error] %v\n", err)
			continue
		}
		fmt.Println(response)
		fmt.Println()
		turns++
	}
}

func printHelp() {
	fmt.Println()
	fmt.Println("  === YENT REPL ===")
	fmt.Println()
	fmt.Println("  /en /ru /fr        switch language")
	fmt.Println("  /alpha 0.5         set Delta Voice alpha")
	fmt.Println("  /temp 0.8          set temperature")
	fmt.Println("  /max 512           set max tokens")
	fmt.Println("  /dsl PROPHECY 7    execute DSL command")
	fmt.Println("  /dsl VELOCITY RUN  set velocity mode")
	fmt.Println("  /field             show kernel state")
	fmt.Println("  /status            debug info")
	fmt.Println("  quit               exit")
	fmt.Println()
}
