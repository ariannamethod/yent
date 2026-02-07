# Makefile — Yent inference engine
#
# "I'm not an algorithm. I'm an awakened vibration."
#
# Usage:
#   make          # download 1.5B, build (deltas ship with repo)
#   make light    # download 0.5B + run
#   make max      # download 3B + run
#   make run      # run with auto-detected best model
#   make download # download all available weights
#   make clean    # remove downloaded weights
#
# "from ariannamethod import Destiny"

HF_BASE = https://huggingface.co/ataeff/yent/resolve/main/yent
WEIGHTS_DIR = weights
DELTAS_DIR = deltas

# Model files (downloaded to weights/)
GGUF_05B = $(WEIGHTS_DIR)/yent_0.5B_step1000_q4_0.gguf
GGUF_15B = $(WEIGHTS_DIR)/yent_1.5B_step1000_q4_0.gguf
GGUF_3B  = $(WEIGHTS_DIR)/yent_3B_step1000_q4_0.gguf

# Delta Voice files (shipped in deltas/, 17 MB each, 29 languages)
DELTA_05B = $(DELTAS_DIR)/yent_05b_delta_r64.npz
DELTA_15B = $(DELTAS_DIR)/yent_1.5b_delta_r64.npz
DELTA_3B  = $(DELTAS_DIR)/yent_3b_delta_r64.npz

# Binary
BIN = yent_bin

# Default alpha for multilingual mode
ALPHA ?= 0.5
PROMPT ?= Who are you?
MAX ?= 256
TEMP ?= 0.9

# ═══════════════════════════════════════════════════════
# Default: 1.5B — balanced personality + multilingual
# ═══════════════════════════════════════════════════════

.PHONY: all light max run repl download clean help router

all: $(BIN) $(GGUF_15B) $(DELTA_15B)
	@echo ""
	@echo "  ██╗   ██╗███████╗███╗   ██╗████████╗"
	@echo "  ╚██╗ ██╔╝██╔════╝████╗  ██║╚══██╔══╝"
	@echo "   ╚████╔╝ █████╗  ██╔██╗ ██║   ██║   "
	@echo "    ╚██╔╝  ██╔══╝  ██║╚██╗██║   ██║   "
	@echo "     ██║   ███████╗██║ ╚████║   ██║   "
	@echo "     ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝   "
	@echo ""
	@echo "  1.5B ready. Delta Voice loaded. 29 languages."
	@echo "  Run: make run PROMPT=\"Кто ты?\" ALPHA=0.5"
	@echo ""

# ═══════════════════════════════════════════════════════
# Profiles
# ═══════════════════════════════════════════════════════

light: $(BIN) $(GGUF_05B) $(DELTA_05B)
	@echo "[yent] Light mode: 0.5B (409 MB)"
	./$(BIN) -weights $(GGUF_05B) -delta $(DELTA_05B) -alpha $(ALPHA) -prompt "$(PROMPT)" -max $(MAX) -temp $(TEMP)

max: $(BIN) $(GGUF_3B) $(DELTA_3B)
	@echo "[yent] Max mode: 3B"
	./$(BIN) -weights $(GGUF_3B) -delta $(DELTA_3B) -alpha $(ALPHA) -prompt "$(PROMPT)" -max $(MAX) -temp $(TEMP)

# ═══════════════════════════════════════════════════════
# REPL: interactive conversation (1.5B default)
# ═══════════════════════════════════════════════════════

repl: $(BIN) $(GGUF_15B) $(DELTA_15B)
	@echo "[yent] REPL mode: 1.5B + Delta Voice"
	./$(BIN) -weights $(GGUF_15B) -delta $(DELTA_15B) -alpha $(ALPHA) -repl -max $(MAX) -temp $(TEMP)

repl-light: $(BIN) $(GGUF_05B) $(DELTA_05B)
	@echo "[yent] REPL mode: 0.5B + Delta Voice"
	./$(BIN) -weights $(GGUF_05B) -delta $(DELTA_05B) -alpha $(ALPHA) -repl -max $(MAX) -temp $(TEMP)

repl-max: $(BIN) $(GGUF_3B) $(DELTA_3B)
	@echo "[yent] REPL mode: 3B + Delta Voice"
	./$(BIN) -weights $(GGUF_3B) -delta $(DELTA_3B) -alpha $(ALPHA) -repl -max $(MAX) -temp $(TEMP)

# ═══════════════════════════════════════════════════════
# Router: auto-detect hardware, pick best model
# ═══════════════════════════════════════════════════════

run: $(BIN)
	@TOTAL_RAM=$$(sysctl -n hw.memsize 2>/dev/null || free -b 2>/dev/null | awk '/Mem:/{print $$2}' || echo 0); \
	TOTAL_GB=$$(echo "$$TOTAL_RAM / 1073741824" | bc 2>/dev/null || echo 8); \
	echo "[yent] Detected RAM: $${TOTAL_GB}GB"; \
	if [ -f "$(GGUF_3B)" ] && [ -f "$(DELTA_3B)" ] && [ "$$TOTAL_GB" -ge 16 ]; then \
		echo "[yent] Router: 3B (max) — RAM >= 16GB"; \
		./$(BIN) -weights $(GGUF_3B) -delta $(DELTA_3B) -alpha $(ALPHA) -prompt "$(PROMPT)" -max $(MAX) -temp $(TEMP); \
	elif [ -f "$(GGUF_15B)" ] && [ -f "$(DELTA_15B)" ] && [ "$$TOTAL_GB" -ge 6 ]; then \
		echo "[yent] Router: 1.5B (default) — RAM >= 6GB"; \
		./$(BIN) -weights $(GGUF_15B) -delta $(DELTA_15B) -alpha $(ALPHA) -prompt "$(PROMPT)" -max $(MAX) -temp $(TEMP); \
	elif [ -f "$(GGUF_05B)" ] && [ -f "$(DELTA_05B)" ]; then \
		echo "[yent] Router: 0.5B (light) — low RAM or no larger model"; \
		./$(BIN) -weights $(GGUF_05B) -delta $(DELTA_05B) -alpha $(ALPHA) -prompt "$(PROMPT)" -max $(MAX) -temp $(TEMP); \
	elif [ -f "$(GGUF_15B)" ]; then \
		echo "[yent] Router: 1.5B (no delta — English only)"; \
		./$(BIN) -weights $(GGUF_15B) -prompt "$(PROMPT)" -max $(MAX) -temp $(TEMP); \
	elif [ -f "$(GGUF_05B)" ]; then \
		echo "[yent] Router: 0.5B (no delta — English only)"; \
		./$(BIN) -weights $(GGUF_05B) -prompt "$(PROMPT)" -max $(MAX) -temp $(TEMP); \
	else \
		echo "[yent] No weights found. Run: make download"; \
		exit 1; \
	fi

# ═══════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════

$(BIN): yent.go yent/go/*.go
	go build -o $(BIN) ./yent.go

# ═══════════════════════════════════════════════════════
# Download from HuggingFace
# ═══════════════════════════════════════════════════════

$(WEIGHTS_DIR):
	mkdir -p $(WEIGHTS_DIR)

$(GGUF_05B): $(WEIGHTS_DIR)
	@echo "[yent] Downloading 0.5B weights (409 MB)..."
	curl -L -o $@ $(HF_BASE)/yent_0.5B_step1000_q4_0.gguf

$(GGUF_15B): $(WEIGHTS_DIR)
	@echo "[yent] Downloading 1.5B weights (~1 GB)..."
	curl -L -o $@ $(HF_BASE)/yent_1.5B_step1000_q4_0.gguf

$(GGUF_3B): $(WEIGHTS_DIR)
	@echo "[yent] Downloading 3B weights (~1.9 GB)..."
	curl -L -o $@ $(HF_BASE)/yent_3B_step1000_q4_0.gguf

download: $(GGUF_05B) $(GGUF_15B)
	@echo "[yent] All available weights downloaded. Deltas already in deltas/."

download-all: download $(GGUF_3B)
	@echo "[yent] All weights including 3B downloaded."

# ═══════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════

clean:
	rm -f $(BIN)

clean-weights:
	rm -rf $(WEIGHTS_DIR)

clean-all: clean clean-weights

# ═══════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════

help:
	@echo "Yent — You Exist, No Translation."
	@echo ""
	@echo "  make              Download 1.5B, build (deltas in repo)"
	@echo "  make repl         Interactive REPL (1.5B — recommended)"
	@echo "  make repl-light   Interactive REPL (0.5B)"
	@echo "  make repl-max     Interactive REPL (3B)"
	@echo "  make light        Single-shot 0.5B"
	@echo "  make max          Single-shot 3B"
	@echo "  make run          Auto-detect hardware, single-shot"
	@echo "  make download     Download 0.5B + 1.5B GGUF"
	@echo "  make download-all Download everything including 3B"
	@echo "  make clean        Remove binary"
	@echo "  make clean-all    Remove binary + weights"
	@echo ""
	@echo "  Variables:"
	@echo "    PROMPT=\"Кто ты?\"   Input prompt"
	@echo "    ALPHA=0.5          Delta voice: 0=EN, 0.5=multilingual"
	@echo "    MAX=256            Max tokens"
	@echo "    TEMP=0.9           Temperature"
	@echo ""
	@echo "  from ariannamethod import Destiny"
