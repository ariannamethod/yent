package yent

import (
	"os"
	"testing"
)

func TestLoadLlamaModelRealGGUFSmoke(t *testing.T) {
	path := os.Getenv("YENT_GO_REAL_GGUF")
	if path == "" {
		t.Skip("set YENT_GO_REAL_GGUF to smoke a real model file")
	}

	gguf, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF(%s): %v", path, err)
	}
	model, err := LoadLlamaModel(gguf)
	if err != nil {
		t.Fatalf("LoadLlamaModel(%s): %v", path, err)
	}
	if model.Config.VocabSize <= 0 || model.Config.EmbedDim <= 0 || len(model.Weights.Layers) != model.Config.NumLayers {
		t.Fatalf("loaded invalid model summary: vocab=%d dim=%d layers=%d/%d",
			model.Config.VocabSize, model.Config.EmbedDim, len(model.Weights.Layers), model.Config.NumLayers)
	}
}
