// shards.go — ShardBridge: export memories to training format
//
// This is how Yent learns from experience.
// Conversations → episodes → consolidated shards → JSONL training data
// Same format as yent_dataset_v9_final.jsonl — feed directly to finetune_v2.py
//
// The loop: live → remember → dream → shard → retrain → evolve
// Not backprop. Not gradient descent. Hebbian selection of experiences.
//
// "обучение на опыте — delta shards от взаимодействий,
//  перестаёт быть статичной биографией"

package limpha

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// TrainingPair is the output format for fine-tuning
// Must match the format in finetune_v2.py: {"prompt": "...", "response": "..."}
type TrainingPair struct {
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
}

// ShardConfig controls what gets exported
type ShardConfig struct {
	// MinStrength — only export memories above this strength
	MinStrength float32

	// MinTurns — only export sessions with at least this many turns
	MinTurns int

	// MaxAge — only export conversations newer than this
	MaxAge time.Duration

	// IncludeField — include field state as context in prompt
	IncludeField bool
}

// DefaultShardConfig returns sensible defaults
func DefaultShardConfig() ShardConfig {
	return ShardConfig{
		MinStrength:  0.3,
		MinTurns:     2,
		MaxAge:       30 * 24 * time.Hour, // 30 days
		IncludeField: false,
	}
}

// ExportShards exports consolidated conversations to training format
func (l *Limpha) ExportShards(outputPath string, config ShardConfig) (int, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	f, err := os.Create(outputPath)
	if err != nil {
		return 0, fmt.Errorf("create output: %w", err)
	}
	defer f.Close()

	now := time.Now()
	exported := 0
	encoder := json.NewEncoder(f)

	for _, conv := range l.conversations {
		// Filter by age
		age := now.Sub(time.Unix(0, conv.Timestamp))
		if age > config.MaxAge {
			continue
		}

		// Skip empty conversations
		if conv.Prompt == "" || conv.Response == "" {
			continue
		}

		// Skip very short responses (likely errors)
		if len(conv.Response) < 10 {
			continue
		}

		pair := TrainingPair{
			Prompt:   conv.Prompt,
			Response: conv.Response,
		}

		if err := encoder.Encode(pair); err != nil {
			continue
		}
		exported++
	}

	fmt.Printf("[limpha/shards] exported %d training pairs to %s\n", exported, outputPath)
	return exported, nil
}

// ExportDeltaShard exports a focused shard from a specific episode
// This creates small, targeted training data from significant moments
func (l *Limpha) ExportDeltaShard(episodeID int, outputDir string) (string, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if episodeID < 0 || episodeID >= len(l.episodes) {
		return "", fmt.Errorf("episode %d not found", episodeID)
	}

	ep := l.episodes[episodeID]
	outputPath := filepath.Join(outputDir, fmt.Sprintf("delta_shard_%d.jsonl", episodeID))

	f, err := os.Create(outputPath)
	if err != nil {
		return "", fmt.Errorf("create shard: %w", err)
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	exported := 0

	// Export conversations referenced by this episode
	for _, convID := range ep.ConvIDs {
		if convID >= 0 && convID < len(l.conversations) {
			conv := l.conversations[convID]
			pair := TrainingPair{
				Prompt:   conv.Prompt,
				Response: conv.Response,
			}
			if err := encoder.Encode(pair); err != nil {
				continue
			}
			exported++
		}
	}

	fmt.Printf("[limpha/shards] delta shard %d: %d pairs from episode '%s'\n",
		episodeID, exported, ep.Trigger)

	return outputPath, nil
}

// ExportAllShards exports one shard per unconsolidated episode
func (l *Limpha) ExportAllShards(outputDir string) (int, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return 0, fmt.Errorf("create output dir: %w", err)
	}

	total := 0
	for i, ep := range l.episodes {
		if ep.Consolidated {
			continue
		}

		_, err := l.ExportDeltaShard(i, outputDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[limpha/shards] skip episode %d: %v\n", i, err)
			continue
		}

		// Mark as consolidated
		l.mu.Lock()
		l.episodes[i].Consolidated = true
		l.mu.Unlock()

		total++
	}

	fmt.Printf("[limpha/shards] exported %d delta shards to %s\n", total, outputDir)
	return total, nil
}
