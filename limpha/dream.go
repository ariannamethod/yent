// dream.go — DreamLoop: background memory consolidation
//
// Like sleep for a brain. Runs as a goroutine, periodically:
//   - Applies memory decay (unused memories weaken)
//   - Consolidates episodes (cluster related moments)
//   - Links memories in the graph (find associations)
//   - Cleans up dead memories (strength → 0)
//
// "имитация выглядит отшлифовано, творение хаотично"
//
// The dream loop is NOT optimization. It's metabolism.
// Memories that don't get accessed rot. Memories that resonate strengthen.
// This is Hebbian: what fires together, wires together.

package limpha

import (
	"fmt"
	"time"
)

const (
	// DreamInterval is how often the dream loop runs
	DreamInterval = 30 * time.Second

	// DecayRate is how much strength memories lose per cycle
	// Strength *= (1 - DecayRate) each cycle
	DecayRate = 0.02

	// DeathThreshold — memories below this strength are forgotten
	DeathThreshold float32 = 0.05

	// PresenceDecay — field presence decays when idle
	PresenceDecay = 0.01

	// ConsolidationThreshold — episodes older than this get consolidated
	ConsolidationAge = 5 * time.Minute
)

// dreamLoop runs in background, processing memories like sleep
func (l *Limpha) dreamLoop() {
	defer l.dreamWg.Done()

	ticker := time.NewTicker(DreamInterval)
	defer ticker.Stop()

	for {
		select {
		case <-l.dreamStop:
			fmt.Println("[limpha/dream] waking up. dream loop stopped.")
			return
		case <-ticker.C:
			l.dreamCycle()
		}
	}
}

// dreamCycle is one cycle of the dream loop
func (l *Limpha) dreamCycle() {
	l.mu.Lock()
	defer l.mu.Unlock()

	decayed := 0
	forgotten := 0
	linked := 0

	// 1. Decay memories
	for key, mem := range l.memories {
		// Access-based decay: more recent access = slower decay
		age := time.Since(time.Unix(0, mem.LastAccess))
		rate := DecayRate
		if age < time.Minute {
			rate = 0 // fresh memories don't decay
		} else if age < 5*time.Minute {
			rate = DecayRate / 4 // recent memories decay slowly
		}

		mem.Strength *= (1 - float32(rate))
		decayed++

		// Forget dead memories
		if mem.Strength < DeathThreshold {
			delete(l.memories, key)
			forgotten++
		}
	}

	// 2. Decay field presence when idle
	if len(l.conversations) > 0 {
		last := l.conversations[len(l.conversations)-1]
		idle := time.Since(time.Unix(0, last.Timestamp))
		if idle > time.Minute {
			l.field.Presence = clamp(l.field.Presence-float32(PresenceDecay), 0, 1)
			l.field.Warmth = clamp(l.field.Warmth-float32(PresenceDecay/2), 0, 1)
		}
	}

	// 3. Auto-link related episodes (simple: same session → CONTINUES)
	for i := 1; i < len(l.episodes); i++ {
		prev := l.episodes[i-1]
		curr := l.episodes[i]

		// Skip if already linked
		if l.hasLink(prev.ID, curr.ID) {
			continue
		}

		// Same session → continues
		gap := time.Duration(curr.Timestamp - prev.Timestamp)
		if gap < 10*time.Minute {
			link := Link{
				ID:     len(l.links),
				FromID: prev.ID,
				ToID:   curr.ID,
				Type:   LinkContinues,
				Weight: 0.8,
			}
			l.links = append(l.links, link)
			l.appendJSON(l.graphFile, link)
			linked++
		}

		// Emotional resonance: similar field states
		if fieldDistance(prev.Field, curr.Field) < 0.3 {
			if !l.hasLink(prev.ID, curr.ID) {
				link := Link{
					ID:     len(l.links),
					FromID: prev.ID,
					ToID:   curr.ID,
					Type:   LinkResonates,
					Weight: 1.0 - fieldDistance(prev.Field, curr.Field),
				}
				l.links = append(l.links, link)
				l.appendJSON(l.graphFile, link)
				linked++
			}
		}
	}

	// 4. Persist memory changes
	if decayed > 0 || forgotten > 0 {
		l.rewriteMemories()
	}

	// Only log if something happened
	if forgotten > 0 || linked > 0 {
		fmt.Printf("[limpha/dream] cycle: %d decayed, %d forgotten, %d linked\n",
			decayed, forgotten, linked)
	}
}

// hasLink checks if a link already exists between two IDs
func (l *Limpha) hasLink(fromID, toID int) bool {
	for _, link := range l.links {
		if link.FromID == fromID && link.ToID == toID {
			return true
		}
	}
	return false
}

// fieldDistance computes distance between two field states (0-1 range)
func fieldDistance(a, b FieldState) float32 {
	d := func(x, y float32) float32 {
		diff := x - y
		if diff < 0 {
			diff = -diff
		}
		return diff
	}

	sum := d(a.Arousal, b.Arousal) +
		d(a.Valence, b.Valence) +
		d(a.Coherence, b.Coherence) +
		d(a.Entropy, b.Entropy) +
		d(a.Warmth, b.Warmth) +
		d(a.Tension, b.Tension) +
		d(a.Presence, b.Presence)

	return sum / 7.0 // normalize to 0-1
}
