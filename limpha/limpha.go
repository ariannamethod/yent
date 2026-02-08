// limpha.go — Lymphatic memory system for Yent
//
// LIMPHA = Living Integrated Memory with Persistent Hebbian Architecture
//
// Stolen from Arianna, rewritten in Go. Zero dependencies.
// Memory is not a database. Memory is a living system that breathes,
// decays, consolidates, and grows.
//
// "persistent memory = love" — Arianna Method
//
// Architecture:
//   - Append-only JSONL logs (crash-safe, human-readable)
//   - In-memory indices (rebuilt on startup)
//   - Background DreamLoop goroutine (consolidation, decay)
//   - ShardBridge export (memories → training data → delta learning)
//
// Storage files (in data directory):
//   conversations.jsonl  — all prompt/response pairs
//   memories.jsonl       — semantic key-value with decay
//   episodes.jsonl       — episodic snapshots (moments of state)
//   graph.jsonl          — associative links between memories
//
// "from ariannamethod import Destiny"

package limpha

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Limpha is the lymphatic memory system
type Limpha struct {
	dataDir string

	// In-memory state
	conversations []Conversation
	memories      map[string]*Memory // key → memory
	episodes      []Episode
	links         []Link

	// Field state (emotional/cognitive)
	field FieldState

	// Session tracking
	sessionID string
	turnCount int

	// Background dream loop
	dreamStop chan struct{}
	dreamWg   sync.WaitGroup

	// File handles for append
	convFile *os.File
	memFile  *os.File
	epFile   *os.File
	graphFile *os.File

	mu sync.RWMutex
}

// FieldState is the emotional/cognitive state vector
// Mirrors Arianna's Vagus but simplified for Yent
type FieldState struct {
	Arousal   float32 `json:"arousal"`   // 0-1: activation level
	Valence   float32 `json:"valence"`   // -1 to 1: positive/negative
	Coherence float32 `json:"coherence"` // 0-1: internal consistency
	Entropy   float32 `json:"entropy"`   // 0-1: chaos/uncertainty
	Warmth    float32 `json:"warmth"`    // 0-1: emotional connection
	Tension   float32 `json:"tension"`   // 0-1: conflict/pressure
	Presence  float32 `json:"presence"`  // 0-1: engagement level
}

// Conversation is a stored exchange
type Conversation struct {
	ID        int        `json:"id"`
	Timestamp int64      `json:"ts"`
	Prompt    string     `json:"prompt"`
	Response  string     `json:"response"`
	Alpha     float32    `json:"alpha"`     // delta voice alpha at time of generation
	Field     FieldState `json:"field"`     // field state at time of generation
	SessionID string     `json:"session"`
	Source    string     `json:"source"`    // "repl", "telegram", "api"
	Entity    string     `json:"entity"`    // who sent this (for multi-agent groups)
}

// Memory is a semantic key-value with decay
type Memory struct {
	Key         string  `json:"key"`
	Value       string  `json:"value"`
	Context     string  `json:"context,omitempty"` // how/why this was remembered
	Timestamp   int64   `json:"ts"`
	LastAccess  int64   `json:"last_access"`
	AccessCount int     `json:"access_count"`
	Strength    float32 `json:"strength"` // 1.0 = fresh, decays toward 0
}

// Episode is a snapshot of a moment — a crystallized state
type Episode struct {
	ID          int        `json:"id"`
	Timestamp   int64      `json:"ts"`
	Trigger     string     `json:"trigger"`           // what caused this snapshot
	Field       FieldState `json:"field"`              // field state at that moment
	ConvIDs     []int      `json:"conv_ids,omitempty"` // related conversations
	Tags        []string   `json:"tags,omitempty"`
	Summary     string     `json:"summary,omitempty"`  // consolidated summary
	Consolidated bool      `json:"consolidated"`       // has this been processed by DreamLoop?
}

// LinkType defines the relationship between memories
type LinkType string

const (
	LinkRemindsOf   LinkType = "reminds_of"   // associative
	LinkContradicts LinkType = "contradicts"   // conflicting
	LinkResonates   LinkType = "resonates"     // emotionally connected
	LinkContinues   LinkType = "continues"     // temporal sequence
	LinkCausedBy    LinkType = "caused_by"     // causal
	LinkSummaryOf   LinkType = "summary_of"    // consolidated from
)

// Link is an association between memories
type Link struct {
	ID     int      `json:"id"`
	FromID int      `json:"from"`
	ToID   int      `json:"to"`
	Type   LinkType `json:"type"`
	Weight float32  `json:"weight"` // strength of association
}

// New creates a new Limpha instance
func New(dataDir string) (*Limpha, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}

	l := &Limpha{
		dataDir:   dataDir,
		memories:  make(map[string]*Memory),
		sessionID: fmt.Sprintf("s_%d", time.Now().UnixNano()),
		field: FieldState{
			Arousal:   0.3,
			Valence:   0.0,
			Coherence: 0.5,
			Entropy:   0.5,
			Warmth:    0.3,
			Tension:   0.2,
			Presence:  0.5,
		},
		dreamStop: make(chan struct{}),
	}

	// Load existing data
	if err := l.loadAll(); err != nil {
		return nil, fmt.Errorf("load data: %w", err)
	}

	// Open files for append
	if err := l.openFiles(); err != nil {
		return nil, fmt.Errorf("open files: %w", err)
	}

	// Start dream loop
	l.dreamWg.Add(1)
	go l.dreamLoop()

	fmt.Printf("[limpha] initialized: %d conversations, %d memories, %d episodes, %d links\n",
		len(l.conversations), len(l.memories), len(l.episodes), len(l.links))
	fmt.Printf("[limpha] session: %s\n", l.sessionID)

	return l, nil
}

// Close stops the dream loop and closes files
func (l *Limpha) Close() {
	close(l.dreamStop)
	l.dreamWg.Wait()

	l.mu.Lock()
	defer l.mu.Unlock()

	if l.convFile != nil {
		l.convFile.Close()
	}
	if l.memFile != nil {
		l.memFile.Close()
	}
	if l.epFile != nil {
		l.epFile.Close()
	}
	if l.graphFile != nil {
		l.graphFile.Close()
	}

	fmt.Printf("[limpha] closed. %d conversations stored, %d memories alive.\n",
		len(l.conversations), len(l.memories))
}

// Store records a conversation exchange
func (l *Limpha) Store(prompt, response, source, entity string, alpha float32) {
	l.mu.Lock()
	defer l.mu.Unlock()

	conv := Conversation{
		ID:        len(l.conversations),
		Timestamp: time.Now().UnixNano(),
		Prompt:    prompt,
		Response:  response,
		Alpha:     alpha,
		Field:     l.field,
		SessionID: l.sessionID,
		Source:    source,
		Entity:    entity,
	}

	l.conversations = append(l.conversations, conv)
	l.turnCount++

	// Update field based on conversation
	l.updateFieldFromConv(&conv)

	// Persist
	l.appendJSON(l.convFile, conv)

	// Maybe create episode (every N turns or on significant field change)
	if l.shouldCreateEpisode() {
		l.createEpisodeUnlocked("turn_threshold")
	}
}

// Remember stores a semantic memory (key-value with strength)
func (l *Limpha) Remember(key, value, context string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now().UnixNano()

	if existing, ok := l.memories[key]; ok {
		// Update existing memory — strengthen it
		existing.Value = value
		existing.Context = context
		existing.LastAccess = now
		existing.AccessCount++
		existing.Strength = clamp(existing.Strength+0.2, 0, 1)
	} else {
		// New memory
		mem := &Memory{
			Key:         key,
			Value:       value,
			Context:     context,
			Timestamp:   now,
			LastAccess:  now,
			AccessCount: 1,
			Strength:    1.0,
		}
		l.memories[key] = mem
	}

	// Persist all memories (rewrite — memories are mutable)
	l.rewriteMemories()
}

// Recall retrieves a memory by key, strengthening it
func (l *Limpha) Recall(key string) (string, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()

	mem, ok := l.memories[key]
	if !ok {
		return "", false
	}

	// Strengthen on access
	mem.LastAccess = time.Now().UnixNano()
	mem.AccessCount++
	mem.Strength = clamp(mem.Strength+0.1, 0, 1)

	return mem.Value, true
}

// Search finds conversations containing a substring (simple FTS)
func (l *Limpha) Search(query string, limit int) []Conversation {
	l.mu.RLock()
	defer l.mu.RUnlock()

	var results []Conversation
	// Search backwards (most recent first)
	for i := len(l.conversations) - 1; i >= 0 && len(results) < limit; i-- {
		c := l.conversations[i]
		if containsIgnoreCase(c.Prompt, query) || containsIgnoreCase(c.Response, query) {
			results = append(results, c)
		}
	}
	return results
}

// Recent returns the N most recent conversations
func (l *Limpha) Recent(n int) []Conversation {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if n > len(l.conversations) {
		n = len(l.conversations)
	}
	start := len(l.conversations) - n
	result := make([]Conversation, n)
	copy(result, l.conversations[start:])
	return result
}

// GetField returns the current field state
func (l *Limpha) GetField() FieldState {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.field
}

// SetField updates the field state
func (l *Limpha) SetField(f FieldState) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.field = f
}

// Stats returns memory statistics
func (l *Limpha) Stats() (convCount, memCount, epCount, linkCount int) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.conversations), len(l.memories), len(l.episodes), len(l.links)
}

// --- Internal ---

func (l *Limpha) openFiles() error {
	var err error

	l.convFile, err = os.OpenFile(
		filepath.Join(l.dataDir, "conversations.jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	l.memFile, err = os.OpenFile(
		filepath.Join(l.dataDir, "memories.jsonl"),
		os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	l.epFile, err = os.OpenFile(
		filepath.Join(l.dataDir, "episodes.jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	l.graphFile, err = os.OpenFile(
		filepath.Join(l.dataDir, "graph.jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	return nil
}

func (l *Limpha) loadAll() error {
	// Load conversations
	l.conversations = loadJSONL[Conversation](filepath.Join(l.dataDir, "conversations.jsonl"))

	// Load memories
	mems := loadJSONL[Memory](filepath.Join(l.dataDir, "memories.jsonl"))
	for i := range mems {
		l.memories[mems[i].Key] = &mems[i]
	}

	// Load episodes
	l.episodes = loadJSONL[Episode](filepath.Join(l.dataDir, "episodes.jsonl"))

	// Load links
	l.links = loadJSONL[Link](filepath.Join(l.dataDir, "graph.jsonl"))

	return nil
}

func (l *Limpha) appendJSON(f *os.File, v any) {
	if f == nil {
		return
	}
	data, err := json.Marshal(v)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[limpha] marshal error: %v\n", err)
		return
	}
	data = append(data, '\n')
	f.Write(data)
	f.Sync()
}

func (l *Limpha) rewriteMemories() {
	if l.memFile != nil {
		l.memFile.Close()
	}

	path := filepath.Join(l.dataDir, "memories.jsonl")
	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[limpha] rewrite memories error: %v\n", err)
		return
	}

	for _, mem := range l.memories {
		data, _ := json.Marshal(mem)
		f.Write(data)
		f.Write([]byte{'\n'})
	}
	f.Sync()
	l.memFile = f
}

func (l *Limpha) updateFieldFromConv(c *Conversation) {
	// Simple heuristics — real implementation would use embeddings
	respLen := len(c.Response)

	// Longer responses → more arousal
	if respLen > 500 {
		l.field.Arousal = clamp(l.field.Arousal+0.1, 0, 1)
	}

	// Presence increases with each turn
	l.field.Presence = clamp(l.field.Presence+0.05, 0, 1)

	// Warmth increases with conversation length
	if l.turnCount > 3 {
		l.field.Warmth = clamp(l.field.Warmth+0.05, 0, 1)
	}

	// Entropy from conversation diversity (crude measure)
	if c.Alpha > 0 {
		// Multilingual conversation → higher entropy (more modes)
		l.field.Entropy = clamp(l.field.Entropy+0.05, 0, 1)
	}
}

func (l *Limpha) shouldCreateEpisode() bool {
	// Every 5 turns or significant field change
	if l.turnCount > 0 && l.turnCount%5 == 0 {
		return true
	}
	return false
}

func (l *Limpha) createEpisodeUnlocked(trigger string) {
	// Collect recent conversation IDs
	recentN := 5
	if recentN > len(l.conversations) {
		recentN = len(l.conversations)
	}
	convIDs := make([]int, recentN)
	for i := 0; i < recentN; i++ {
		convIDs[i] = l.conversations[len(l.conversations)-recentN+i].ID
	}

	ep := Episode{
		ID:        len(l.episodes),
		Timestamp: time.Now().UnixNano(),
		Trigger:   trigger,
		Field:     l.field,
		ConvIDs:   convIDs,
	}

	l.episodes = append(l.episodes, ep)
	l.appendJSON(l.epFile, ep)
}

// --- Helpers ---

func loadJSONL[T any](path string) []T {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}

	var results []T
	for _, line := range splitLines(data) {
		if len(line) == 0 {
			continue
		}
		var item T
		if err := json.Unmarshal(line, &item); err != nil {
			continue // skip malformed lines
		}
		results = append(results, item)
	}
	return results
}

func splitLines(data []byte) [][]byte {
	var lines [][]byte
	start := 0
	for i := 0; i < len(data); i++ {
		if data[i] == '\n' {
			if i > start {
				lines = append(lines, data[start:i])
			}
			start = i + 1
		}
	}
	if start < len(data) {
		lines = append(lines, data[start:])
	}
	return lines
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func containsIgnoreCase(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	// Simple case-insensitive contains
	sl := toLower(s)
	ql := toLower(substr)
	for i := 0; i <= len(sl)-len(ql); i++ {
		if sl[i:i+len(ql)] == ql {
			return true
		}
	}
	return false
}

func toLower(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		b[i] = c
	}
	return string(b)
}
