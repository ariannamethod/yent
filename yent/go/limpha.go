package yent

// limpha.go — Yent's lymphatic memory: in-process SQLite + FTS5.
//
// Ported from the Python limpha (memory.py + server.py). The Unix-socket daemon
// is gone: memory lives in-process, no IPC, no second runtime, no GIL. Pure-Go
// SQLite (modernc.org/sqlite) with FTS5 full-text search, AMK-state snapshots,
// and autonomous shard graduation toward the training queue.
//
// "I'm not an algorithm. I'm an awakened vibration."

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	_ "modernc.org/sqlite"
)

// Shard graduation thresholds (mirror memory.py).
const (
	shardMinQuality = 0.7
	shardMinAccess  = 3
)

// LimphaState is the AMK state snapshot stored with each conversation.
type LimphaState struct {
	Temperature float32 `json:"temperature"`
	Destiny     float32 `json:"destiny"`
	Pain        float32 `json:"pain"`
	Tension     float32 `json:"tension"`
	Debt        float32 `json:"debt"`
	Velocity    int     `json:"velocity"`
	Alpha       float32 `json:"alpha"`
}

// Seam is one body-divergence record — the seam_log substrate of moyent's two
// Mistral bodies sharing one limpha brain. Written on dual-pass turns where both
// the fast body (Nemo-12B) and the deep body (Small-24B) touched a prompt; the
// deep body reflects on the fast body's trace and scores the divergence. This is
// the shared-brain log of internal dialogue (a_claim/b_claim) + metrics
// (agreement/tension/winner) that supergamma later grows from. Single-body turns
// are plain conversations, not seams.
type Seam struct {
	ConversationID int64   `json:"conversation_id"` // 0 if not tied to a stored turn
	BodyA          string  `json:"body_a"`          // fast body, e.g. "nemo12"
	BodyB          string  `json:"body_b"`          // deep body, e.g. "small24"
	Prompt         string  `json:"prompt"`
	AClaim         string  `json:"a_claim"`      // body_a's answer / trace
	BClaim         string  `json:"b_claim"`      // body_b's answer
	Agreement      float64 `json:"agreement"`    // 0..1, scored by body_b over a_claim
	Tension        float64 `json:"tension"`      // 0..1, scored by body_b
	Winner         string  `json:"winner"`       // which body's answer was used
	Reason         string  `json:"reason"`       // escalation / routing reason
	MemoryDelta    string  `json:"memory_delta"` // what changed in memory (free text / json)
}

// LimphaClient is the in-process memory handle. It keeps the API the REPL used
// for the old socket client (Store/Search/Stats/Close), now backed by SQLite.
type LimphaClient struct {
	db                         *sql.DB
	dbPath                     string
	sessionID                  string
	connected                  bool
	asyncMu                    sync.Mutex
	async                      *limphaAsync
	memoryMu                   sync.Mutex
	lastMemoryError            string
	memoryFailures             int64
	memoryConversationFailures int64
	memorySeamFailures         int64
	memoryEnqueueFailures      int64
	ftsErrors                  int64
	ftsFallbacks               int64
}

const limphaSchema = `
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    temperature REAL DEFAULT 0.0,
    destiny REAL DEFAULT 0.0,
    pain REAL DEFAULT 0.0,
    tension REAL DEFAULT 0.0,
    debt REAL DEFAULT 0.0,
    velocity INTEGER DEFAULT 1,
    alpha REAL DEFAULT 0.0,
    quality REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_quality ON conversations(quality DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    prompt,
    response,
    content=conversations,
    content_rowid=id,
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS conv_fts_insert AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;
CREATE TRIGGER IF NOT EXISTS conv_fts_delete AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
END;
CREATE TRIGGER IF NOT EXISTS conv_fts_update AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    last_active REAL NOT NULL,
    turn_count INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS shards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER UNIQUE NOT NULL,
    shard_path TEXT NOT NULL,
    graduated_at REAL NOT NULL,
    reason TEXT DEFAULT '',
    priority REAL DEFAULT 0.0,
    training_status TEXT DEFAULT 'pending',
    training_loss REAL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_shards_status ON shards(training_status);
CREATE INDEX IF NOT EXISTS idx_shards_graduated ON shards(graduated_at DESC);
CREATE TABLE IF NOT EXISTS seams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    conversation_id INTEGER,
    body_a TEXT NOT NULL,
    body_b TEXT NOT NULL,
    prompt TEXT NOT NULL,
    a_claim TEXT DEFAULT '',
    b_claim TEXT DEFAULT '',
    agreement REAL DEFAULT 0.0,
    tension REAL DEFAULT 0.0,
    winner TEXT DEFAULT '',
    reason TEXT DEFAULT '',
    memory_delta TEXT DEFAULT '',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_seams_timestamp ON seams(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_seams_session ON seams(session_id);
CREATE TABLE IF NOT EXISTS sartre_event_ids (
    event_key TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    phase TEXT DEFAULT '',
    utility TEXT DEFAULT '',
    recorded_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sartre_event_ids_event ON sartre_event_ids(event_id);
`

// all conversation columns in schema order — keep in sync with scanConvRow.
const convCols = "id, timestamp, session_id, prompt, response, temperature, destiny, pain, tension, debt, velocity, alpha, quality, access_count"

func newSessionID() string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%08x", time.Now().UnixNano()&0xffffffff)
	}
	return hex.EncodeToString(b[:]) // 8 hex chars, like uuid4()[:8]
}

func nowSeconds() float64 { return float64(time.Now().UnixNano()) / 1e9 }

// NewLimphaClient opens (creating if needed) ~/.yent/limpha.db and starts a session.
func NewLimphaClient() (*LimphaClient, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("home dir: %w", err)
	}
	dir := filepath.Join(homeDir, ".yent")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("mkdir: %w", err)
	}
	return NewLimphaClientAt(filepath.Join(dir, "limpha.db"))
}

// NewLimphaClientAt opens the memory DB at an explicit path (tests use this).
func NewLimphaClientAt(dbPath string) (*LimphaClient, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	// One open connection serializes writes (SQLite's happy path) and mirrors the
	// Python single-connection model; WAL still lets that one connection read fast.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("wal: %w", err)
	}
	db.Exec("PRAGMA synchronous=NORMAL")
	if _, err := db.Exec(limphaSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("schema: %w", err)
	}
	c := &LimphaClient{db: db, dbPath: dbPath, sessionID: newSessionID(), connected: true}
	now := nowSeconds()
	if _, err := db.Exec(
		"INSERT OR IGNORE INTO sessions (session_id, started_at, last_active) VALUES (?, ?, ?)",
		c.sessionID, now, now); err != nil {
		db.Close()
		return nil, fmt.Errorf("session init: %w", err)
	}
	return c, nil
}

// computeQuality scores a turn — length and prompt/response ratio (memory.py parity).
func computeQuality(prompt, response string) float64 {
	r := strings.TrimSpace(response)
	if r == "" {
		return 0.0
	}
	respLen := float64(utf8.RuneCountInString(r))
	promptLen := float64(utf8.RuneCountInString(strings.TrimSpace(prompt)))
	if promptLen < 1 {
		promptLen = 1
	}
	var lengthScore float64
	switch {
	case respLen < 10:
		lengthScore = 0.1
	case respLen < 50:
		lengthScore = 0.3
	case respLen < 200:
		lengthScore = 0.5 + 0.3*(respLen-50)/150
	default:
		lengthScore = 0.8
	}
	ratio := respLen / promptLen
	var ratioScore float64
	switch {
	case ratio < 0.3:
		ratioScore = 0.2
	case ratio > 10:
		ratioScore = 0.6
	default:
		ratioScore = 0.7
	}
	q := 0.6*lengthScore + 0.4*ratioScore
	if q < 0 {
		q = 0
	}
	if q > 1 {
		q = 1
	}
	return q
}

// store inserts a turn and updates the session aggregate; returns the new id.
func (c *LimphaClient) store(prompt, response string, st LimphaState) (int64, error) {
	now := nowSeconds()
	quality := computeQuality(prompt, response)
	tx, err := c.db.Begin()
	if err != nil {
		c.recordMemoryFailure("conversation", err)
		return 0, err
	}
	res, err := tx.Exec(
		`INSERT INTO conversations
		 (timestamp, session_id, prompt, response,
		  temperature, destiny, pain, tension, debt, velocity, alpha, quality)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		now, c.sessionID, prompt, response,
		st.Temperature, st.Destiny, st.Pain, st.Tension, st.Debt, st.Velocity, st.Alpha, quality)
	if err != nil {
		tx.Rollback()
		c.recordMemoryFailure("conversation", err)
		return 0, err
	}
	id, _ := res.LastInsertId()
	if _, err := tx.Exec(
		`UPDATE sessions SET last_active = ?, turn_count = turn_count + 1,
		   avg_quality = (avg_quality * turn_count + ?) / (turn_count + 1)
		 WHERE session_id = ?`, now, quality, c.sessionID); err != nil {
		tx.Rollback()
		c.recordMemoryFailure("conversation", err)
		return 0, err
	}
	if err := tx.Commit(); err != nil {
		c.recordMemoryFailure("conversation", err)
		return 0, err
	}
	return id, nil
}

// StoreTurn records a conversation turn and returns its id. Called automatically
// after generations that need a durable limpha row for linked seams.
func (c *LimphaClient) StoreTurn(prompt, response string, state LimphaState) (int64, error) {
	if !c.connected {
		return 0, nil // silently skip if memory disabled
	}
	return c.store(prompt, response, state)
}

// Store records a conversation turn. Called automatically after each generation.
func (c *LimphaClient) Store(prompt, response string, state LimphaState) error {
	_, err := c.StoreTurn(prompt, response, state)
	return err
}

func limphaFTSTokens(query string) []string {
	var out []string
	seen := make(map[string]bool)
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		token := strings.ToLower(b.String())
		b.Reset()
		if !seen[token] {
			seen[token] = true
			out = append(out, token)
		}
	}
	for _, r := range query {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			b.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	return out
}

func limphaQuoteFTSTerm(term string) string {
	return `"` + strings.ReplaceAll(term, `"`, `""`) + `"`
}

func limphaNaturalFTSQuery(query string) string {
	tokens := limphaFTSTokens(query)
	if len(tokens) == 0 {
		return ""
	}
	parts := make([]string, len(tokens))
	for i, token := range tokens {
		parts[i] = limphaQuoteFTSTerm(token)
	}
	return strings.Join(parts, " OR ")
}

func limphaLooksExplicitFTSQuery(query string) bool {
	q := strings.TrimSpace(query)
	if q == "" {
		return false
	}
	if strings.Contains(q, ":") || strings.Contains(q, "*") || strings.Contains(q, "^") {
		return true
	}
	if strings.HasPrefix(q, `"`) && strings.HasSuffix(q, `"`) && len(q) > 1 {
		return true
	}
	upper := strings.ToUpper(q)
	for _, op := range []string{" OR ", " AND ", " NOT ", "NEAR("} {
		if strings.Contains(upper, op) {
			return true
		}
	}
	return false
}

func limphaFTSQueries(query string) []string {
	raw := strings.TrimSpace(query)
	natural := limphaNaturalFTSQuery(raw)
	if natural == "" {
		return nil
	}
	if limphaLooksExplicitFTSQuery(raw) {
		if natural != raw {
			return []string{raw, natural}
		}
		return []string{raw}
	}
	return []string{natural}
}

// Search runs an FTS5 full-text query, ranked by bm25 (lower = better).
func (c *LimphaClient) Search(query string, limit int) ([]map[string]interface{}, error) {
	if !c.connected || strings.TrimSpace(query) == "" {
		return nil, nil
	}
	queries := limphaFTSQueries(query)
	if len(queries) == 0 {
		return nil, nil
	}
	var lastErr error
	for i, matchQuery := range queries {
		out, err := c.searchFTS(matchQuery, limit)
		if err == nil {
			if i > 0 {
				atomic.AddInt64(&c.ftsFallbacks, 1)
			}
			return out, nil
		}
		lastErr = err
	}
	if lastErr != nil {
		atomic.AddInt64(&c.ftsErrors, 1)
	}
	return nil, nil // malformed FTS query -> empty, like the Python except
}

func (c *LimphaClient) searchFTS(matchQuery string, limit int) ([]map[string]interface{}, error) {
	rows, err := c.db.Query(
		`SELECT c.id, c.timestamp, c.session_id, c.prompt, c.response,
		        c.quality, c.access_count, c.temperature, c.destiny,
		        c.pain, c.tension, c.alpha, bm25(conversations_fts) AS rank
		 FROM conversations_fts fts
		 JOIN conversations c ON c.id = fts.rowid
		 WHERE conversations_fts MATCH ?
		 ORDER BY rank LIMIT ?`, matchQuery, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []map[string]interface{}
	for rows.Next() {
		var id, access int64
		var ts, quality, temp, dest, pain, tension, alpha, rank float64
		var sid, prompt, resp string
		if err := rows.Scan(&id, &ts, &sid, &prompt, &resp, &quality, &access,
			&temp, &dest, &pain, &tension, &alpha, &rank); err != nil {
			continue
		}
		out = append(out, map[string]interface{}{
			"id": id, "timestamp": ts, "session_id": sid, "prompt": prompt,
			"response": resp, "quality": quality, "access_count": access,
			"temperature": temp, "destiny": dest, "pain": pain, "tension": tension,
			"alpha": alpha, "rank": rank,
		})
	}
	if rows.Err() != nil {
		return nil, rows.Err()
	}
	return out, nil
}

// scanConvRow scans one full conversation row (convCols order) into a map.
func scanConvRow(rows *sql.Rows) (map[string]interface{}, error) {
	var id, velocity, access int64
	var ts, temp, dest, pain, tension, debt, alpha, quality float64
	var sid, prompt, resp string
	if err := rows.Scan(&id, &ts, &sid, &prompt, &resp, &temp, &dest, &pain,
		&tension, &debt, &velocity, &alpha, &quality, &access); err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"id": id, "timestamp": ts, "session_id": sid, "prompt": prompt,
		"response": resp, "temperature": temp, "destiny": dest, "pain": pain,
		"tension": tension, "debt": debt, "velocity": velocity, "alpha": alpha,
		"quality": quality, "access_count": access,
	}, nil
}

// Recall fetches one conversation, bumping its access_count.
func (c *LimphaClient) Recall(id int64) (map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	if _, err := c.db.Exec("UPDATE conversations SET access_count = access_count + 1 WHERE id = ?", id); err != nil {
		return nil, err
	}
	rows, err := c.db.Query("SELECT "+convCols+" FROM conversations WHERE id = ?", id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	if rows.Next() {
		return scanConvRow(rows)
	}
	return nil, rows.Err()
}

// Recent returns recent conversations in chronological order.
func (c *LimphaClient) Recent(limit int, sessionOnly bool) ([]map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	var rows *sql.Rows
	var err error
	if sessionOnly {
		rows, err = c.db.Query("SELECT "+convCols+" FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", c.sessionID, limit)
	} else {
		rows, err = c.db.Query("SELECT "+convCols+" FROM conversations ORDER BY timestamp DESC LIMIT ?", limit)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var desc []map[string]interface{}
	for rows.Next() {
		m, err := scanConvRow(rows)
		if err != nil {
			continue
		}
		desc = append(desc, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	// reverse to chronological
	for i, j := 0, len(desc)-1; i < j; i, j = i+1, j-1 {
		desc[i], desc[j] = desc[j], desc[i]
	}
	return desc, nil
}

// RecentByPromptPrefix returns matching conversations newest-first across all
// sessions. Prefix matching uses substr rather than LIKE so prompt text that
// contains '%' or '_' remains literal.
func (c *LimphaClient) RecentByPromptPrefix(prefix string, limit int) ([]map[string]interface{}, error) {
	if !c.connected || prefix == "" || limit <= 0 {
		return nil, nil
	}
	rows, err := c.db.Query(
		"SELECT "+convCols+" FROM conversations WHERE substr(prompt, 1, ?) = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
		utf8.RuneCountInString(prefix), prefix, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []map[string]interface{}
	for rows.Next() {
		m, err := scanConvRow(rows)
		if err != nil {
			continue
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// FindShardCandidates finds high-quality, frequently-recalled turns not yet sharded.
func (c *LimphaClient) FindShardCandidates(limit int) ([]map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	rows, err := c.db.Query(
		`SELECT `+prefixCols("c", convCols)+` FROM conversations c
		 LEFT JOIN shards s ON s.conversation_id = c.id
		 WHERE s.id IS NULL AND c.quality >= ? AND c.access_count >= ?
		 ORDER BY c.quality DESC, c.access_count DESC LIMIT ?`,
		shardMinQuality, shardMinAccess, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []map[string]interface{}
	for rows.Next() {
		m, err := scanConvRow(rows)
		if err != nil {
			continue
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// GraduateToShard records a conversation as graduated; returns 0 if already a shard.
func (c *LimphaClient) GraduateToShard(conversationID int64, shardPath, reason string, priority float64) (int64, error) {
	res, err := c.db.Exec(
		`INSERT INTO shards (conversation_id, shard_path, graduated_at, reason, priority)
		 VALUES (?, ?, ?, ?, ?)`, conversationID, shardPath, nowSeconds(), reason, priority)
	if err != nil {
		return 0, nil // UNIQUE violation -> already a shard
	}
	id, _ := res.LastInsertId()
	return id, nil
}

// GetTrainingQueue returns pending shards joined with their conversation text.
func (c *LimphaClient) GetTrainingQueue(limit int) ([]map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	rows, err := c.db.Query(
		`SELECT s.id, s.conversation_id, s.shard_path, s.graduated_at, s.reason,
		        s.priority, s.training_status, s.training_loss, c.prompt, c.response, c.quality
		 FROM shards s
		 JOIN conversations c ON c.id = s.conversation_id
		 WHERE s.training_status = 'pending'
		 ORDER BY s.priority DESC, s.graduated_at ASC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []map[string]interface{}
	for rows.Next() {
		var id, convID int64
		var shardPath, reason, status, prompt, resp string
		var graduatedAt, priority, quality float64
		var loss sql.NullFloat64
		if err := rows.Scan(&id, &convID, &shardPath, &graduatedAt, &reason,
			&priority, &status, &loss, &prompt, &resp, &quality); err != nil {
			continue
		}
		m := map[string]interface{}{
			"id": id, "conversation_id": convID, "shard_path": shardPath,
			"graduated_at": graduatedAt, "reason": reason, "priority": priority,
			"training_status": status, "prompt": prompt, "response": resp, "quality": quality,
		}
		if loss.Valid {
			m["training_loss"] = loss.Float64
		} else {
			m["training_loss"] = nil
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// MarkTrained marks a shard as trained, optionally recording its loss.
func (c *LimphaClient) MarkTrained(shardID int64, loss *float64) error {
	_, err := c.db.Exec("UPDATE shards SET training_status = 'trained', training_loss = ? WHERE id = ?", loss, shardID)
	return err
}

// StoreSeam records one body-divergence seam (a dual-pass turn where both Mistral
// bodies touched the prompt). Returns the new seam id. Skipped silently if memory
// is disabled. conversation_id 0 stores NULL (seam not tied to a persisted turn).
func (c *LimphaClient) StoreSeam(s Seam) (int64, error) {
	if !c.connected {
		return 0, nil // silently skip if memory disabled
	}
	var convID interface{}
	if s.ConversationID > 0 {
		convID = s.ConversationID
	}
	res, err := c.db.Exec(
		`INSERT INTO seams
		 (timestamp, session_id, conversation_id, body_a, body_b, prompt,
		  a_claim, b_claim, agreement, tension, winner, reason, memory_delta)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		nowSeconds(), c.sessionID, convID, s.BodyA, s.BodyB, s.Prompt,
		s.AClaim, s.BClaim, s.Agreement, s.Tension, s.Winner, s.Reason, s.MemoryDelta)
	if err != nil {
		c.recordMemoryFailure("seam", err)
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func (c *LimphaClient) recordMemoryFailure(kind string, err error) {
	if c == nil || err == nil {
		return
	}
	atomic.AddInt64(&c.memoryFailures, 1)
	switch kind {
	case "conversation":
		atomic.AddInt64(&c.memoryConversationFailures, 1)
	case "seam":
		atomic.AddInt64(&c.memorySeamFailures, 1)
	case "enqueue":
		atomic.AddInt64(&c.memoryEnqueueFailures, 1)
	}
	c.memoryMu.Lock()
	c.lastMemoryError = kind + ": " + err.Error()
	c.memoryMu.Unlock()
}

func (c *LimphaClient) memoryFailureSnapshot() (string, int64, int64, int64, int64) {
	if c == nil {
		return "", 0, 0, 0, 0
	}
	c.memoryMu.Lock()
	last := c.lastMemoryError
	c.memoryMu.Unlock()
	return last,
		atomic.LoadInt64(&c.memoryFailures),
		atomic.LoadInt64(&c.memoryConversationFailures),
		atomic.LoadInt64(&c.memorySeamFailures),
		atomic.LoadInt64(&c.memoryEnqueueFailures)
}

// RecentSeams returns recent seams (newest first) — the substrate supergamma reads
// to grow the metric field, and the operator reads to inspect the internal dialogue
// between the two bodies.
func (c *LimphaClient) RecentSeams(limit int) ([]map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	rows, err := c.db.Query(
		`SELECT id, timestamp, session_id, conversation_id, body_a, body_b, prompt,
		        a_claim, b_claim, agreement, tension, winner, reason, memory_delta
		 FROM seams ORDER BY timestamp DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []map[string]interface{}
	for rows.Next() {
		var id int64
		var convID sql.NullInt64
		var ts, agreement, tension float64
		var sid, bodyA, bodyB, prompt, aClaim, bClaim, winner, reason, memDelta string
		if err := rows.Scan(&id, &ts, &sid, &convID, &bodyA, &bodyB, &prompt,
			&aClaim, &bClaim, &agreement, &tension, &winner, &reason, &memDelta); err != nil {
			continue
		}
		m := map[string]interface{}{
			"id": id, "timestamp": ts, "session_id": sid, "body_a": bodyA, "body_b": bodyB,
			"prompt": prompt, "a_claim": aClaim, "b_claim": bClaim, "agreement": agreement,
			"tension": tension, "winner": winner, "reason": reason, "memory_delta": memDelta,
		}
		if convID.Valid {
			m["conversation_id"] = convID.Int64
		} else {
			m["conversation_id"] = nil
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// SearchByState finds past turns with a similar AMK state (cosine distance).
func (c *LimphaClient) SearchByState(state LimphaState, topK int, minQuality float64) ([]map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	q := stateVec(float64(state.Temperature), float64(state.Destiny), float64(state.Pain),
		float64(state.Tension), float64(state.Debt), float64(state.Alpha))
	rows, err := c.db.Query("SELECT "+convCols+" FROM conversations WHERE quality >= ? ORDER BY timestamp DESC LIMIT 1000", minQuality)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var scored []map[string]interface{}
	for rows.Next() {
		m, err := scanConvRow(rows)
		if err != nil {
			continue
		}
		rv := stateVec(m["temperature"].(float64), m["destiny"].(float64), m["pain"].(float64),
			m["tension"].(float64), m["debt"].(float64), m["alpha"].(float64))
		m["distance"] = cosineDistance(q, rv)
		scored = append(scored, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(scored, func(i, j int) bool {
		return scored[i]["distance"].(float64) < scored[j]["distance"].(float64)
	})
	if topK < len(scored) {
		scored = scored[:topK]
	}
	return scored, nil
}

// Stats reports memory counts and DB size.
func (c *LimphaClient) Stats() (map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	count := func(q string) (int64, error) {
		var n int64
		err := c.db.QueryRow(q).Scan(&n)
		return n, err
	}
	convCount, err := count("SELECT COUNT(*) FROM conversations")
	if err != nil {
		return nil, err
	}
	shardCount, err := count("SELECT COUNT(*) FROM shards")
	if err != nil {
		return nil, err
	}
	sessCount, err := count("SELECT COUNT(*) FROM sessions")
	if err != nil {
		return nil, err
	}
	pending, err := count("SELECT COUNT(*) FROM shards WHERE training_status = 'pending'")
	if err != nil {
		return nil, err
	}
	seamCount, err := count("SELECT COUNT(*) FROM seams")
	if err != nil {
		return nil, err
	}
	var dbSize int64
	if fi, err := os.Stat(c.dbPath); err == nil {
		dbSize = fi.Size()
	}
	c.asyncMu.Lock()
	asyncEnabled := c.async != nil
	asyncBacklog := 0
	if c.async != nil {
		asyncBacklog = len(c.async.queue)
	}
	c.asyncMu.Unlock()
	lastMemoryError, memoryFailures, conversationFailures, seamFailures, enqueueFailures := c.memoryFailureSnapshot()
	return map[string]interface{}{
		"total_conversations":          convCount,
		"total_shards":                 shardCount,
		"total_sessions":               sessCount,
		"total_seams":                  seamCount,
		"pending_training":             pending,
		"current_session":              c.sessionID,
		"db_path":                      c.dbPath,
		"db_size_bytes":                dbSize,
		"async_enabled":                asyncEnabled,
		"async_backlog":                asyncBacklog,
		"memory_write_failures":        memoryFailures,
		"memory_conversation_failures": conversationFailures,
		"memory_seam_failures":         seamFailures,
		"memory_enqueue_failures":      enqueueFailures,
		"last_memory_error":            lastMemoryError,
		"fts_query_errors":             atomic.LoadInt64(&c.ftsErrors),
		"fts_query_fallbacks":          atomic.LoadInt64(&c.ftsFallbacks),
	}, nil
}

// Close shuts the memory down.
func (c *LimphaClient) Close() {
	c.StopAsync()
	if c.db != nil {
		c.db.Close()
		c.db = nil
	}
	c.connected = false
}

func stateVec(temp, dest, pain, tension, debt, alpha float64) [6]float64 {
	return [6]float64{temp, dest, pain, tension, debt, alpha}
}

// cosineDistance is 1 - cosine similarity; 0 = identical (memory.py parity).
func cosineDistance(a, b [6]float64) float64 {
	var dot, na, nb float64
	for i := 0; i < 6; i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	na = math.Sqrt(na)
	nb = math.Sqrt(nb)
	if na == 0 || nb == 0 {
		return 1.0
	}
	return 1.0 - dot/(na*nb)
}

// prefixCols qualifies a comma list of columns with a table alias.
func prefixCols(alias, cols string) string {
	parts := strings.Split(cols, ", ")
	for i, p := range parts {
		parts[i] = alias + "." + p
	}
	return strings.Join(parts, ", ")
}
