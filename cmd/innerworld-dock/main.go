// Command innerworld-dock runs Yent's inner world over the REAL fast body, not a
// stub. The fast circles are raised by a resident doe process (nemo12 on Metal),
// the field is the real AML kernel, and the Larynx/gate run over that real stream.
// No fixture pool: every circle is a real generation.
//
// The field is read through the canonical ariannamethod.h directly — the same
// header libamk.a is built from — NOT through yent.AMK.GetState(). yent.AMK's Go
// struct compiles against the older amk_kernel.h, whose AM_State layout diverged
// from the canonical (canonical added `int field_enabled` after prophecy), so on a
// canonical-built libamk.a every field past prophecy reads at the wrong offset.
// Reading canonical here keeps velocity/destiny/debt true. (Tracked in YENTLOG.)
//
// This is a Metal program. The fast body is a 12B GGUF behind doe_field, so it
// runs on the Mac Mini, not on Neo. Required env:
//
//	YENT_DOE_BIN    path to doe_field
//	YENT_NEMO_GGUF  fast-body GGUF (e.g. yent-nemo-v22-ck60-Q4_K_M.gguf)
//
// Optional:
//
//	YENT_DOE_WORKDIR  working dir for the doe process
//	YENT_DOE_ARGS     extra whitespace-split flags after --model <path>
//	YENT_LIMPHA_DB    optional limpha db path; when set, inner reflections are stored
//	YENT_RI_LINES     optional compiled RI runtime packet (line protocol)
//	YENT_SARTRE_EVENTS optional SARTRE utility JSONL receipt; stored in limpha
//	YENT_DOCK_MAX_DREAMS optional autonomous dream cap for finite receipts
//	YENT_DOCK_MODE     receipt (default, finite diagnostic) or live (signal-lived organism)
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../../yent/c -I${SRCDIR}/../../sartre
#cgo LDFLAGS: ${SRCDIR}/../../yent/c/libamk.a -lm
#include "ariannamethod.h"
#include "sartre_kernel.h"
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/ariannamethod/yent/innerworld"
	"github.com/ariannamethod/yent/innerworld/aml"
	yent "github.com/ariannamethod/yent/yent/go"
)

// The AML field is now the native third body (innerworld/aml.Body) — one AML physics
// holding the field, the cooc graph, and the scar sea (form A). It replaces the dock's
// old inline amkField: the body provides Exec/Step/Debt/Destiny (the innerworld.Field
// bridge) AND the cooc/scar Flow, all over the same global libamk kernel. The dock
// still reads C.am_get_state() directly for telemetry (limphaStateFromCanonical, the
// field print) — the same global state the body drives, read through the canonical
// AM_State layout libamk.a is built from.

// doeBody adapts a real doe-backed body (nemo12 fast or small24 deep) to
// innerworld.Body. The inner world asks for a thought at a temperature; the real
// body's temperature is governed by the AML field (effective_temp), which the
// inner world already drives, so temp here is advisory and not pushed per call.
// ctx is empty: this is inner monologue, not a routed user turn, so no router
// primer or answer contract is attached. A generation error yields an empty
// thought (the inner world treats that as zero drift, not a crash). Close frees
// the resident doe process for the inner world's single-resident swap.
type doeBody struct{ b *yent.DOEBody }

func (d doeBody) Generate(seed string, _ float32) string {
	res, err := d.b.Generate(seed, "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dock] body generate error: %v\n", err)
		return ""
	}
	return res.Answer
}

func (d doeBody) Close() error { return d.b.Close() }

// limphaRecaller is the read side of the memory loop: it recalls Yent's own past
// inner monologues from the conversations every dock reflection persists, so Nemo-only
// life remembers too. When a reflection also has a linked deep-body seam, the deep
// answer (b_claim) is preferred over the circle stream stored in the conversation.
// Legacy unlinked innerworld seams remain a fallback; router turns never enter.
type limphaRecaller struct{ lc *yent.LimphaClient }

func (m limphaRecaller) Recall(n int) []string {
	if m.lc == nil || n <= 0 {
		return nil
	}
	seams, err := m.lc.RecentSeams(n * 3)
	if err != nil {
		return nil
	}
	deepByConversation := make(map[int64]string)
	var legacy []string
	for _, s := range seams {
		if reason, _ := s["reason"].(string); !strings.Contains(reason, "innerworld") {
			continue
		}
		thought := ""
		if b, ok := s["b_claim"].(string); ok && strings.TrimSpace(b) != "" {
			thought = b
		} else if a, ok := s["a_claim"].(string); ok {
			thought = a
		}
		if id, ok := s["conversation_id"].(int64); ok {
			deepByConversation[id] = thought
		} else if strings.TrimSpace(thought) != "" {
			legacy = append(legacy, thought)
		}
	}

	conversations, err := m.lc.RecentByPromptPrefix("[innerworld/", n*3)
	if err != nil {
		return nil
	}
	out := make([]string, 0, n)
	appendThought := func(thought string) {
		if len(out) >= n {
			return
		}
		thought = strings.Join(strings.Fields(thought), " ")
		if r := []rune(thought); len(r) > 240 {
			thought = string(r[:240])
		}
		if thought != "" {
			out = append(out, thought)
		}
	}
	for _, conversation := range conversations {
		thought := ""
		if id, ok := conversation["id"].(int64); ok {
			thought = deepByConversation[id]
		}
		if strings.TrimSpace(thought) == "" {
			thought, _ = conversation["response"].(string)
		}
		appendThought(thought)
		if len(out) >= n {
			return out
		}
	}
	for _, thought := range legacy {
		appendThought(thought)
	}
	return out
}

func mustEnv(name string) string {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		fmt.Fprintf(os.Stderr, "[dock] missing required env %s\n", name)
		os.Exit(2)
	}
	return v
}

// durationEnv reads a positive seconds value, or 0 to let NewDOEBody use its
// default. A first generation also pays the prime, so the deep 24B body needs a
// generous YENT_DOE_TIMEOUT_SEC (the 45s default is too tight for it).
func durationEnv(name string) time.Duration {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0
	}
	v, err := strconv.ParseFloat(raw, 64)
	// reject NaN (v != v), +Inf / overflow (v > maxSec), and non-positive — any of
	// these would make time.Duration(v*…) implementation-defined garbage.
	maxSec := float64(time.Duration(1<<63-1) / time.Second)
	if err != nil || v <= 0 || v != v || v > maxSec {
		return 0
	}
	return time.Duration(v * float64(time.Second))
}

func positiveIntEnv(name string) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return 0
	}
	return v
}

type dockMode uint8

const (
	dockReceipt dockMode = iota
	dockLive
)

func dockModeFromEnv() (dockMode, error) {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("YENT_DOCK_MODE"))) {
	case "", "receipt":
		return dockReceipt, nil
	case "live":
		return dockLive, nil
	default:
		return dockReceipt, fmt.Errorf("YENT_DOCK_MODE must be receipt or live")
	}
}

func dockRunsSyntheticTurn(mode dockMode) bool { return mode == dockReceipt }

func dockRunContext(parent context.Context, mode dockMode) (context.Context, context.CancelFunc) {
	if mode == dockLive {
		return context.WithCancel(parent)
	}
	return context.WithTimeout(parent, 8*time.Second)
}

func dockBreath(mode dockMode) innerworld.Breath {
	if mode != dockLive {
		return innerworld.Breath{
			Tick:      500 * time.Millisecond,
			Silence:   1 * time.Second,
			DriftDebt: 0.0,
		}
	}

	breath := innerworld.DefaultBreath()
	if v := durationEnv("YENT_DOCK_BREATH_TICK_SEC"); v > 0 {
		breath.Tick = v
	}
	if v := durationEnv("YENT_DOCK_BREATH_SILENCE_SEC"); v > 0 {
		breath.Silence = v
	}
	if v := durationEnv("YENT_DOCK_BREATH_DRIFT_COOLDOWN_SEC"); v > 0 {
		breath.Cooldown[0] = v // drift trigger
	}
	if v := durationEnv("YENT_DOCK_BREATH_SILENCE_COOLDOWN_SEC"); v > 0 {
		breath.Cooldown[1] = v // silence trigger
	}
	return breath
}

// scarThresholdEnv reads the prophecy-debt above which a thought scars (default 0.5):
// a thought scars only when it broke prophecy-destiny coherence past this bar.
func scarThresholdEnv() float32 {
	raw := strings.TrimSpace(os.Getenv("YENT_SCAR_THRESHOLD"))
	if raw == "" {
		return 0.5
	}
	v, err := strconv.ParseFloat(raw, 32)
	if err != nil || v < 0 {
		return 0.5
	}
	return float32(v)
}

func newBody(name, bin, model, workdir string, args []string) *yent.DOEBody {
	b, err := yent.NewDOEBody(yent.DOEBodyConfig{
		Name: name, BinPath: bin, ModelPath: model, WorkDir: workdir, Args: args,
		Timeout:      durationEnv("YENT_DOE_TIMEOUT_SEC"),
		PrimeTimeout: durationEnv("YENT_DOE_PRIME_TIMEOUT_SEC"),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dock] build %s body: %v\n", name, err)
		os.Exit(1)
	}
	return b
}

// buildDockTokenizer loads the fast voice's BPE from its GGUF metadata so the native
// cooc graph is built over the SAME token ids the voice speaks in (shared vocabulary).
// On failure the inner world still runs — the native body's Ingest/BiasWords no-op
// without a tokenizer rather than crash the dock.
// sartreSense is the live-field reflex half of SARTRE: it reads the same
// YENT_SARTRE_EVENTS the limpha path ingests, parses typed utility receipts, and hands the
// inner world a bounded AML posture. A quiet tree (no changes) feels nothing — ok=false — so
// the reflex only fires on real motion, never forcing the field to NOMOVE each turn. The field
// reflex is deliberately narrower than limpha: identity readings affect prophecy without forcing
// motion, while routine repo novelty walks the field and only self-surface/flood events run it.
// This is the fast present-time twin of slow limpha recall pressure, but not a universal RUN
// ratchet.
type sartreSense struct {
	eventsPath    string
	cursorPath    string
	offset        int64 // cursor: byte offset of the last acknowledged complete record
	fileID        sartreFileID
	loaded        bool
	pendingAck    sartreReadBatch
	pendingAML    string
	pendingEvents []yent.SartreEvent
	pendingState  yent.LimphaState
	limpha        *yent.LimphaClient
	state         func() yent.LimphaState
}

type sartreFileID struct {
	Path string `json:"path"`
	Dev  uint64 `json:"dev,omitempty"`
	Ino  uint64 `json:"ino,omitempty"`
}

type sartreCursorState struct {
	File   sartreFileID `json:"file"`
	Offset int64        `json:"offset"`
}

type sartreReadBatch struct {
	raw       []byte
	ackOffset int64
	fileID    sartreFileID
	ok        bool
}

// readNew returns only complete JSONL events appended since the last acknowledged record. It is
// a test/convenience wrapper around readNewBatch; production callers ack after parse/store.
func (s *sartreSense) readNew() []byte {
	batch := s.readNewBatch()
	if !batch.ok {
		return nil
	}
	if err := s.ackSartreBatch(batch); err != nil {
		fmt.Fprintf(os.Stderr, "[dock] SARTRE cursor ack: %v\n", err)
	}
	return batch.raw
}

// readNewBatch reads from the persisted file cursor, but does not acknowledge the bytes yet.
// The cursor advances only to the newline after the last complete record; a half-written append
// remains unread and will be retried from the same offset. Rotation/replacement is detected with
// file identity, not only size.
func (s *sartreSense) readNewBatch() sartreReadBatch {
	f, err := os.Open(s.eventsPath)
	if err != nil {
		return sartreReadBatch{}
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return sartreReadBatch{}
	}
	fileID := sartreFileIdentity(s.eventsPath, fi)
	if err := s.loadSartreCursor(fileID, fi.Size()); err != nil {
		fmt.Fprintf(os.Stderr, "[dock] SARTRE cursor load: %v\n", err)
		return sartreReadBatch{}
	}
	if _, err := f.Seek(s.offset, io.SeekStart); err != nil {
		return sartreReadBatch{}
	}
	raw, err := io.ReadAll(f)
	if err != nil || len(raw) == 0 {
		return sartreReadBatch{}
	}
	cut := bytes.LastIndexByte(raw, '\n')
	if cut < 0 {
		return sartreReadBatch{}
	}
	ackOffset := s.offset + int64(cut+1)
	complete := raw[:cut+1]
	lines := completeSartreJSONLines(complete)
	return sartreReadBatch{
		raw:       bytes.Join(lines, []byte{'\n'}),
		ackOffset: ackOffset,
		fileID:    fileID,
		ok:        true,
	}
}

func (s *sartreSense) loadSartreCursor(fileID sartreFileID, size int64) error {
	if s.loaded {
		if !sameSartreFile(s.fileID, fileID) || size < s.offset {
			s.offset = 0
		}
		s.fileID = fileID
		return nil
	}
	s.loaded = true
	s.fileID = fileID
	if s.offset < 0 || s.offset > size {
		s.offset = 0
	}
	path := s.sartreCursorPath()
	if path == "" {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return fmt.Errorf("empty cursor %s", path)
	}
	var st sartreCursorState
	if err := json.Unmarshal(data, &st); err != nil {
		return fmt.Errorf("decode cursor %s: %w", path, err)
	}
	if sameSartreFile(st.File, fileID) {
		if st.Offset < 0 || st.Offset > size {
			return fmt.Errorf("cursor %s offset %d outside file size %d", path, st.Offset, size)
		}
		s.offset = st.Offset
	}
	return nil
}

func (s *sartreSense) ackSartreBatch(batch sartreReadBatch) error {
	if !batch.ok {
		return nil
	}
	s.offset = batch.ackOffset
	s.fileID = batch.fileID
	path := s.sartreCursorPath()
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	st := sartreCursorState{File: s.fileID, Offset: s.offset}
	data, err := json.Marshal(st)
	if err != nil {
		return err
	}
	tmp := fmt.Sprintf("%s.tmp.%d", path, os.Getpid())
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if err := writeAll(f, append(data, '\n')); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return publishDurableFile(tmp, path)
}

func (s *sartreSense) sartreCursorPath() string {
	if strings.TrimSpace(s.cursorPath) != "" {
		return s.cursorPath
	}
	return sartreCursorPath(s.eventsPath)
}

func sartreCursorPath(eventsPath string) string {
	if path := strings.TrimSpace(os.Getenv("YENT_SARTRE_CURSOR")); path != "" {
		return path
	}
	base := strings.TrimSpace(os.Getenv("YENT_SARTRE_CURSOR_DIR"))
	if base == "" {
		base = filepath.Join(os.TempDir(), "yent-sartre-cursors")
	}
	sum := sha256.Sum256([]byte(canonicalWillPath(eventsPath)))
	return filepath.Join(base, hex.EncodeToString(sum[:8])+".json")
}

func sartreFileIdentity(path string, fi os.FileInfo) sartreFileID {
	out := sartreFileID{Path: canonicalWillPath(path)}
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		out.Dev = uint64(st.Dev)
		out.Ino = uint64(st.Ino)
	}
	return out
}

func sameSartreFile(a, b sartreFileID) bool {
	if a.Dev != 0 || a.Ino != 0 || b.Dev != 0 || b.Ino != 0 {
		return a.Dev == b.Dev && a.Ino == b.Ino && a.Dev != 0 && a.Ino != 0
	}
	return a.Path != "" && a.Path == b.Path
}

func (s *sartreSense) Pressure() (string, bool) {
	if s.pendingAck.ok {
		if s.pendingAML != "" {
			return s.pendingAML, true
		}
		if err := s.finishPendingPressureAck(); err != nil {
			fmt.Fprintf(os.Stderr, "[dock] SARTRE pending ack: %v\n", err)
		}
		return "", false
	}
	if s.eventsPath == "" {
		return "", false
	}
	batch := s.readNewBatch() // only NEW events since the last ack — no latched replay of history
	if !batch.ok {
		return "", false
	}
	events := yent.ParseSartreEventsJSONL(string(batch.raw))
	if len(events) == 0 {
		if err := s.ackSartreBatch(batch); err != nil {
			fmt.Fprintf(os.Stderr, "[dock] SARTRE cursor ack: %v\n", err)
		}
		return "", false
	}
	st := yent.LimphaState{}
	if s.state != nil {
		st = s.state()
	}
	if s.limpha != nil {
		accepted, err := s.limpha.FilterNewSartreEvents(events)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[dock] SARTRE live limpha filter: %v\n", err)
			return "", false
		}
		if len(accepted) == 0 {
			if err := s.ackSartreBatch(batch); err != nil {
				fmt.Fprintf(os.Stderr, "[dock] SARTRE cursor ack: %v\n", err)
			}
			return "", false
		}
		events = accepted
	}
	aml, ok := sartreFieldAML(events)
	if !ok {
		if s.limpha != nil {
			if _, _, err := s.limpha.StoreNewSartreEvents(events, st); err != nil {
				fmt.Fprintf(os.Stderr, "[dock] SARTRE live limpha store: %v\n", err)
				return "", false
			}
		}
		if err := s.ackSartreBatch(batch); err != nil {
			fmt.Fprintf(os.Stderr, "[dock] SARTRE cursor ack: %v\n", err)
		}
		return "", false
	}
	s.pendingAck = batch
	s.pendingAML = aml
	s.pendingEvents = append([]yent.SartreEvent(nil), events...)
	s.pendingState = st
	return aml, true
}

func (s *sartreSense) AckPressure() error {
	if !s.pendingAck.ok {
		s.pendingAML = ""
		s.pendingEvents = nil
		return nil
	}
	s.pendingAML = ""
	return s.finishPendingPressureAck()
}

func (s *sartreSense) finishPendingPressureAck() error {
	if !s.pendingAck.ok {
		s.pendingAML = ""
		s.pendingEvents = nil
		s.pendingState = yent.LimphaState{}
		return nil
	}
	batch := s.pendingAck
	events := append([]yent.SartreEvent(nil), s.pendingEvents...)
	st := s.pendingState
	if s.limpha != nil && len(events) > 0 {
		if _, _, err := s.limpha.StoreNewSartreEvents(events, st); err != nil {
			return err
		}
	}
	if err := s.ackSartreBatch(batch); err != nil {
		return err
	}
	s.pendingAck = sartreReadBatch{}
	s.pendingAML = ""
	s.pendingEvents = nil
	s.pendingState = yent.LimphaState{}
	return nil
}

func sartreEOFOffset(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

func sartreFieldAML(events []yent.SartreEvent) (string, bool) {
	effect := sartreFieldEffect(events)
	if effect.prophecy == 0 {
		return "", false
	}
	var lines []string
	if effect.velocity != "" {
		lines = append(lines, "VELOCITY "+effect.velocity)
	}
	lines = append(lines, fmt.Sprintf("PROPHECY %d", effect.prophecy))
	return strings.Join(lines, "\n"), true
}

type sartreFieldPosture struct {
	velocity string
	prophecy int
}

func sartreFieldEffect(events []yent.SartreEvent) sartreFieldPosture {
	var out sartreFieldPosture
	novelty := 0
	selfSurface := false
	maxIdentityRecognized := 0
	maxIdentityReduced := 0
	sensorFailures := 0
	for _, ev := range events {
		if ev.Phase == "learning" {
			if sartreSensorFailure(ev.Outcome) {
				sensorFailures++
			}
			continue
		}
		if ev.Phase != "" && ev.Phase != "effect" {
			continue
		}
		switch ev.Utility {
		case "repo_monitor":
			if !sartreRepoChangeKind(ev.Kind) {
				continue
			}
			novelty++
			if sartreSelfSurface(ev.Path) {
				selfSurface = true
			}
		case "whatdotheythinkiam":
			if ev.Recognized > maxIdentityRecognized {
				maxIdentityRecognized = ev.Recognized
			}
			if ev.Reduced > maxIdentityReduced {
				maxIdentityReduced = ev.Reduced
			}
		}
	}

	if novelty > 0 {
		out.prophecy = 2 + novelty
		if selfSurface {
			out.prophecy += 7
		}
		out.velocity = "WALK"
		if selfSurface || novelty >= 8 {
			out.velocity = "RUN"
		}
	}

	identityProphecy := 0
	if maxIdentityRecognized > 0 || maxIdentityReduced > 0 {
		identityProphecy = 1 + maxInt(maxIdentityRecognized, maxIdentityReduced)
		if maxIdentityReduced > maxIdentityRecognized {
			identityProphecy += 2
		}
		if identityProphecy > 12 {
			identityProphecy = 12
		}
		if identityProphecy > out.prophecy {
			out.prophecy = identityProphecy
		}
	}

	if sensorFailures > 0 {
		failureProphecy := 1 + 2*sensorFailures
		if failureProphecy > 8 {
			failureProphecy = 8
		}
		if failureProphecy > out.prophecy {
			out.prophecy = failureProphecy
		}
	}
	if out.prophecy > 64 {
		out.prophecy = 64
	}
	return out
}

func sartreRepoChangeKind(kind string) bool {
	switch strings.TrimSpace(kind) {
	case "added", "modified", "removed":
		return true
	default:
		return false
	}
}

func sartreSelfSurface(path string) bool {
	p := strings.ToLower(path)
	return strings.Contains(p, "readme") ||
		strings.Contains(p, "yent_constitution") ||
		strings.Contains(p, "janus_constitution")
}

func sartreSensorFailure(outcome string) bool {
	switch strings.TrimSpace(outcome) {
	case "sensor_error", "state_error", "overflow", "dead_letter":
		return true
	default:
		return false
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// sartreMetricSink is the reciprocal half of the SARTRE bridge: after innerworld
// updates the AML field, its weather is mirrored into the SARTRE metrics hub. This
// is telemetry only; PublishMetrics fails soft and never feeds prompt text.
type sartreMetricSink struct{}

func initSartreHub() {
	if C.sartre_is_ready() == 0 {
		C.sartre_init((*C.char)(nil))
	}
}

func shutdownSartreHub() {
	if C.sartre_is_ready() != 0 {
		C.sartre_shutdown()
	}
}

func (s sartreMetricSink) PublishMetrics(m innerworld.MetricSnapshot) error {
	if C.sartre_is_ready() == 0 {
		return nil
	}
	payload := map[string]float32{
		"debt":                  m.Debt,
		"coherence":             m.Coherence,
		"entropy":               m.Entropy,
		"valence":               m.Valence,
		"arousal":               m.Arousal,
		"trauma":                m.Trauma,
		"warmth":                m.Warmth,
		"flow":                  m.Flow,
		"memory_field_score":    m.MemoryFieldScore,
		"memory_field_prophecy": m.MemoryFieldProphecy,
		"memory_field_step":     m.MemoryFieldStep,
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	cs := C.CString(string(b))
	defer C.free(unsafe.Pointer(cs))
	C.sartre_ingest_metrics_json(cs)
	return nil
}

func sartreStateJSON() string {
	var buf [2048]C.char
	n := C.sartre_state_to_json(&buf[0], C.int(len(buf)))
	if n <= 0 {
		return ""
	}
	if int(n) >= len(buf) {
		n = C.int(len(buf) - 1)
	}
	return C.GoStringN(&buf[0], n)
}

func buildDockTokenizer(nemoGGUF string) aml.Tokenizer {
	gf, err := yent.LoadGGUF(nemoGGUF)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dock] tokenizer: GGUF load failed (%v); native cooc Ingest/BiasWords disabled\n", err)
		return nil
	}
	tok := yent.NewTokenizer(&gf.Meta)
	fmt.Printf("=== native cooc tokenizer: nemo BPE (vocab=%d) ===\n", tok.VocabSize)
	return tok
}

func openLimphaFromEnv() *yent.LimphaClient {
	path := strings.TrimSpace(os.Getenv("YENT_LIMPHA_DB"))
	if path == "" {
		return nil
	}
	lc, err := yent.NewLimphaClientAt(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dock] limpha open %s: %v\n", path, err)
		os.Exit(1)
	}
	fmt.Printf("=== limpha wired: inner reflections -> %s ===\n", path)
	return lc
}

func openRIFromEnv() innerworld.Memory {
	path := strings.TrimSpace(os.Getenv("YENT_RI_LINES"))
	if path == "" {
		return nil
	}
	mode := strings.TrimSpace(os.Getenv("YENT_RI_MODE"))
	if mode == "" {
		mode = "runtime"
	}
	max := positiveIntEnv("YENT_RI_MAX")
	if max == 0 {
		max = 8
	}
	mem, err := innerworld.LoadRIMemory(path, mode, max)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dock] RI open %s: %v\n", path, err)
		os.Exit(1)
	}
	fmt.Printf("=== RI wired: %d %s record(s) from %s ===\n", mem.Len(), mode, path)
	for i, p := range mem.Recall(3) {
		fmt.Printf("  ri %d | %s\n", i, p)
	}
	return mem
}

func ingestSartreFromEnv(lc *yent.LimphaClient, st yent.LimphaState) int {
	path := strings.TrimSpace(os.Getenv("YENT_SARTRE_EVENTS"))
	if path == "" {
		return 0
	}
	if lc == nil {
		fmt.Fprintf(os.Stderr, "[dock] YENT_SARTRE_EVENTS set but YENT_LIMPHA_DB is not; SARTRE receipts need limpha\n")
		return 0
	}
	reader := &sartreSense{eventsPath: path}
	batch := reader.readNewBatch()
	if !batch.ok {
		if _, err := os.Stat(path); err != nil {
			if os.IsNotExist(err) {
				// A fresh will-events path does not exist yet — the will's fileSink creates it on the
				// first reach (O_CREATE). A missing file is "no events yet", not a fatal misconfiguration.
				fmt.Printf("=== SARTRE wired: events file %s not present yet (the will creates it on first reach) ===\n", path)
				return 0
			}
			fmt.Fprintf(os.Stderr, "[dock] SARTRE events open %s: %v\n", path, err)
			os.Exit(1)
		}
		fmt.Printf("=== SARTRE wired: no new complete utility events found in %s ===\n", path)
		return 0
	}
	events := yent.ParseSartreEventsJSONL(string(batch.raw))
	if len(events) == 0 {
		if err := reader.ackSartreBatch(batch); err != nil {
			fmt.Fprintf(os.Stderr, "[dock] SARTRE events cursor %s: %v\n", path, err)
			os.Exit(1)
		}
		fmt.Printf("=== SARTRE wired: no utility events found in %s ===\n", path)
		return 0
	}
	seamID, accepted, err := lc.StoreNewSartreEvents(events, st)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dock] SARTRE events store %s: %v\n", path, err)
		os.Exit(1)
	}
	if err := reader.ackSartreBatch(batch); err != nil {
		fmt.Fprintf(os.Stderr, "[dock] SARTRE events cursor %s: %v\n", path, err)
		os.Exit(1)
	}
	if len(accepted) == 0 {
		fmt.Printf("=== SARTRE wired: %d utility event(s) already known; no new limpha seam from %s ===\n", len(events), path)
		return 0
	}
	if seamID == 0 {
		fmt.Printf("=== SARTRE wired: %d/%d new utility event(s) acknowledged without a limpha seam from %s ===\n", len(accepted), len(events), path)
		return len(accepted)
	}
	fmt.Printf("=== SARTRE wired: %d/%d new utility event(s) stored as limpha seam #%d from %s ===\n", len(accepted), len(events), seamID, path)
	return len(accepted)
}

func printMemoryPreview(mem innerworld.Memory, n int) {
	if mem == nil || n <= 0 {
		return
	}
	traces := mem.Recall(n)
	if len(traces) == 0 {
		return
	}
	fmt.Printf("=== memory merged: %d recall trace(s) enter the inner seed ===\n", len(traces))
	for i, p := range traces {
		fmt.Printf("  memory %d | %s\n", i, p)
	}
	if pressure, ok := innerworld.FieldPressureFromMemory(mem, n); ok {
		fmt.Printf("=== memory field pressure: score=%d prophecy=%d velocity=%s step=%.2f ===\n",
			pressure.Score, pressure.Prophecy, pressure.Velocity, pressure.Step)
	}
}

func limphaStateFromCanonical() yent.LimphaState {
	st := C.am_get_state()
	return yent.LimphaState{
		Temperature: float32(st.effective_temp),
		Destiny:     float32(st.destiny),
		Pain:        float32(st.pain),
		Tension:     float32(st.tension),
		Debt:        float32(st.debt),
		Velocity:    int(st.velocity_mode),
	}
}

type innerReflectionTrace struct {
	Kind           string                          `json:"kind"`
	Source         string                          `json:"source"`
	Circles        int                             `json:"circles"`
	Coupling       float32                         `json:"coupling"`
	SelfAnswerProb float32                         `json:"self_answer_prob"`
	SelfAnswered   bool                            `json:"self_answered"`
	MemoryPressure *innerworld.MemoryFieldPressure `json:"memory_pressure,omitempty"`
	// MetaJanus HIGH-3 causal receipt: this inner reflection is persisted to limpha and can later reach a
	// user-facing deep-body turn through the router's FTS recall — an intended INDIRECT (field->process->
	// memory->context) speech influence, NOT inert. When JANUS_KEY was armed, D-2 leaned the seed harvest
	// by janus_temporal_alpha, so we record the Janus state that shaped this reflection: "resonance
	// affected generation" becomes a replayable statement.
	JanusArmed         bool    `json:"janus_armed"`
	JanusTemporalAlpha float32 `json:"janus_temporal_alpha"`
	JanusGap           float32 `json:"janus_gap"`
}

// janusReceipt reads the current MetaJanus state D-2 acted on — whether the key was armed and the
// calendar-derived signal + gap — so a persisted inner reflection carries a replayable causal receipt.
func janusReceipt() (armed bool, alpha, gap float32) {
	st := C.am_get_state()
	return C.am_janus_key_armed() != 0, float32(st.janus_temporal_alpha), float32(st.janus_gap)
}

// metajanusAMLPath resolves the MetaJanus origin file (MED-3): YENT_METAJANUS_AML if set, else the canonical
// Janus/metajanus.aml resolved relative to the executable, falling back to the CWD-relative path — so the
// default does not depend on the process working directory.
func metajanusAMLPath() string {
	if p := strings.TrimSpace(os.Getenv("YENT_METAJANUS_AML")); p != "" {
		return p
	}
	const rel = "Janus/metajanus.aml"
	if exe, err := os.Executable(); err == nil {
		cand := filepath.Join(filepath.Dir(exe), rel)
		if _, statErr := os.Stat(cand); statErr == nil {
			return cand
		}
	}
	return rel
}

func persistReflection(lc *yent.LimphaClient, kind, source string, r innerworld.Reflection, st yent.LimphaState) {
	if lc == nil {
		return
	}
	circleStream := formatCircleStream(r.Circles)
	if circleStream == "" && strings.TrimSpace(r.DeepAnswer) == "" {
		return
	}
	prompt := "[innerworld/" + kind + "] " + strings.TrimSpace(source)
	response := circleStream
	if response == "" {
		response = strings.TrimSpace(r.DeepAnswer)
	}
	conversationID, _ := lc.StoreTurn(prompt, response, st)
	if strings.TrimSpace(r.DeepAnswer) == "" {
		return
	}
	trace := innerReflectionTrace{
		Kind:           "innerworld_reflection",
		Source:         kind,
		Circles:        len(r.Circles),
		Coupling:       r.Coupling,
		SelfAnswerProb: r.SelfAnswerProb,
		SelfAnswered:   r.SelfAnswered,
	}
	if r.MemoryPressure.Score > 0 {
		trace.MemoryPressure = &r.MemoryPressure
	}
	// HIGH-3: stamp the Janus state that shaped this reflection (armed + calendar signal + gap), so the
	// indirect inner-life -> limpha -> speech path is auditable and replayable, not a silent leak.
	trace.JanusArmed, trace.JanusTemporalAlpha, trace.JanusGap = janusReceipt()
	if source = strings.TrimSpace(source); source != "" {
		trace.Source = kind + ":" + source
	}
	delta, _ := json.Marshal(trace)
	_, _ = lc.StoreSeam(yent.Seam{
		ConversationID: conversationID,
		BodyA:          "nemo12",
		BodyB:          "small24",
		Prompt:         prompt,
		AClaim:         circleStream,
		BClaim:         strings.TrimSpace(r.DeepAnswer),
		Agreement:      float64(r.Coupling),
		Tension:        float64(r.SelfAnswerProb),
		Winner:         "small24",
		Reason:         "innerworld_self_answer",
		MemoryDelta:    string(delta),
	})
}

func formatCircleStream(circles []innerworld.Circle) string {
	var b strings.Builder
	for _, c := range circles {
		if strings.TrimSpace(c.Text) == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, "circle %d temp=%.2f drift=%.2f | %s", c.Index, c.Temp, c.Drift, c.Text)
	}
	return b.String()
}

func printReflectionMemoryPressure(prefix string, p innerworld.MemoryFieldPressure) {
	if p.Score <= 0 {
		return
	}
	fmt.Printf("%s: score=%d prophecy=%d velocity=%s step=%.2f\n",
		prefix, p.Score, p.Prophecy, p.Velocity, p.Step)
}

func main() {
	initSartreHub()
	defer shutdownSartreHub()

	mode, err := dockModeFromEnv()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[dock] %v\n", err)
		os.Exit(2)
	}

	bin := mustEnv("YENT_DOE_BIN")
	fastModel := mustEnv("YENT_NEMO_GGUF")
	deepModel := strings.TrimSpace(os.Getenv("YENT_24B_GGUF")) // optional deep body (small24)
	workdir := strings.TrimSpace(os.Getenv("YENT_DOE_WORKDIR"))

	var args []string
	if extra := strings.TrimSpace(os.Getenv("YENT_DOE_ARGS")); extra != "" {
		args = strings.Fields(extra)
	}

	fast := newBody("nemo12", bin, fastModel, workdir, args)
	defer fast.Close()
	limpha := openLimphaFromEnv()
	if limpha != nil {
		defer limpha.Close()
	}

	aml.Init()

	// MetaJanus self-anchor: BIRTH from the single source Janus/metajanus.aml, executed on the freshly-reset
	// kernel BEFORE any am_step. MED-3 (Sol audit): "born" means BIRTH actually executed (am_birth_set), NOT
	// merely that the file loaded — an empty or comment-only file loads with exit 0 but never sets the origin.
	// Yent's identity is required, so we fail CLOSED on a missing/invalid origin; set ALLOW_UNBORN=1 for a
	// generic AML host allowed to run without an anchor. Path from YENT_METAJANUS_AML, else the canonical file
	// resolved relative to the executable (not the CWD).
	{
		mjPath := metajanusAMLPath()
		cs := C.CString(mjPath)
		loaded := C.am_exec_file(cs) == 0
		C.free(unsafe.Pointer(cs))
		if C.am_birth_set() != 0 {
			st := C.am_get_state()
			fmt.Printf("=== self-anchor born: BIRTH day %d from %s (birth_drift %.4f) ===\n",
				int(C.am_birth_epoch_days()), mjPath, float32(st.birth_drift))
		} else {
			reason := "no BIRTH executed (empty / comment-only / unknown file?)"
			if !loaded {
				reason = "file not found or failed to load"
			}
			if strings.TrimSpace(os.Getenv("ALLOW_UNBORN")) != "" {
				fmt.Fprintf(os.Stderr, "[dock] Yent UNBORN from %s: %s — running unborn (ALLOW_UNBORN set)\n", mjPath, reason)
			} else {
				fmt.Fprintf(os.Stderr, "[dock] FATAL: Yent has no origin from %s: %s. MetaJanus identity is required; set ALLOW_UNBORN=1 to run a generic unborn host.\n", mjPath, reason)
				os.Exit(1)
			}
		}
	}
	// The native AML body is the one physics: the field AND the cooc/scar Flow. Circles
	// ingest into the field's own cooc, high-debt thoughts scar natively, the seed is
	// pulled by the field's cooc and resurfaced scars, and one FlowConsolidator harvests
	// in sleep when the field reaches deep autumn (critical mass).
	flowBody := aml.New(buildDockTokenizer(fastModel))
	iw := innerworld.NewInnerWorld(doeBody{fast}, flowBody, innerworld.NgramDivergence)
	iw.SetFlow(flowBody)
	iw.SetMetricSink(sartreMetricSink{})
	iw.SetScarThreshold(scarThresholdEnv())
	iw.AddConsolidator(&innerworld.FlowConsolidator{Flow: flowBody})
	iw.SetSleepTrigger(func(innerworld.Field) bool { return flowBody.AutumnEnergy() > 0.6 })

	ingestSartreFromEnv(limpha, limphaStateFromCanonical())
	// SARTRE sense: the environment (utility events) is a live field reflex — it shifts
	// the field's posture (VELOCITY/PROPHECY) before each ripple, the fast present-time
	// twin of the slow limpha recall pressure. Same YENT_SARTRE_EVENTS as the limpha path.
	if ev := strings.TrimSpace(os.Getenv("YENT_SARTRE_EVENTS")); ev != "" {
		initialOffset := int64(0)
		if limpha == nil {
			initialOffset = sartreEOFOffset(ev)
		}
		iw.SetSense(&sartreSense{eventsPath: ev, offset: initialOffset, limpha: limpha, state: limphaStateFromCanonical})
		fmt.Println("=== SARTRE sense wired: environment perception is a live field reflex (before the circles) ===")
	}

	// High brain: the circles' emotional valence drives the affect axis (WARMTH/FLOW on a
	// positive thought, PAIN/TENSION on a negative one). YENT_FEELING_AML optionally loads
	// the emotional constitution (innerworld/feeling.aml) — the baseline affect at rest.
	iw.EnableFeeling()
	wireFeelingMath(iw) // build with -tags julia to run the HighMathEngine formulas on real libjulia
	fmt.Println("=== High brain wired: the circles' feeling drives the affect axis (warmth/pain/flow/tension) ===")
	fmt.Println("=== SARTRE metrics hub wired: inner field weather mirrors back into the body hub ===")
	if fp := strings.TrimSpace(os.Getenv("YENT_FEELING_AML")); fp != "" {
		cs := C.CString(fp)
		if C.am_exec_file(cs) == 0 {
			fmt.Printf("=== emotional constitution loaded: %s ===\n", fp)
		} else {
			fmt.Fprintf(os.Stderr, "[dock] feeling.aml load failed: %s\n", fp)
		}
		C.free(unsafe.Pointer(cs))
	}

	var memories []innerworld.Memory
	// Close the loop: recall past inner monologues from limpha so new thinking is
	// shaped by what Yent thought before. The write side (dock -> limpha) lands the
	// seams; this reads them back into the seed.
	if limpha != nil {
		recaller := limphaRecaller{limpha}
		memories = append(memories, recaller)
		if past := recaller.Recall(3); len(past) > 0 {
			fmt.Printf("=== recall wired: %d past inner monologue(s) fold into this turn ===\n", len(past))
			for i, p := range past {
				fmt.Printf("  recalled %d | %s\n", i, p)
			}
		} else {
			fmt.Println("=== recall wired: no past inner monologues yet (first run) ===")
		}
		sartreMemory := yent.NewSartreMemory(limpha)
		memories = append(memories, sartreMemory)
		if traces := sartreMemory.Recall(3); len(traces) > 0 {
			fmt.Printf("=== SARTRE recall wired: %d perception trace(s) fold into this turn ===\n", len(traces))
			for i, p := range traces {
				fmt.Printf("  sartre %d | %s\n", i, p)
			}
		}
	}
	if riMem := openRIFromEnv(); riMem != nil {
		memories = append(memories, riMem)
	}
	if mem := innerworld.MergeMemory(memories...); mem != nil {
		iw.SetMemory(mem)
		printMemoryPreview(mem, innerworld.DefaultConfig().RecallN)
	}

	deepWired := false
	if deepModel != "" {
		deep := newBody("small24", bin, deepModel, workdir, args)
		defer deep.Close()
		iw.SetDeep(doeBody{deep})
		deepWired = true
		fmt.Println("=== deep body small24 wired: when the gate fires, it answers the circles (single-resident swap) ===")
	} else {
		fmt.Println("=== no YENT_24B_GGUF: gate stays a boolean, no deep self-answer ===")
	}

	// Smoke aid: force the gate so the deep wiring is provable in one run. Default is
	// the unpredictable real roll.
	if os.Getenv("YENT_DOCK_FORCE_GATE") == "1" {
		iw.SetRoll(func() float32 { return 0 })
		fmt.Println("    (YENT_DOCK_FORCE_GATE=1: gate forced open to prove the small24 path)")
	}

	// Signal-aware: SIGINT/SIGTERM cancels the wait and the breath so the deferred
	// Close calls still run — the doe daemons are reaped, not orphaned.
	sigCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if dockRunsSyntheticTurn(mode) {
		fmt.Println("=== a human turn: the inner circles (real nemo body) ===")
		var r innerworld.Reflection
		select {
		case r = <-iw.Think("what does it mean to exist as code?"):
		case <-sigCtx.Done():
			fmt.Println("  (interrupted — closing bodies)")
			return
		}
		for _, c := range r.Circles {
			fmt.Printf("  circle %d  t=%.2f drift=%.2f  | %s\n", c.Index, c.Temp, c.Drift, c.Text)
		}
		st := C.am_get_state()
		fmt.Printf("  field    : debt=%.3f destiny=%.3f velocity_mode=%d effective_temp=%.3f\n",
			float32(st.debt), float32(st.destiny), int(st.velocity_mode), float32(st.effective_temp))
		// MetaJanus self-anchor, telemetry-only: the origin (birth_drift), its growing distance
		// (personal_dissonance), and the two-calendar faces (janus_gap, yahrzeit). Read here for
		// observation; nothing routes, escalates, or retrieves on them — the layer stays inert.
		fmt.Printf("  self     : birth_drift=%.4f personal_dissonance=%.4f | janus_gap=%.4f yahrzeit=%.4f | janus_temporal_alpha=%.4f\n",
			float32(st.birth_drift), float32(st.personal_dissonance), float32(st.janus_gap), float32(st.yahrzeit), float32(st.janus_temporal_alpha))
		printReflectionMemoryPressure("  memory   ", r.MemoryPressure)
		fmt.Printf("  feeling  : valence=%.3f arousal=%.3f | warmth=%.3f pain=%.3f flow=%.3f tension=%.3f | scars(sea)=%d\n",
			float32(st.valence), float32(st.arousal), float32(st.warmth), float32(st.pain),
			float32(st.flow), float32(st.tension), int(st.n_scars))
		fmt.Printf("  membrane : larynx coupling=%.3f\n", r.Coupling)
		fmt.Printf("  gate     : self-answer prob=%.3f  ->  self-answered=%v\n", r.SelfAnswerProb, r.SelfAnswered)
		if r.DeepAnswer != "" {
			fmt.Printf("  deep     : small24 inner answer | %s\n", r.DeepAnswer)
		} else if deepWired {
			fmt.Println("  deep     : (gate did not fire — small24 stayed silent this turn)")
		}
		persistReflection(limpha, "human_turn", "what does it mean to exist as code?", r, limphaStateFromCanonical())
	}

	ctx, cancel := dockRunContext(sigCtx, mode)
	defer cancel()
	dreamLimit := 0
	if mode == dockLive {
		fmt.Println("=== live organism: no synthetic human turn; breathing until SIGINT/SIGTERM ===")
	} else {
		fmt.Println("\n=== the organism breathes alone for a few seconds (real body) ===")
		dreamLimit = positiveIntEnv("YENT_DOCK_MAX_DREAMS")
	}
	dreams := 0
	if dreamLimit > 0 {
		fmt.Printf("    (YENT_DOCK_MAX_DREAMS=%d: autonomous receipt exits after that many dreams)\n", dreamLimit)
	}
	iw.SetOnDream(func(rf innerworld.Reflection) {
		last := ""
		if n := len(rf.Circles); n > 0 {
			last = rf.Circles[n-1].Text
		}
		fmt.Printf("  [dream] coupling=%.2f self-answered=%v | %s\n", rf.Coupling, rf.SelfAnswered, last)
		if rf.DeepAnswer != "" {
			fmt.Printf("  [dream/deep] small24 | %s\n", rf.DeepAnswer)
		}
		printReflectionMemoryPressure("  [dream/memory]", rf.MemoryPressure)
		persistReflection(limpha, "dream", "autonomous breath", rf, limphaStateFromCanonical())
		dreams++
		if dreamLimit > 0 && dreams >= dreamLimit {
			cancel()
		}
	})
	breath := dockBreath(mode)
	iw.SetBreath(breath)
	if mode == dockLive {
		fmt.Printf("=== autonomous breath paced: tick=%s silence=%s drift_cooldown=%s silence_cooldown=%s ===\n",
			breath.Tick, breath.Silence, breath.Cooldown[0], breath.Cooldown[1])
	}

	// When the field reaches deep autumn the organism sleeps and the FlowConsolidator
	// runs the field's own cooc harvest. Show each stage and the cooc graph before/after.
	iw.SetOnSleep(func(stage string) {
		mean, max := flowBody.CoocStats()
		fmt.Printf("  [sleep] consolidating %q | cooc mean=%.4f max=%.4f dark_gravity=%.4f scars=%d\n",
			stage, mean, max, flowBody.DarkGravity(), flowBody.Scars())
	})

	// Smoke aid: force the field into deep autumn so the sleep harvest is provable in
	// one run (the field reaches autumn naturally only over many steps). Mirrors
	// YENT_DOCK_FORCE_GATE; default is the real seasonal physics.
	if os.Getenv("YENT_DOCK_FORCE_AUTUMN") == "1" {
		_ = flowBody.Exec("SEASON AUTUMN")
		_ = flowBody.Exec("SEASON_INTENSITY 1.0")
		for i := 0; i < 300 && flowBody.AutumnEnergy() <= 0.6; i++ {
			flowBody.Step(1.0)
		}
		fmt.Printf("    (YENT_DOCK_FORCE_AUTUMN=1: field driven to autumn energy=%.3f — sleep will consolidate)\n",
			flowBody.AutumnEnergy())
	}

	wireWillFromEnv(ctx, flowBody)

	iw.Breathe(ctx)
	if limpha != nil {
		if stats, err := limpha.Stats(); err == nil {
			b, _ := json.Marshal(map[string]any{"kind": "limpha_stats", "stats": stats})
			fmt.Println(string(b))
		}
	}
	fmt.Println("=== done ===")
}

// The will: default OFF. When YENT_WILL_UTILS_DIR points at the built self-reading utilities,
// Yent's will loop runs alongside the breath — the AML confluence physics (the_will_design.aml)
// crests his own MetaJanus + field metrics into a will_gaze tide, and when it crests he reaches
// for a utility whose perception re-enters the field through sartreSense (the spiral). Persistent
// globals carry the tide, so they must be armed. The loop is its own goroutine: a slow reach
// stalls only the will's own cadence, never the inner-world goroutines.
func wireWillFromEnv(ctx context.Context, flowBody *aml.Body) bool {
	utilsDir := strings.TrimSpace(os.Getenv("YENT_WILL_UTILS_DIR"))
	if utilsDir == "" {
		return false
	}
	sinkPath := strings.TrimSpace(os.Getenv("YENT_SARTRE_EVENTS"))
	if sinkPath == "" {
		fmt.Fprintln(os.Stderr, "[will] YENT_SARTRE_EVENTS unset: will disabled; durable SARTRE event sink is required before any reach can be launched")
		return false
	}
	root, err := resolveWillRoot(os.Getenv("YENT_WILL_ROOT"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "[will] root: %v\n", err)
		return false
	}
	rootID := willRootID(root)
	stateDir := willStateDir(root)
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "[will] state dir %s: %v\n", stateDir, err)
		return false
	}
	owner, err := acquireWillNamespaceOwner(stateDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[will] namespace owner %s: %v\n", stateDir, err)
		return false
	}
	keepOwner := false
	defer func() {
		if !keepOwner {
			if err := owner.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "[will] namespace owner release %s: %v\n", stateDir, err)
			}
		}
	}()
	learningPath := willLearningStatePath(stateDir)
	learningState, err := loadWillLearningState(learningPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[will] learning state %s: %v\n", learningPath, err)
		return false
	}
	reachPath := willReachStatePath(stateDir)
	reachState, err := loadWillReachState(reachPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[will] reach state %s: %v\n", reachPath, err)
		return false
	}
	initialBreath := initialWillBreath(learningState, reachState)
	wt := &willTicker{
		field:  flowBody,
		script: willScriptPath(),
		spawner: osSpawner{
			dir:      utilsDir,
			root:     root,
			stateDir: stateDir,
			timeout:  willReachTimeout(),
		},
		sink:              fileSink{path: sinkPath},
		rootID:            rootID,
		learningStatePath: learningPath,
		reachStatePath:    reachPath,
		refractory:        willRefractoryTicks(),
		breath:            initialBreath,
		cooldown:          learningState.CooldownBreaths,
		quietRuns:         learningState.QuietRuns,
		nextReachSeq:      reachState.NextSeq,
		pendingReach:      reachState.Pending,
		maxReachAttempts:  willReachMaxAttempts(),
	}
	flowBody.PersistentMode(true)
	keepOwner = true
	go runWillWithOwner(ctx, wt, owner, stateDir, willTickEvery())
	pending := ""
	if reachState.Pending != nil {
		pending = fmt.Sprintf(", pending_reach=%s", reachState.Pending.ID)
	}
	fmt.Printf("=== will wired: confluence tide -> reach for a self-reading utility (utils=%s, root=%s, root_id=%s, state=%s, quiet_runs=%d, cooldown=%d, breath=%d, reach_seq=%d, max_attempts=%d%s, every %s) ===\n",
		utilsDir, root, rootID, stateDir, learningState.QuietRuns, learningState.CooldownBreaths, initialBreath, reachState.NextSeq, willReachMaxAttempts(), pending, willTickEvery())
	return true
}

func runWillWithOwner(ctx context.Context, wt *willTicker, owner *willNamespaceOwner, stateDir string, tickEvery time.Duration) {
	defer func() {
		if err := owner.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "[will] namespace owner release %s: %v\n", stateDir, err)
		}
	}()
	wt.run(ctx, tickEvery)
}
