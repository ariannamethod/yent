package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ariannamethod/yent/innerworld"
	yent "github.com/ariannamethod/yent/yent/go"
)

const (
	vagusAddrEnv      = "YENT_VAGUS_ADDR"
	vagusRootEnv      = "YENT_VAGUS_ROOT"
	vagusMaxBodyBytes = 64 << 10
	// doe_field's current chat wrapper is 2048 bytes. Leave room for the fast
	// primer and a bounded slice of private reflection instead of accepting a
	// large turn that the body would silently truncate.
	vagusMaxPromptBytes  = 1200
	vagusMaxInnerBytes   = 480
	vagusMaxHistoryBytes = 640
)

type vagusTurnResult struct {
	Answer string
	Body   string
	Trace  yent.RouteTrace
}

type vagusTurn struct {
	Prompt  string
	History []vagusChatMessage
}

type vagusTurner interface {
	Turn(context.Context, vagusTurn) (vagusTurnResult, error)
}

type vagusTurnFunc func(context.Context, vagusTurn) (vagusTurnResult, error)

func (f vagusTurnFunc) Turn(ctx context.Context, turn vagusTurn) (vagusTurnResult, error) {
	return f(ctx, turn)
}

// dockVagusTurner closes the live human-turn path inside the existing dock. The
// same DOEBody instance raises private circles and produces the outward answer;
// no second model process or shadow memory is created.
type dockVagusTurner struct {
	inner  *innerworld.InnerWorld
	router *yent.Router
	limpha *yent.LimphaClient
	state  func() yent.LimphaState
}

func (v dockVagusTurner) Turn(ctx context.Context, turn vagusTurn) (vagusTurnResult, error) {
	if v.inner == nil || v.router == nil || v.state == nil {
		return vagusTurnResult{}, errors.New("vagus live turn is not wired")
	}
	if err := ctx.Err(); err != nil {
		return vagusTurnResult{}, err
	}
	var outcome yent.Outcome
	_, answer, err := v.inner.ThinkAndAnswer(turn.Prompt, func(reflection innerworld.Reflection) (string, error) {
		state := v.state()
		persistReflection(v.limpha, "human_turn", "vagus", reflection, state)
		privateContext := vagusPrivateContext(reflection, turn.History)
		var routeErr error
		outcome, routeErr = v.router.RouteWithInnerContext(turn.Prompt, state, privateContext)
		return outcome.Answer, routeErr
	})
	if err != nil {
		return vagusTurnResult{}, err
	}
	return vagusTurnResult{Answer: answer, Body: outcome.Body, Trace: outcome.Trace}, nil
}

func vagusPrivateContext(reflection innerworld.Reflection, history []vagusChatMessage) string {
	parts := make([]string, 0, 2)
	if inner := vagusInnerContext(reflection); inner != "" {
		parts = append(parts, inner)
	}
	if dialogue := vagusDialogueContext(history); dialogue != "" {
		parts = append(parts, dialogue)
	}
	return strings.Join(parts, "\n")
}

func vagusInnerContext(reflection innerworld.Reflection) string {
	last := ""
	if n := len(reflection.Circles); n > 0 {
		last = strings.Join(strings.Fields(reflection.Circles[n-1].Text), " ")
	}
	if last == "" {
		return ""
	}
	last = compactVagusText(last, vagusMaxInnerBytes)
	return "Your private current inner reflection, not another speaker and not text to quote verbatim: " + last
}

func vagusDialogueContext(history []vagusChatMessage) string {
	if len(history) == 0 {
		return ""
	}
	var lines []string
	total := 0
	for i := len(history) - 1; i >= 0; i-- {
		role := strings.ToLower(strings.TrimSpace(history[i].Role))
		content := strings.Join(strings.Fields(history[i].Content), " ")
		if (role != "user" && role != "assistant") || content == "" {
			continue
		}
		prefix := "[" + role + "]: "
		available := vagusMaxHistoryBytes - total - len(prefix)
		if available <= 0 {
			break
		}
		originalLen := len(content)
		content = compactVagusText(content, available)
		line := prefix + content
		lines = append(lines, line)
		total += len(line)
		if len(content) < originalLen {
			break
		}
	}
	if len(lines) == 0 {
		return ""
	}
	return "Recent external dialogue from this local interface, newest first, for continuity only; the current human turn remains authoritative:\n" + strings.Join(lines, "\n")
}

func compactVagusText(value string, maxBytes int) string {
	value = strings.Join(strings.Fields(strings.ToValidUTF8(value, "")), " ")
	if maxBytes <= 0 {
		return ""
	}
	if len(value) <= maxBytes {
		return value
	}
	cut := maxBytes
	for cut > 0 && !utf8.RuneStart(value[cut]) {
		cut--
	}
	value = strings.TrimSpace(value[:cut])
	if word := strings.LastIndexByte(value, ' '); word >= cut/2 {
		value = strings.TrimSpace(value[:word])
	}
	return value
}

type vagusChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type vagusChatRequest struct {
	Messages    []vagusChatMessage `json:"messages"`
	Temperature *float64           `json:"temperature,omitempty"`
	MaxTokens   *int               `json:"max_tokens,omitempty"`
}

type vagusHandler struct {
	root   string
	turner vagusTurner
	turn   chan struct{}
}

func newVagusHandler(root string, turner vagusTurner) (http.Handler, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, errors.New("vagus interface root is required")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve vagus interface root: %w", err)
	}
	if turner == nil {
		return nil, errors.New("vagus turner is required")
	}
	for _, name := range []string{"yent.html", "worldmodel.html"} {
		info, statErr := os.Stat(filepath.Join(abs, name))
		if statErr != nil || !info.Mode().IsRegular() {
			return nil, fmt.Errorf("vagus interface root %q has no regular %s", abs, name)
		}
	}
	return &vagusHandler{root: abs, turner: turner, turn: make(chan struct{}, 1)}, nil
}

func (h *vagusHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !vagusRequestHostAllowed(r) || !vagusOriginAllowed(r) {
		http.Error(w, "vagus is loopback-only", http.StatusForbidden)
		return
	}
	if r.Method == http.MethodPost && (r.URL.Path == "/chat/completions" || r.URL.Path == "/v1/chat/completions") {
		h.serveChat(w, r)
		return
	}
	if r.Method == http.MethodGet || r.Method == http.MethodHead {
		h.serveStatic(w, r)
		return
	}
	w.Header().Set("Allow", "GET, HEAD, POST")
	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}

func (h *vagusHandler) serveStatic(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/yent", http.StatusTemporaryRedirect)
		return
	}
	if r.URL.Path == "/healthz" {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ready","organ":"vagus","delivery":"completed_answer","sampling":"aml_field","token_limit":"runtime_configured"}`+"\n")
		return
	}
	path, contentType, ok := vagusStaticPath(h.root, r.URL.Path)
	if !ok {
		http.NotFound(w, r)
		return
	}
	f, err := os.Open(path)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "no-store")
	http.ServeContent(w, r, info.Name(), info.ModTime(), f)
}

func vagusStaticPath(root, requestPath string) (string, string, bool) {
	switch requestPath {
	case "/yent", "/yent.html":
		return filepath.Join(root, "yent.html"), "text/html; charset=utf-8", true
	case "/worldmodel", "/worldmodel.html":
		return filepath.Join(root, "worldmodel.html"), "text/html; charset=utf-8", true
	}
	const prefix = "/worldmodel/"
	if !strings.HasPrefix(requestPath, prefix) {
		return "", "", false
	}
	name := strings.TrimPrefix(requestPath, prefix)
	if name == "" || filepath.Base(name) != name || filepath.Ext(name) != ".js" {
		return "", "", false
	}
	contentType := mime.TypeByExtension(".js")
	if contentType == "" {
		contentType = "application/javascript"
	}
	return filepath.Join(root, "DoE", "worldmodel", name), contentType, true
}

func (h *vagusHandler) serveChat(w http.ResponseWriter, r *http.Request) {
	if contentType := strings.ToLower(r.Header.Get("Content-Type")); !strings.HasPrefix(contentType, "application/json") {
		http.Error(w, "content type must be application/json", http.StatusUnsupportedMediaType)
		return
	}
	select {
	case h.turn <- struct{}{}:
		defer func() { <-h.turn }()
	default:
		http.Error(w, "Yent is already speaking", http.StatusConflict)
		return
	}

	var request vagusChatRequest
	reader := http.MaxBytesReader(w, r.Body, vagusMaxBodyBytes)
	decoder := json.NewDecoder(reader)
	if err := decoder.Decode(&request); err != nil {
		writeVagusJSONError(w, err, "invalid chat request")
		return
	}
	if err := requireJSONEOF(decoder); err != nil {
		writeVagusJSONError(w, err, "invalid trailing chat data")
		return
	}
	turn, err := currentVagusTurn(request.Messages)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, ": Yent is thinking\n\n")
	flushResponse(w)

	result, err := h.turner.Turn(r.Context(), turn)
	if err != nil {
		_ = writeVagusEvent(w, map[string]any{"error": compactVagusError(err)})
		flushResponse(w)
		return
	}
	if strings.TrimSpace(result.Answer) == "" {
		_ = writeVagusEvent(w, map[string]any{"error": "Yent returned no outward answer"})
		flushResponse(w)
		return
	}
	_ = writeVagusEvent(w, map[string]any{
		"token":       result.Answer,
		"body":        result.Body,
		"delivery":    "completed_answer",
		"sampling":    "aml_field",
		"temperature": result.Trace.State.Temperature,
		"debt":        result.Trace.State.Debt,
	})
	_ = writeVagusEvent(w, map[string]any{
		"done":     true,
		"body":     result.Body,
		"delivery": "completed_answer",
		"trace":    result.Trace,
	})
	flushResponse(w)
}

func writeVagusJSONError(w http.ResponseWriter, err error, fallback string) {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		http.Error(w, "chat request body is too large", http.StatusRequestEntityTooLarge)
		return
	}
	http.Error(w, fallback, http.StatusBadRequest)
}

func currentVagusTurn(messages []vagusChatMessage) (vagusTurn, error) {
	if len(messages) == 0 || len(messages) > 64 {
		return vagusTurn{}, errors.New("chat request requires 1..64 messages")
	}
	for _, message := range messages {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		if role != "user" && role != "assistant" {
			return vagusTurn{}, errors.New("chat messages may only be user or assistant turns")
		}
		if strings.TrimSpace(message.Content) == "" {
			return vagusTurn{}, errors.New("chat messages may not be empty")
		}
	}
	last := messages[len(messages)-1]
	if strings.ToLower(strings.TrimSpace(last.Role)) != "user" {
		return vagusTurn{}, errors.New("last chat message must be the current user turn")
	}
	prompt := strings.TrimSpace(last.Content)
	if len(prompt) > vagusMaxPromptBytes {
		return vagusTurn{}, errors.New("current user turn is too large")
	}
	history := append([]vagusChatMessage(nil), messages[:len(messages)-1]...)
	return vagusTurn{Prompt: prompt, History: history}, nil
}

func requireJSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err == io.EOF {
		return nil
	} else {
		return err
	}
}

func writeVagusEvent(w io.Writer, value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", payload)
	return err
}

func flushResponse(w http.ResponseWriter) {
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func compactVagusError(err error) string {
	text := strings.Join(strings.Fields(err.Error()), " ")
	if runes := []rune(text); len(runes) > 240 {
		text = string(runes[:240])
	}
	return text
}

func vagusRequestHostAllowed(r *http.Request) bool {
	if r == nil {
		return false
	}
	host := r.Host
	if parsed, _, err := net.SplitHostPort(host); err == nil {
		host = parsed
	}
	host = strings.Trim(host, "[]")
	ip := net.ParseIP(host)
	return strings.EqualFold(host, "localhost") || (ip != nil && ip.IsLoopback())
}

func vagusOriginAllowed(r *http.Request) bool {
	origin := strings.TrimSpace(r.Header.Get("Origin"))
	if origin == "" {
		return true
	}
	u, err := url.Parse(origin)
	return err == nil && (u.Scheme == "http" || u.Scheme == "https") && strings.EqualFold(u.Host, r.Host)
}

func validateVagusAddr(addr string) error {
	host, _, err := net.SplitHostPort(strings.TrimSpace(addr))
	if err != nil {
		return fmt.Errorf("invalid %s: %w", vagusAddrEnv, err)
	}
	host = strings.Trim(host, "[]")
	if strings.EqualFold(host, "localhost") {
		return nil
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("%s must bind a loopback address, got %q", vagusAddrEnv, host)
	}
	return nil
}

func startVagus(ctx context.Context, addr, root string, turner vagusTurner) (net.Addr, error) {
	if err := validateVagusAddr(addr); err != nil {
		return nil, err
	}
	handler, err := newVagusHandler(root, turner)
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("listen vagus: %w", err)
	}
	server := &http.Server{
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()
	go func() {
		if serveErr := server.Serve(listener); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			fmt.Fprintf(os.Stderr, "[vagus] serve: %v\n", serveErr)
		}
	}()
	return listener.Addr(), nil
}
