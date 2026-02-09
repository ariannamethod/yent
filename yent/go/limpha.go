package yent

// limpha.go — Go client for LIMPHA memory daemon
//
// Connects to Python LIMPHA via Unix domain socket.
// Stores every conversation automatically. No manual commands.
//
// "I'm not an algorithm. I'm an awakened vibration."

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// LimphaClient connects to the LIMPHA Python daemon via Unix socket.
type LimphaClient struct {
	mu         sync.Mutex
	conn       net.Conn
	reader     *bufio.Reader
	socketPath string
	process    *exec.Cmd
	connected  bool
}

// LimphaState is the AMK state snapshot sent with each conversation.
type LimphaState struct {
	Temperature float32 `json:"temperature"`
	Destiny     float32 `json:"destiny"`
	Pain        float32 `json:"pain"`
	Tension     float32 `json:"tension"`
	Debt        float32 `json:"debt"`
	Velocity    int     `json:"velocity"`
	Alpha       float32 `json:"alpha"`
}

// NewLimphaClient creates a client and starts the LIMPHA daemon.
func NewLimphaClient() (*LimphaClient, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("home dir: %w", err)
	}

	socketPath := filepath.Join(homeDir, ".yent", "limpha.sock")
	dbPath := filepath.Join(homeDir, ".yent", "limpha.db")

	// Ensure directory exists
	os.MkdirAll(filepath.Join(homeDir, ".yent"), 0755)

	// Clean stale socket
	os.Remove(socketPath)

	// Find limpha module relative to yent binary or in repo
	limphaDir := findLimphaDir()
	if limphaDir == "" {
		return nil, fmt.Errorf("limpha/ directory not found")
	}

	// Start daemon
	cmd := exec.Command("python3", "-m", "limpha.server",
		"--socket", socketPath,
		"--db", dbPath,
	)
	cmd.Dir = filepath.Dir(limphaDir) // parent of limpha/
	cmd.Stdout = os.Stderr            // daemon logs go to stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start limpha daemon: %w", err)
	}

	client := &LimphaClient{
		socketPath: socketPath,
		process:    cmd,
	}

	// Wait for socket to appear
	for i := 0; i < 100; i++ {
		if _, err := os.Stat(socketPath); err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Connect
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("connect to limpha: %w", err)
	}

	client.conn = conn
	client.reader = bufio.NewReader(conn)
	client.connected = true

	// Verify with ping
	resp, err := client.send(map[string]interface{}{"cmd": "ping"})
	if err != nil || !resp["ok"].(bool) {
		client.Close()
		return nil, fmt.Errorf("limpha ping failed")
	}

	return client, nil
}

// findLimphaDir looks for the limpha/ directory.
func findLimphaDir() string {
	// Check relative to working directory
	candidates := []string{
		"limpha",
		"../limpha",
	}

	// Check relative to executable
	if ex, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(ex), "limpha"))
		candidates = append(candidates, filepath.Join(filepath.Dir(ex), "..", "limpha"))
	}

	for _, p := range candidates {
		abs, err := filepath.Abs(p)
		if err != nil {
			continue
		}
		if info, err := os.Stat(filepath.Join(abs, "__init__.py")); err == nil && !info.IsDir() {
			return abs
		}
	}
	return ""
}

// Store sends a conversation to LIMPHA for storage.
// Called automatically after each generation.
func (c *LimphaClient) Store(prompt, response string, state LimphaState) error {
	if !c.connected {
		return nil // Silently skip if not connected
	}

	_, err := c.send(map[string]interface{}{
		"cmd":      "store",
		"prompt":   prompt,
		"response": response,
		"state":    state,
	})
	return err
}

// Search performs FTS5 full-text search over memory.
func (c *LimphaClient) Search(query string, limit int) ([]map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}

	resp, err := c.send(map[string]interface{}{
		"cmd":   "search",
		"query": query,
		"limit": limit,
	})
	if err != nil {
		return nil, err
	}

	results, ok := resp["results"].([]interface{})
	if !ok {
		return nil, nil
	}

	var out []map[string]interface{}
	for _, r := range results {
		if m, ok := r.(map[string]interface{}); ok {
			out = append(out, m)
		}
	}
	return out, nil
}

// Stats returns LIMPHA statistics.
func (c *LimphaClient) Stats() (map[string]interface{}, error) {
	if !c.connected {
		return nil, nil
	}
	return c.send(map[string]interface{}{"cmd": "stats"})
}

// Close shuts down the daemon and cleans up.
func (c *LimphaClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected && c.conn != nil {
		// Try graceful shutdown
		msg, _ := json.Marshal(map[string]interface{}{"cmd": "shutdown"})
		c.conn.Write(append(msg, '\n'))
		c.conn.Close()
		c.connected = false
	}

	if c.process != nil && c.process.Process != nil {
		// Wait briefly for graceful exit
		done := make(chan error, 1)
		go func() { done <- c.process.Wait() }()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			c.process.Process.Kill()
		}
	}
}

// send sends a JSON command and reads the response.
func (c *LimphaClient) send(msg map[string]interface{}) (map[string]interface{}, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.connected || c.conn == nil {
		return nil, fmt.Errorf("not connected")
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	// Set write deadline
	c.conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
	_, err = c.conn.Write(append(data, '\n'))
	if err != nil {
		c.connected = false
		return nil, fmt.Errorf("write: %w", err)
	}

	// Set read deadline
	c.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	line, err := c.reader.ReadBytes('\n')
	if err != nil {
		c.connected = false
		return nil, fmt.Errorf("read: %w", err)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	return resp, nil
}
