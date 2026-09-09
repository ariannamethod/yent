// Package bodylease provides the machine-wide residency seam for Yent's model
// bodies. A lease is held for the whole lifetime of loaded weights, not merely
// for one generation, so independent runtime processes cannot quietly make two
// large bodies resident on the same unified-memory host.
package bodylease

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/unix"
)

const EnvPath = "YENT_BODY_LEASE_PATH"

const retryInterval = 25 * time.Millisecond

// Owner is the bounded, non-secret receipt left in the locked file while a body
// is resident. Detail should identify an artifact, never contain credentials.
type Owner struct {
	Body      string    `json:"body"`
	Backend   string    `json:"backend,omitempty"`
	Detail    string    `json:"detail,omitempty"`
	PID       int       `json:"pid"`
	Host      string    `json:"host,omitempty"`
	StartedAt time.Time `json:"started_at"`
}

// Lease owns both an OS file lock (for other processes) and a process-local slot
// (for independent bodies in one Go process). Close is idempotent.
type Lease struct {
	path string
	file *os.File
	slot *localSlot
	once sync.Once
	err  error
}

type localSlot struct{ token chan struct{} }

var (
	slotsMu sync.Mutex
	slots   = make(map[string]*localSlot)
)

// DefaultPath returns the shared lock path. Operators may move it with
// YENT_BODY_LEASE_PATH, but every body on one host must use the same path.
func DefaultPath() (string, error) {
	if path := strings.TrimSpace(os.Getenv(EnvPath)); path != "" {
		return filepath.Clean(path), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve body lease home: %w", err)
	}
	return filepath.Join(home, ".yent", "state", "body-residency.lock"), nil
}

// Acquire waits until no other Yent body owns path or ctx expires. Kernel file
// locking releases the cross-process claim automatically if the holder crashes.
func Acquire(ctx context.Context, path string, owner Owner) (*Lease, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if strings.TrimSpace(path) == "" {
		var err error
		path, err = DefaultPath()
		if err != nil {
			return nil, err
		}
	}
	path = filepath.Clean(path)
	if !filepath.IsAbs(path) {
		return nil, fmt.Errorf("body lease path must be absolute: %s", path)
	}
	slot := slotFor(path)
	select {
	case <-ctx.Done():
		return nil, busyError(path, ctx.Err())
	case <-slot.token:
	}
	releaseLocal := true
	defer func() {
		if releaseLocal {
			slot.token <- struct{}{}
		}
	}()

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("create body lease directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open body lease: %w", err)
	}
	locked := false
	defer func() {
		if !locked {
			_ = f.Close()
		}
	}()

	err = waitFlock(ctx, unix.Flock, int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB, retryInterval)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, busyError(path, err)
		}
		return nil, fmt.Errorf("lock body lease: %w", err)
	}
	locked = true

	owner = normalizeOwner(owner)
	if err := writeOwner(f, owner); err != nil {
		// Closing the owning descriptor in the deferred cleanup releases flock.
		locked = false
		return nil, fmt.Errorf("write body lease owner: %w", err)
	}
	releaseLocal = false
	return &Lease{path: path, file: f, slot: slot}, nil
}

// Path is the concrete machine-wide seam this lease owns.
func (l *Lease) Path() string {
	if l == nil {
		return ""
	}
	return l.path
}

// Close clears the owner receipt, closes the owning descriptor, and releases
// the in-process slot. Closing the descriptor is the kernel-authoritative flock
// release and happens only after the caller has stopped the body process.
func (l *Lease) Close() error {
	if l == nil {
		return nil
	}
	l.once.Do(func() {
		if l.file != nil {
			if err := l.file.Truncate(0); err != nil {
				l.err = errors.Join(l.err, err)
			}
			if err := l.file.Close(); err != nil {
				l.err = errors.Join(l.err, err)
			}
		}
		if l.slot != nil {
			l.slot.token <- struct{}{}
		}
	})
	return l.err
}

func waitFlock(ctx context.Context, call func(fd int, how int) error, fd int, how int, retry time.Duration) error {
	if retry <= 0 {
		retry = retryInterval
	}
	ticker := time.NewTicker(retry)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		err := call(fd, how)
		if err == nil {
			return nil
		}
		if !errors.Is(err, unix.EINTR) && !errors.Is(err, unix.EWOULDBLOCK) && !errors.Is(err, unix.EAGAIN) {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// CurrentOwner reads the best-effort receipt. A stale receipt may remain only
// after an unclean crash; Acquire remains authoritative because the kernel lock
// itself is released on process death.
func CurrentOwner(path string) (Owner, error) {
	if strings.TrimSpace(path) == "" {
		var err error
		path, err = DefaultPath()
		if err != nil {
			return Owner{}, err
		}
	}
	f, err := os.Open(path)
	if err != nil {
		return Owner{}, err
	}
	defer f.Close()
	var owner Owner
	if err := json.NewDecoder(io.LimitReader(f, 4096)).Decode(&owner); err != nil {
		return Owner{}, err
	}
	return owner, nil
}

func slotFor(path string) *localSlot {
	slotsMu.Lock()
	defer slotsMu.Unlock()
	if slot := slots[path]; slot != nil {
		return slot
	}
	slot := &localSlot{token: make(chan struct{}, 1)}
	slot.token <- struct{}{}
	slots[path] = slot
	return slot
}

func normalizeOwner(owner Owner) Owner {
	owner.Body = strings.TrimSpace(owner.Body)
	owner.Backend = strings.TrimSpace(owner.Backend)
	owner.Detail = strings.TrimSpace(owner.Detail)
	if owner.PID == 0 {
		owner.PID = os.Getpid()
	}
	if owner.Host == "" {
		owner.Host, _ = os.Hostname()
	}
	if owner.StartedAt.IsZero() {
		owner.StartedAt = time.Now().UTC()
	} else {
		owner.StartedAt = owner.StartedAt.UTC()
	}
	return owner
}

func writeOwner(f *os.File, owner Owner) error {
	if err := f.Truncate(0); err != nil {
		return err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(owner); err != nil {
		return err
	}
	return f.Sync()
}

func busyError(path string, cause error) error {
	owner, err := CurrentOwner(path)
	if err == nil && owner.Body != "" {
		return fmt.Errorf("body residency lease busy: body=%s backend=%s pid=%d host=%s since=%s: %w",
			owner.Body, owner.Backend, owner.PID, owner.Host, owner.StartedAt.Format(time.RFC3339), cause)
	}
	return fmt.Errorf("body residency lease busy at %s: %w", path, cause)
}
