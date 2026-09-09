package bodylease

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLeaseExcludesSecondOwnerInProcess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "body.lock")
	first, err := Acquire(context.Background(), path, Owner{Body: "nemo12", Backend: "doe"})
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer cancel()
	_, err = Acquire(ctx, path, Owner{Body: "gptoss20", Backend: "harmony"})
	if err == nil || !strings.Contains(err.Error(), "body=nemo12") || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second owner should see the current resident and timeout, got %v", err)
	}

	owner, err := CurrentOwner(path)
	if err != nil {
		t.Fatal(err)
	}
	if owner.Body != "nemo12" || owner.Backend != "doe" || owner.PID != os.Getpid() {
		t.Fatalf("owner receipt = %+v", owner)
	}
}

func TestLeaseTransfersAfterClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "body.lock")
	first, err := Acquire(context.Background(), path, Owner{Body: "nemo12"})
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := Acquire(context.Background(), path, Owner{Body: "gptoss20"})
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	owner, err := CurrentOwner(path)
	if err != nil || owner.Body != "gptoss20" {
		t.Fatalf("transferred owner = %+v, err=%v", owner, err)
	}
}

func TestLeaseExcludesOtherProcessAndCrashReleases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "body.lock")
	cmd := exec.Command(os.Args[0], "-test.run=TestLeaseHelperProcess")
	cmd.Env = append(os.Environ(), "YENT_LEASE_HELPER=1", "YENT_LEASE_HELPER_PATH="+path)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	killed := false
	defer func() {
		if !killed && cmd.Process != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil || strings.TrimSpace(line) != "READY" {
		t.Fatalf("helper did not acquire lease: line=%q err=%v", line, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	_, err = Acquire(ctx, path, Owner{Body: "gptoss20"})
	cancel()
	if err == nil || !strings.Contains(err.Error(), "body=helper-nemo") {
		t.Fatalf("parent crossed subprocess lease: %v", err)
	}

	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err == nil {
		t.Fatal("killed helper unexpectedly exited cleanly")
	}
	killed = true

	ctx, cancel = context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	lease, err := Acquire(ctx, path, Owner{Body: "gptoss20"})
	if err != nil {
		t.Fatalf("kernel did not release crashed holder: %v", err)
	}
	defer lease.Close()
}

func TestLeaseHelperProcess(t *testing.T) {
	if os.Getenv("YENT_LEASE_HELPER") != "1" {
		return
	}
	path := os.Getenv("YENT_LEASE_HELPER_PATH")
	lease, err := Acquire(context.Background(), path, Owner{Body: "helper-nemo", Backend: "test"})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	defer lease.Close()
	fmt.Println("READY")
	time.Sleep(time.Hour)
}
