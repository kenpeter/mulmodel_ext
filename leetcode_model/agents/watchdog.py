"""
Watchdog — the safety net (generic).

Spawns a process, monitors heartbeat, auto-restarts on crash/hang.

Usage:
    python watchdog.py <script_to_run> [args...]
    python watchdog.py run_loop.py --project /path/to/project
"""

import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class Watchdog:
    def __init__(
        self,
        cmd,
        project_dir=None,
        heartbeat_timeout=900,  # 15 min — training takes ~10 min
        check_interval=60,
        max_restarts_per_hour=5,
    ):
        self.cmd = cmd  # ["python", "run_loop.py", ...]
        self.project = Path(project_dir) if project_dir else Path.cwd()
        self.heartbeat_file = self.project / "heartbeat.txt"
        self.log_file = self.project / "watchdog.log"
        self.alert_file = self.project / "to_human" / "ALERT.md"
        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval
        self.max_restarts = max_restarts_per_hour
        self.restart_times = []
        self.process = None

    def _log(self, msg):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def _heartbeat_age(self):
        try:
            ts = float(self.heartbeat_file.read_text().strip())
            return time.time() - ts
        except (FileNotFoundError, ValueError):
            return float("inf")

    def _recent_restarts(self):
        now = time.time()
        return sum(1 for t in self.restart_times if t > now - 3600)

    def _alert(self, msg):
        self.alert_file.parent.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.alert_file, "a") as f:
            f.write(f"\n## {ts} — WATCHDOG ALERT\n\n{msg}\n")

    def _start(self):
        self._log(f"Starting: {' '.join(self.cmd)}")
        self.process = subprocess.Popen(self.cmd, cwd=str(self.project))
        # Init heartbeat
        with open(self.heartbeat_file, "w") as f:
            f.write(str(time.time()))

    def _kill(self):
        if self.process is None or self.process.poll() is not None:
            return
        self._log(f"Killing PID {self.process.pid}")
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        except Exception as e:
            self._log(f"Kill error: {e}")
        # Wait for GPU memory to free
        self._log("Waiting 10s for GPU memory to free...")
        time.sleep(10)

    def _should_restart(self):
        restarts = self._recent_restarts()
        if restarts >= self.max_restarts:
            msg = f"Agent restarted {restarts}x in 1 hour. Manual intervention needed."
            self._log(msg)
            self._alert(msg)
            self._log("Waiting 5 minutes before retry...")
            time.sleep(300)
        self.restart_times.append(time.time())
        return True

    def run(self):
        self._log("Watchdog started")
        self._log(
            f"Timeout: {self.heartbeat_timeout}s, max restarts/hr: {self.max_restarts}"
        )
        self._start()

        try:
            while True:
                time.sleep(self.check_interval)

                # Process exited?
                if self.process.poll() is not None:
                    self._log(f"Process exited with code {self.process.returncode}")
                    self._log("Waiting 10s for GPU memory to free...")
                    time.sleep(10)
                    if self._should_restart():
                        self._start()
                    continue

                # Heartbeat stale?
                age = self._heartbeat_age()
                if age > self.heartbeat_timeout:
                    self._log(f"Heartbeat stale ({age:.0f}s)")
                    self._kill()
                    if self._should_restart():
                        self._start()

        except KeyboardInterrupt:
            self._log("Ctrl+C, shutting down")
            self._kill()
            self._log("Watchdog stopped")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watchdog.py <script> [args...]")
        print(
            "Example: python watchdog.py python run_loop.py --project /path/to/project"
        )
        sys.exit(1)

    # Find project dir from --project arg or use cwd
    project_dir = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--project" and i + 2 < len(sys.argv):
            project_dir = sys.argv[i + 2]
            break

    wd = Watchdog(sys.argv[1:], project_dir=project_dir)
    wd.run()
