"""fast_server와 같은 stdin JSON worker 풀. 로그 줄은 건너뛰고 JSON 한 줄만 결과로 받는다."""

from __future__ import annotations

import json
import subprocess
import sys
import threading


def _open(module_name: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", module_name, "--stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


class WorkerPool:
    def __init__(self, module_name: str, pool_size: int = 1):
        self.module_name = module_name
        self.workers = [_open(module_name) for _ in range(pool_size)]
        self.locks = [threading.Lock() for _ in range(pool_size)]
        self.index = 0
        self.pool_lock = threading.Lock()

    def process(self, payload: dict) -> dict:
        """학습 또는 판정 JSON을 worker에 보내고 결과 JSON 한 줄을 받는다."""
        with self.pool_lock:
            idx = self.index
            self.index = (self.index + 1) % len(self.workers)
        with self.locks[idx]:
            worker = self.workers[idx]
            if worker.poll() is not None:
                worker = _open(self.module_name)
                self.workers[idx] = worker
            assert worker.stdin is not None and worker.stdout is not None
            worker.stdin.write(json.dumps(payload) + "\n")
            worker.stdin.flush()
            while True:
                output = worker.stdout.readline().strip()
                if not output:
                    return {"ok": False, "error": "worker가 종료되었습니다."}
                if output.startswith("{") and output.endswith("}"):
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        continue

    def stop(self) -> None:
        for worker in self.workers:
            if worker.poll() is None:
                worker.terminate()
