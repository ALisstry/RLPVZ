class DDQNWorkerStatus:
    def __init__(self, worker_count: int):
        self.active_workers = set(range(worker_count))
        self.dead_workers = {}

    @property
    def has_active_workers(self) -> bool:
        return bool(self.active_workers)

    def check_processes(self, processes):
        for worker_id, process in enumerate(processes):
            if (
                worker_id in self.active_workers
                and not process.is_alive()
                and process.exitcode not in (None, 0)
            ):
                self.active_workers.discard(worker_id)
                self.dead_workers[worker_id] = (
                    f"进程异常退出，exitcode={process.exitcode}"
                )
                print(
                    f"\n[DDQN][Worker {worker_id}] 进程异常退出，已从训练中移除",
                    flush=True,
                )

    def handle_error(self, message):
        worker_id = message["worker_id"]
        self.active_workers.discard(worker_id)
        self.dead_workers[worker_id] = message["message"]
        print(
            f"\n[DDQN][Worker {worker_id}] 失败，已从训练中移除: {message['message']} "
            f"(pid={message['pid']}, port={message['port']})",
            flush=True,
        )
        self.raise_if_all_dead()

    def handle_warning(self, message):
        print(
            f"\n[DDQN][Worker {message['worker_id']}] {message['message']} "
            f"(pid={message['pid']}, port={message['port']})",
            flush=True,
        )

    def raise_if_all_dead(self):
        if self.active_workers or not self.dead_workers:
            return
        raise RuntimeError("所有 DDQN worker 都已失效: " + self._failure_summary())

    def _failure_summary(self) -> str:
        return "; ".join(
            f"worker {worker_id}: {reason}"
            for worker_id, reason in sorted(self.dead_workers.items())
        )
