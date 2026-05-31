import multiprocessing as mp


class AsyncWorkerPool:
    """
    Minimal process-pool infrastructure for async environment workers.

    Algorithm modules own the worker target and payload semantics; this class only
    owns process lifecycle and shared stop signaling.
    """

    def __init__(self, instances):
        self.instances = instances
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.workers = []

    def make_queue(self, maxsize=0):
        return self.ctx.Queue(maxsize=maxsize)

    def make_per_worker_queues(self, maxsize=0):
        return [self.make_queue(maxsize=maxsize) for _ in self.instances]

    def start_workers(self, target, build_args, label):
        for worker_id, instance in enumerate(self.instances):
            process = self.ctx.Process(
                target=target,
                args=build_args(worker_id, instance),
            )
            process.start()
            self.workers.append(process)
            print(
                f"[{label}] Worker {worker_id} 已启动: "
                f"pid={instance['pid']} port={instance['port']}"
            )

    def request_stop(self):
        self.stop_event.set()

    def stop(self):
        self.request_stop()
        for process in self.workers:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
