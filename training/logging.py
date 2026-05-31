import logging
import os
import sys


_tee_stdout = None
_tee_stderr = None
_active_log_file = None


class _TeeStream:
    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def setup_logging(args, run_paths=None):
    from utils.logger import LogLevel, get_logger

    global _tee_stdout, _tee_stderr, _active_log_file

    if getattr(args, "log_file_path", None):
        log_file = args.log_file_path
    else:
        log_file = run_paths.log_file_path if run_paths is not None else None
        if log_file is None:
            raise ValueError("setup_logging 需要 run_paths 或 args.log_file_path")
        args.log_file_path = log_file
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    level_map = {
        0: LogLevel.WARNING,
        1: LogLevel.INFO,
        2: LogLevel.DEBUG,
    }
    logger = get_logger(
        level=level_map.get(getattr(args, "file_log_level", 1), LogLevel.INFO),
        file_path=log_file,
    )
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "file_log_level", 1) >= 2 else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )

    if _active_log_file != log_file:
        log_handle = open(log_file, "a", encoding="utf-8")
        _tee_stdout = _TeeStream(sys.__stdout__, log_handle)
        _tee_stderr = _TeeStream(sys.__stderr__, log_handle)
        sys.stdout = _tee_stdout
        sys.stderr = _tee_stderr
        _active_log_file = log_file

    print(f"\r\n[日志] 调试信息将保存到: {log_file}")
    return logger


def setup_worker_logging(args):
    if not getattr(args, "log_file_path", None):
        return

    from utils.logger import LogLevel, get_logger

    level_map = {
        0: LogLevel.WARNING,
        1: LogLevel.INFO,
        2: LogLevel.DEBUG,
    }
    get_logger(
        level=level_map.get(getattr(args, "file_log_level", 1), LogLevel.INFO),
        file_path=args.log_file_path,
    )
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "file_log_level", 1) >= 2 else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(args.log_file_path, encoding="utf-8")],
        force=True,
    )
