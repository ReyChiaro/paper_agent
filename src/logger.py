import io
import re
import sys
import logging
from contextlib import contextmanager, redirect_stderr
from rich.logging import RichHandler
from rich.console import Console
from rich.live import Live
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            markup=True,
        )
        formatter = logging.Formatter(fmt="%(message)s", datefmt="%H:%M:%S")
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)

    return logger


class TqdmRedirector(io.TextIOBase):

    def __init__(self, callback=None, prefix=""):
        super().__init__()

        self.buffer = ""
        self.callback = callback
        self.prefix = prefix

        self._tqdm_re = re.compile(
            r"(?P<task>.*?):\s+"
            r"(?P<percent>\d+)%\|(?P<bar>[^\|]+)\|\s+"
            r"(?P<done>\d+)/(?P<total>\d+)\s+"
            r"\[(?P<elapsed>[0-9:.]+)<(?P<eta>[0-9:.]+),\s+"
            r"(?P<speed>.+?)\]"
        )
        self._ansi = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

        self.progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(file=sys.__stderr__),
        )

        self._task_ids = {}

    def _clean_line(self, line: str) -> str:
        line = line.replace("\r", "")
        line = self._ansi.sub("", line)
        return line.strip()

    def _handle_line(self, line: str):
        line = self._clean_line(line)
        matches = self._tqdm_re.search(line)
        if not matches:
            return

        info = matches.groupdict()
        task = info["task"].strip()
        done = int(info["done"])
        total = int(info["total"])

        if task not in self._task_ids:
            self._task_ids[task] = self.progress.add_task(task, total=total)

        self.progress.update(self._task_ids[task], completed=done)

    def write(self, text: str):
        self.buffer += text

        while "\r" in self.buffer or "\n" in self.buffer:
            line_end = min(
                self.buffer.find("\r") if "\r" in self.buffer else len(self.buffer),
                self.buffer.find("\n") if "\n" in self.buffer else len(self.buffer),
            )
            line = self.buffer[:line_end]
            self.buffer = self.buffer[line_end + 1 :]
            self._handle_line(line)

    def flush(self):
        pass


@contextmanager
def beautified_tqdm():
    original_stderr = sys.stderr
    redirector = TqdmRedirector()

    redirector.progress.start()

    try:
        sys.stderr = redirector
        yield redirector.progress
    finally:
        redirector.progress.stop()
        sys.stderr = original_stderr
