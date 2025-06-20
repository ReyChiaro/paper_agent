import io
import re
import sys
import time
import logging
import threading

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.console import Console


console = Console()


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

import io
import re
import sys
import time
import threading
from contextlib import redirect_stderr

from rich.progress import (
    Progress, BarColumn, TextColumn,
    TimeRemainingColumn, TimeElapsedColumn
)

# === Step 1: 定义正则解析 tqdm 风格的输出 ===
tqdm_re = re.compile(
    r'(?P<task>.*?):\s+'
    r'(?P<percent>\d+)%\|(?P<bar>[^\|]+)\|\s+'
    r'(?P<done>\d+)/(?P<total>\d+)\s+'
    r'\[(?P<elapsed>[0-9:.]+)<(?P<eta>[0-9:.]+),\s+'
    r'(?P<speed>.+?)\]'
)

# === Step 2: 设置 rich 进度条 ===
progress = Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    transient=True  # 任务完成后自动清除
)

# 保存每个 task 的 ID
task_id_map = {}

# === Step 3: 处理每一行 stderr ===
def handle_line(line):
    match = tqdm_re.search(line)
    if not match:
        return

    info = match.groupdict()
    task = info['task'].strip()
    done = int(info['done'])
    total = int(info['total'])

    # 注册任务
    if task not in task_id_map:
        task_id_map[task] = progress.add_task(task, total=total)

    # 更新任务进度
    progress.update(task_id_map[task], completed=done)

# === Step 4: 包装 stderr 并实时读取 ===
class StderrInterceptor(io.StringIO):
    def __init__(self):
        super().__init__()
        self.buffer = ""

    def write(self, s):
        self.buffer += s
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            for line in lines[:-1]:
                handle_line(line)
            self.buffer = lines[-1]

# # === Step 5: 示例黑盒函数 ===
# def black_box_function():
#     from time import sleep
#     from tqdm import tqdm

#     for _ in tqdm(range(5), desc="Recognizing layout"):
#         sleep(0.5)
#     for _ in tqdm(range(7), desc="Running OCR Detection"):
#         sleep(0.3)

# # === Step 6: 启动整体包装器 ===
# def main():
#     stderr_interceptor = StderrInterceptor()
#     with progress:
#         with redirect_stderr(stderr_interceptor):
#             black_box_function()

# if __name__ == "__main__":
#     main()
