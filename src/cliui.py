import time
import shutil

from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.text import Text
from pyfiglet import Figlet


PROJECT_NAME = "Paper Agent"


class CLIUI:

    def __init__(self):
        self.console = Console()
        self.header_ratio = 0.7

    def _build_layout(self, term_height: int):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=int(term_height * self.header_ratio)),
            Layout(name="main", ratio=1),
        )
        return layout

    def _update_layout(self, layout: Layout):
        current_width = shutil.get_terminal_size().columns
        current_height = shutil.get_terminal_size().lines
        rendered_name = (
            Figlet(
                font="small",
                width=current_width,
            ).renderText(PROJECT_NAME)
            if current_height > 12
            else Text(PROJECT_NAME, style="bold magenta")
        )
        layout["header"].update(
            Panel(
                Align.center(rendered_name, vertical="middle"),
                title="https://github.com/reychiaro/paper_agent",
                border_style="magenta",
                padding=(1, 2),
            )
        )

        _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        layout["main"].update(
            Panel(
                Align.center(
                    Text(f"Chat: {_now}", style="bold green"),
                )
            )
        )
        return layout

    def run(self):
        layout = self._build_layout(shutil.get_terminal_size().lines)
        try:
            with Live(console=self.console, refresh_per_second=1, screen=True) as live:
                while True:
                    prev_height = shutil.get_terminal_size().lines
                    time.sleep(0.5)
                    current_height = shutil.get_terminal_size().lines
                    if current_height != prev_height:
                        layout = self._build_layout(current_height)
                        live.update(self._update_layout(layout))
                    live.update(self._update_layout(layout))
        except KeyboardInterrupt:
            self.console.print("[bold red]Exiting...[/]")


if __name__ == "__main__":
    cli_ui = CLIUI()
    cli_ui.run()
