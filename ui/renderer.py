from typing import Any, Tuple
from rich.console import Console
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from pathlib import Path
from utils.paths import display_path_rel_to_cwd
from rich import box

# centralized console for the UI rendering
AGENT_THEME = Theme(
    {
        # general
        "info":"cyan",
        "warning":"yellow",
        "error":"bright_red bold",
        "success":"green",
        "dim":"dim",
        "muted":"grey50",
        "border":"grey37",
        "highlight":"bold cyan",
        # Roles
        "user":"bright_blue bold",
        "assistant":"bright_white bold",
        # Tools
        "tool":"bright_magenta bold",
        "tool.read" : "cyan",
        "tool.write" : "yellow",
        "tool.shell" : "magenta",
        "tool.network" : "bright_blue",
        "tool.memory" : "green",
        "tool.mcp" : "bright_cyan",
        # Code / blocks
        "code":"white",
    }
)

_console: Console | None = None

def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)
    
    return _console

class TUI:
    def __init__(
            self,
            console: Console | None = None,
            )->None:
        self.console = console or get_console()
        self._assistant_stream_open = False
        self._tool_args_by_call_id: dict[str, dict[str,Any]] = {}
        self.cwd = Path.cwd()
    
    def begin_assistant(self)-> None:
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant"), style="border"))
        self._assistant_stream_open = True
    
    def end_assistant(self)-> None:
        if self._assistant_stream_open:
            self.console.print()  # ensure we end with a newline
        self._assistant_stream_open = False

    def stream_assistant_delta(self, content: str)-> None:
        self.console.print(content, end="", markup=False)
    
    def ordered_arguments(self, tool_name: str, args: dict[str,Any])->list[Tuple]:
        _PREFERED_ORDER = {
            'read_file' : ['path', 'offset', 'limit'],
        }
        prefered = _PREFERED_ORDER.get(tool_name, [])
        ordered : list[Tuple[str, Any]] = []
        seen = set()
        for key in prefered:
            if key in args:
                ordered.append((key, args[key]))
                seen.add(key)
        
        remaining_keys = set(args.keys() - seen)
        ordered.extend((key, args[key]) for key in remaining_keys)
        return ordered
    
    def render_arguments_table(self, tool_name: str, arguments: dict[str,Any])-> Table:
        table = Table.grid(padding=(0,1))
        table.add_column(justify="right", style="muted", no_wrap=True)
        table.add_column(style="code", overflow="fold")

        for key, value in self.ordered_arguments(tool_name, arguments):
            table.add_row(key, value)

        return table

    
    def tool_call_start(self, call_id: str, name: str,tool_kind: str, arguments: dict[str,Any])-> None:
        self._tool_args_by_call_id[call_id] = arguments
        border_style = f"tool.{tool_kind}" if tool_kind else "tool"

        title = Text.assemble(
            ("⏺ ", "muted"),
            (name, "tool"),
            ("  ", "muted"),
            (f"#{call_id[:8]}", "muted"),
        )

        display_args = dict(arguments)

        for key in ('path', 'cwd'):
            if key in display_args:
                val = display_args.get(key)
                if isinstance(val, str) and self.cwd:
                    display_args[key] = str(display_path_rel_to_cwd(val, self.cwd))
                    

        panel = Panel(
            renderable=self.render_arguments_table(name, display_args) if display_args else Text("(no arguments)", style="muted"),
            title=title,
            title_align="left",
            subtitle=Text('running',style="muted"),
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1,2),
        )

        self.console.print()
        self.console.print(panel)
        