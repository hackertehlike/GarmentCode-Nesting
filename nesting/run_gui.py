import sys
from nicegui import ui
from pathlib import Path
from .gui import NestingGUI

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m nesting.run_gui <pattern.json> [port]")
        sys.exit(1)

    pattern_path = Path(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8082

    NestingGUI(pattern_path)
    ui.run(port=port)

if __name__ in {"__main__", "__mp_main__"}:
    main()
