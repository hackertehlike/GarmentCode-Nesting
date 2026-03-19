import sys
from nicegui import ui
from pathlib import Path
import os
from nesting.gui import NestingGUI
import nesting.config as config

def main() -> None:
    # Check if pattern path is provided
    use_default = False
    if len(sys.argv) < 2:
        print("No pattern specified. Using default pattern.")
        pattern_path = Path(config.DEFAULT_PATTERN_PATH)
        use_default = True
    else:
        pattern_path = Path(sys.argv[1])
    
    # Set port
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8082
    
    # Initialize with pattern path and use_default flag
    # This tells NestingGUI whether to load default style/body params
    NestingGUI(pattern_path, use_default_params=use_default)
    ui.run(port=port)

if __name__ in {"__main__", "__mp_main__"}:
    import multiprocessing
    # Only the actual GUI process should start the server.
    # On macOS, ProcessPoolExecutor workers re-import __main__ with spawn;
    # parent_process() is None only in the originating process.
    if multiprocessing.parent_process() is None:
        main()