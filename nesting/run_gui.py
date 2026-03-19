"""Backward-compatibility shim — redirects to nesting.gui.run_gui.

Multiprocessing on macOS (spawn start method) re-imports __main__ by its
original dotted name when spawning worker processes.  This shim keeps the
old entry-point nesting.run_gui alive so those workers can resolve the module.

Prefer the canonical entry-point:
    python -m nesting.gui.run_gui
"""
from nesting.gui.run_gui import *  # noqa: F401, F403

if __name__ in {"__main__", "__mp_main__"}:
    import multiprocessing
    if multiprocessing.parent_process() is None:
        from nesting.gui.run_gui import main
        main()
