from __future__ import annotations
import itertools
import tempfile
from typing import Dict, List, Tuple

from nicegui import ui, events
from nesting.path_extractor import PatternPathExtractor

# Maximum canvas size (in pixels) that should be used to display the pattern.
MAX_CANVAS_PX_WIDTH = 800
MAX_CANVAS_PX_HEIGHT = 600

class NestingGUI:
    """Display garment‑pattern JSON files as filled, centred SVG outlines with draggable panels.
       The container dimensions and JSON coordinates are in centimeters.
       The entire canvas is scaled to fit within the predefined maximum pixel size while preserving ratio.
    """

    def __init__(self) -> None:
        # Input dimensions in cm (for the container and JSON pattern).
        self.container_width_cm = 20.0  # for example, 20 cm
        self.container_height_cm = 15.0  # for example, 15 cm

        # Compute effective scale so that the canvas fits the maximum allowed size.
        self.effective_scale = min(
            MAX_CANVAS_PX_WIDTH / self.container_width_cm,
            MAX_CANVAS_PX_HEIGHT / self.container_height_cm,
        )
        self.container_width = int(self.container_width_cm * self.effective_scale)
        self.container_height = int(self.container_height_cm * self.effective_scale)

        self.pattern_loaded: bool = False
        self.panel_outlines: Dict[str, List[List[float]]] = {}
        # For drag-and-drop:
        self.panel_path_refs: Dict[str, any] = {}   # Store SVG path elements by panel key
        self.panel_transforms: Dict[str, Tuple[float, float]] = {}  # Current translate offsets
        self.drag_data: Dict = {}  # Temporary drag state storage
        self._build_layout()

    def _build_layout(self) -> None:
        with ui.column().classes("w-full h-screen items-center justify-center"):
            self._build_canvas()
            self._build_toolbar()

    def _build_canvas(self) -> None:
        with ui.element("div").classes("relative").style(
            f"width:{self.container_width}px;height:{self.container_height}px"
        ):
            # Use a background image (note: if the image is a raster, its scale may not match exactly).
            ui.image("/img/millimiter_paper_1500_900.png").classes(
                "absolute top-0 left-0 w-full h-full object-contain"
            )
            self.scene = (
                ui.element("svg")
                .props(
                    f'width="{self.container_width}" '
                    f'height="{self.container_height}" '
                    f'viewBox="0 0 {self.container_width} {self.container_height}"'
                )
                .style("position:absolute;top:0;left:0")
            )
            # Attach global pointer events so dragging isn’t lost if the cursor moves off a path.
            self.scene.on("pointermove", lambda e: self._global_drag_move(e))
            self.scene.on("pointerup", lambda e: self._global_drag_end(e))

    def _build_toolbar(self) -> None:
        with ui.row().classes("w-full items-center justify-center gap-4"):
            self.width_input = ui.number(
                "Width (cm)", value=self.container_width_cm, on_change=self._update_dimensions
            )
            self.height_input = ui.number(
                "Height (cm)", value=self.container_height_cm, on_change=self._update_dimensions
            )
            ui.button("Reset position", on_click=self._reset_position)
            ui.upload(
                label="Load JSON pattern",
                on_upload=self._load_pattern,
                auto_upload=True,
            )

    def _update_dimensions(self, _) -> None:
        # Read new container size in cm from inputs.
        self.container_width_cm = float(self.width_input.value or self.container_width_cm)
        self.container_height_cm = float(self.height_input.value or self.container_height_cm)
        # Compute effective scale so that the canvas fits within the maximum dimensions.
        self.effective_scale = min(
            MAX_CANVAS_PX_WIDTH / self.container_width_cm,
            MAX_CANVAS_PX_HEIGHT / self.container_height_cm,
        )
        self.container_width = int(self.container_width_cm * self.effective_scale)
        self.container_height = int(self.container_height_cm * self.effective_scale)
        self.scene.props(
            f'width="{self.container_width}" '
            f'height="{self.container_height}" '
            f'viewBox="0 0 {self.container_width} {self.container_height}"'
        )
        if self.pattern_loaded:
            self._draw_outlines()

    def _reset_position(self) -> None:
        self.scene.style("left:0;top:0")

    def _load_pattern(self, e: events.UploadEventArguments) -> None:
        self.panel_outlines = {}
        self.panel_path_refs = {}
        self.panel_transforms = {}
        self.drag_data = {}
        self.scene.clear()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(e.content.read())
            tmp_path = tmp.name
        try:
            extractor = PatternPathExtractor(tmp_path)
            # Assuming PatternPathExtractor returns outlines in centimeters
            self.panel_outlines = extractor.get_all_panel_outlines(samples_per_edge=20)
            self.pattern_loaded = True
            self._draw_outlines()
            ui.notify("Pattern loaded ✓")
        except Exception as exc:
            ui.notify(f"Could not load pattern: {exc}")

    def _draw_outlines(self) -> None:
        self.scene.clear()
        # Convert all JSON coordinates (in cm) to pixels using the effective scale.
        all_points = []
        for outline in self.panel_outlines.values():
            for x, y in outline:
                all_points.append((x * self.effective_scale, y * self.effective_scale))
        if not all_points:
            return
        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pattern_w, pattern_h = max_x - min_x, max_y - min_y
        # Compute offsets to center the pattern in the canvas.
        offset_x = (self.container_width - pattern_w) / 2 - min_x
        offset_y = (self.container_height - pattern_h) / 2 - min_y

        # Optionally draw the container border.
        cw, ch = self.container_width, self.container_height
        for x1, y1, x2, y2 in [(0, 0, cw, 0), (cw, 0, cw, ch),
                               (cw, ch, 0, ch), (0, ch, 0, 0)]:
            self._svg_line(x1, y1, x2, y2, stroke="#8a8a8a")

        fills = itertools.cycle(["#c3e6cb", "#bee5eb", "#ffeeba", "#f5c6cb"])
        self.panel_transforms = {}

        for panel_name, outline in self.panel_outlines.items():
            # Scale each coordinate (cm to px) and then shift to center.
            shifted = [(x * self.effective_scale + offset_x, y * self.effective_scale + offset_y)
                       for (x, y) in outline]
            d = "M " + " L ".join(f"{x} {y}" for x, y in shifted) + " Z"
            self.panel_transforms[panel_name] = (0, 0)
            path = self._svg_path_draggable(
                d,
                panel_id=panel_name,
                stroke="#ed7ea7",
                stroke_width=2,
                fill=next(fills),
                fill_opacity=0.35,
            )
            self.panel_path_refs[panel_name] = path

    def _svg_line(self, x1, y1, x2, y2, stroke="#000") -> None:
        with self.scene:
            ui.element("line").props(
                f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="1"'
            )

    def _svg_path_draggable(
        self,
        d: str,
        *,
        panel_id: str,
        stroke="#000",
        stroke_width=1,
        fill="#ffffff",
        fill_opacity=1.0,
    ):
        with self.scene:
            element = ui.element("path").props(
                f'd="{d}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}" '
                f'fill="{fill}" fill-opacity="{fill_opacity}"'
            )
        element.on("pointerdown", lambda e, p=panel_id: self._on_drag_start(e, p))
        element.on("pointermove", lambda e, p=panel_id: self._on_drag_move(e, p))
        element.on("pointerup", lambda e, p=panel_id: self._on_drag_end(e, p))
        return element

    def _on_drag_start(self, e, panel_id: str) -> None:
        start_x = e.args.get('clientX', 0)
        start_y = e.args.get('clientY', 0)
        self.drag_data = {
            'panel_id': panel_id,
            'start_x': start_x,
            'start_y': start_y,
            'orig_offset': self.panel_transforms.get(panel_id, (0, 0))
        }

    def _on_drag_move(self, e, panel_id: str) -> None:
        if not self.drag_data or self.drag_data.get('panel_id') != panel_id:
            return
        current_x = e.args.get('clientX', 0)
        current_y = e.args.get('clientY', 0)
        dx = current_x - self.drag_data['start_x']
        dy = current_y - self.drag_data['start_y']
        orig_x, orig_y = self.drag_data['orig_offset']
        new_offset = (orig_x + dx, orig_y + dy)
        self.panel_transforms[panel_id] = new_offset
        path_el = self.panel_path_refs[panel_id]
        path_el.props(f'transform="translate({new_offset[0]},{new_offset[1]})"')

    def _on_drag_end(self, e, panel_id: str) -> None:
        self.drag_data = {}

    def _global_drag_move(self, e) -> None:
        if self.drag_data:
            panel_id = self.drag_data.get('panel_id')
            if panel_id:
                self._on_drag_move(e, panel_id)

    def _global_drag_end(self, e) -> None:
        if self.drag_data:
            panel_id = self.drag_data.get('panel_id')
            if panel_id:
                self._on_drag_end(e, panel_id)

if __name__ in {"__main__", "__mp_main__"}:
    NestingGUI()
    ui.run(port=8080)