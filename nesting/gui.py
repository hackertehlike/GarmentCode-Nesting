from __future__ import annotations

import itertools
import tempfile
from typing import Dict, List, Tuple

from nicegui import ui, events
from nesting import utils                 # add_seam_allowance, polygons_overlap …
from nesting.path_extractor import PatternPathExtractor

# ─────────────────────────────────────────────────────────────────────────────
MAX_CANVAS_PX_WIDTH  = 800   # viewer size in CSS‑pixels
MAX_CANVAS_PX_HEIGHT = 600
# ─────────────────────────────────────────────────────────────────────────────


class NestingGUI:
    """
    Display garment‑pattern JSON files as SVG outlines with seam allowance.

    • All JSON coordinates and container inputs are in **centimetres**.
    • A uniform seam allowance (cm) can be typed in the toolbar.
    • Both the original outline (dashed grey) and the enlarged outline
      (filled colour, draggable) are rendered for each panel.
    • A button tests all enlarged outlines for pairwise intersections.
    """

    # ────────────────────────────────────────────────────────────────── init ──
    def __init__(self) -> None:
        # container dimensions in cm
        self.container_width_cm  = 140.0
        self.container_height_cm = 200.0

        # compute pixel scale so container fits 800 × 600 px
        self._update_scale_factors()

        # user‑defined seam allowance (cm)
        self.seam_allowance_cm: float = 0.0

        # geometry stores
        self.raw_panel_outlines: Dict[str, List[List[float]]] = {}  # as read
        self.panel_outlines:     Dict[str, List[List[float]]] = {}  # after SA
        self.offset_px: Tuple[float, float] = (0.0, 0.0)            # centring

        # drag bookkeeping
        self.panel_path_refs:   Dict[str, Tuple[any, any]]  = {}
        self.panel_transforms:  Dict[str, Tuple[float, float]] = {}
        self.drag_data: Dict = {}

        self.pattern_loaded = False

        self._build_layout()

    # ─────────────────────────────────────────────────────────── UI builders ──
    def _build_layout(self) -> None:
        with ui.column().classes("w-full h-screen items-center justify-center"):
            self._build_canvas()
            self._build_toolbar()

    def _build_canvas(self) -> None:
        with ui.element("div").classes("relative").style(
            f"width:{self.container_width}px;height:{self.container_height}px"
        ):
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
            self.scene.on("pointermove", self._global_drag_move)
            self.scene.on("pointerup",   self._global_drag_end)

    def _build_toolbar(self) -> None:
        with ui.row().classes("w-full items-center justify-center gap-4"):
            self.width_input  = ui.number(
                "Width (cm)",  value=self.container_width_cm,
                on_change=self._update_dimensions,
            )
            self.height_input = ui.number(
                "Height (cm)", value=self.container_height_cm,
                on_change=self._update_dimensions,
            )
            self.sa_input = ui.number(                      # seam allowance
                "Seam allowance (cm)", value=self.seam_allowance_cm,
                on_change=self._update_seam_allowance,
            )
            ui.button("Check intersections", on_click=self._check_intersections)
            ui.button("Reset position",      on_click=self._reset_position)
            ui.upload(label = "Load JSON pattern",   on_upload=self._load_pattern,
                      auto_upload=True)

    # ─────────────────────────────────────────────────── toolbar callbacks ──
    def _update_dimensions(self, _) -> None:
        self.container_width_cm  = float(self.width_input.value  or self.container_width_cm)
        self.container_height_cm = float(self.height_input.value or self.container_height_cm)
        self._update_scale_factors()
        self.scene.props(
            f'width="{self.container_width}" '
            f'height="{self.container_height}" '
            f'viewBox="0 0 {self.container_width} {self.container_height}"'
        )
        if self.pattern_loaded:
            self._draw_outlines()

    def _update_seam_allowance(self, _) -> None:
        self.seam_allowance_cm = float(self.sa_input.value or 0.0)
        if self.pattern_loaded:
            self._rebuild_panel_outlines()
            self._draw_outlines()

    # ───────────────────────────────────────────────────── pattern loading ──
    def _load_pattern(self, e: events.UploadEventArguments) -> None:
        self.raw_panel_outlines.clear()
        self.panel_outlines.clear()
        self.panel_path_refs.clear()
        self.panel_transforms.clear()
        self.drag_data.clear()
        self.scene.clear()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(e.content.read())
            tmp_path = tmp.name

        try:
            extractor = PatternPathExtractor(tmp_path)     # outlines in cm
            self.raw_panel_outlines = extractor.get_all_panel_outlines(samples_per_edge=20)
            self._rebuild_panel_outlines()
            self.pattern_loaded = True
            self._draw_outlines()
            ui.notify("Pattern loaded ✓", type="positive")
        except Exception as exc:                           # noqa: BLE001
            ui.notify(f"Could not load pattern: {exc}", type="negative")

    # ───────────────────────────────────────────── geometry construction ──
    def _update_scale_factors(self) -> None:
        self.effective_scale = min(
            MAX_CANVAS_PX_WIDTH  / self.container_width_cm,
            MAX_CANVAS_PX_HEIGHT / self.container_height_cm,
        )
        self.container_width  = int(self.container_width_cm  * self.effective_scale)
        self.container_height = int(self.container_height_cm * self.effective_scale)

    def _rebuild_panel_outlines(self) -> None:
        """Re‑compute `panel_outlines` from `raw_panel_outlines` + allowance."""
        sa = self.seam_allowance_cm
        self.panel_outlines = {}
        for name, outline in self.raw_panel_outlines.items():
            self.panel_outlines[name] = (
                outline if abs(sa) < 1e-9
                else utils.add_seam_allowance(outline, allowance=sa)[0]
            )

    # ───────────────────────────────────────────────────── drawing logic ──
    def _draw_outlines(self) -> None:
        self.scene.clear()

        # build list of *all* vertices in px space
        all_px = [
            (x * self.effective_scale, y * self.effective_scale)
            for outline in self.panel_outlines.values()
            for (x, y) in outline
        ]
        if not all_px:
            return

        xs, ys = zip(*all_px)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pattern_w, pattern_h = max_x - min_x, max_y - min_y

        offset_x = (self.container_width  - pattern_w) / 2 - min_x
        offset_y = (self.container_height - pattern_h) / 2 - min_y
        self.offset_px = (offset_x, offset_y)      # store for overlap tester

        # optional border
        for x1, y1, x2, y2 in [(0, 0, self.container_width, 0),
                               (self.container_width, 0, self.container_width, self.container_height),
                               (self.container_width, self.container_height, 0, self.container_height),
                               (0, self.container_height, 0, 0)]:
            self._svg_line(x1, y1, x2, y2, stroke="#8a8a8a")

        fills = itertools.cycle(["#c3e6cb", "#bee5eb", "#ffeeba", "#f5c6cb"])
        self.panel_path_refs.clear()
        self.panel_transforms.clear()

        for name, outer_cm in self.panel_outlines.items():
            inner_cm = self.raw_panel_outlines.get(name, [])

            # convert to px + centre offset
            out_px = [(x*self.effective_scale + offset_x,
                       y*self.effective_scale + offset_y) for x, y in outer_cm]
            in_px  = [(x*self.effective_scale + offset_x,
                       y*self.effective_scale + offset_y) for x, y in inner_cm]

            d_out = "M " + " L ".join(f"{x} {y}" for x, y in out_px) + " Z"
            d_in  = "M " + " L ".join(f"{x} {y}" for x, y in in_px)  + " Z"

            # static dashed inner outline
            inner_path = self._svg_path_static(
                d_in, stroke="#4b5563", stroke_width=1, stroke_dash="4 2"
            )
            # draggable outer outline
            outer_path = self._svg_path_draggable(
                d_out, panel_id=name,
                stroke="#ed7ea7", stroke_width=2,
                fill=next(fills), fill_opacity=0.35,
            )

            self.panel_path_refs[name]  = (outer_path, inner_path)
            self.panel_transforms[name] = (0, 0)

    # ─────────────────────────────────────────────── low‑level SVG helpers ──
    def _svg_line(self, x1, y1, x2, y2, *, stroke="#000") -> None:
        with self.scene:
            ui.element("line").props(
                f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="1"'
            )

    def _svg_path_static(
        self, d: str, *,
        stroke="#000", stroke_width=1, stroke_dash: str | None = None
    ):
        props = (
            f'd="{d}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'fill="none" pointer-events="none"'
        )
        if stroke_dash:
            props += f' stroke-dasharray="{stroke_dash}"'

        with self.scene:
            element = ui.element("path").props(props)
        return element

    def _svg_path_draggable(
        self, d: str, *, panel_id: str,
        stroke="#000", stroke_width=1,
        fill="#ffffff", fill_opacity=1.0,
    ):
        with self.scene:
            element = ui.element("path").props(
                f'd="{d}" stroke="{stroke}" stroke-width="{stroke_width}" '
                f'fill="{fill}" fill-opacity="{fill_opacity}"'
            )
        element.on("pointerdown", lambda e, p=panel_id: self._on_drag_start(e, p))
        element.on("pointermove", lambda e, p=panel_id: self._on_drag_move(e, p))
        element.on("pointerup",   lambda e, p=panel_id: self._on_drag_end(e, p))
        return element

    # ──────────────────────────────────────────────────────── drag logic ──
    def _on_drag_start(self, e, panel_id: str) -> None:
        self.drag_data = {
            'panel_id':    panel_id,
            'start_x':     e.args.get('clientX', 0),
            'start_y':     e.args.get('clientY', 0),
            'orig_offset': self.panel_transforms.get(panel_id, (0, 0)),
        }

    def _on_drag_move(self, e, panel_id: str) -> None:
        if not self.drag_data or self.drag_data['panel_id'] != panel_id:
            return
        dx = e.args.get('clientX', 0) - self.drag_data['start_x']
        dy = e.args.get('clientY', 0) - self.drag_data['start_y']
        orig_x, orig_y = self.drag_data['orig_offset']
        new_offset = (orig_x + dx, orig_y + dy)
        self.panel_transforms[panel_id] = new_offset

        outer_path, inner_path = self.panel_path_refs[panel_id]
        for path in (outer_path, inner_path):
            path.props(f'transform="translate({new_offset[0]},{new_offset[1]})"')

    def _on_drag_end(self, *_):    # noqa: D401, ANN001
        self.drag_data = {}

    def _global_drag_move(self, e) -> None:
        if self.drag_data:
            self._on_drag_move(e, self.drag_data['panel_id'])

    def _global_drag_end(self, e) -> None:   # noqa: D401, ANN001
        if self.drag_data:
            self._on_drag_end(e)

    def _reset_position(self) -> None:
        for pid, (outer, inner) in self.panel_path_refs.items():
            outer.props('transform=""')
            inner.props('transform=""')
            self.panel_transforms[pid] = (0, 0)

    # ───────────────────────────────────────────── intersection checker ──
    def _check_intersections(self) -> None:
        if not self.pattern_loaded:
            ui.notify("Load a pattern first", type="warning")
            return

        ox, oy = self.offset_px
        overlaps: List[Tuple[str, str]] = []
        names = list(self.panel_outlines.keys())

        def to_px(path_cm, dx, dy):
            return [(x*self.effective_scale + ox + dx,
                     y*self.effective_scale + oy + dy) for x, y in path_cm]

        for i, name_i in enumerate(names):
            dx_i, dy_i = self.panel_transforms.get(name_i, (0, 0))
            poly_i = to_px(self.panel_outlines[name_i], dx_i, dy_i)
            for name_j in names[i+1:]:
                dx_j, dy_j = self.panel_transforms.get(name_j, (0, 0))
                poly_j = to_px(self.panel_outlines[name_j], dx_j, dy_j)
                if utils.polygons_overlap(poly_i, poly_j):
                    overlaps.append((name_i, name_j))

        if overlaps:
            ui.notify(
                "Overlaps:\n" + "\n".join(f"• {a} × {b}" for a, b in overlaps),
                type="negative", multi_line=True,
            )
        else:
            ui.notify("No intersections ☑", type="positive")


# ─────────────────────────────────────────────────────────── entry‑point ──
if __name__ in {"__main__", "__mp_main__"}:
    NestingGUI()
    ui.run(port=8080)
