from __future__ import annotations
import itertools
import tempfile
import copy
from typing import Dict, List, Tuple

from nicegui import ui, events
from nesting import utils  # add_seam_allowance, polygons_overlap, etc.
from nesting.path_extractor import PatternPathExtractor
from nesting.layout import Layout, Container
from nesting.placement_engine import BottomLeftDecoder, GreedyBLDecoder, NFPDecoder

# viewer size in CSS‑pixels
MAX_CANVAS_PX_WIDTH  = 800   
MAX_CANVAS_PX_HEIGHT = 600

class NestingGUI:
    """
    Display garment‑pattern JSON files as SVG outlines with seam allowance.
    • All JSON coordinates and container inputs are in centimetres.
    • A uniform seam allowance (cm) can be typed in the toolbar.
    • Both the original outline (dashed grey) and the enlarged outline
      (filled colour, draggable) are rendered for each panel.
    • A button tests all enlarged outlines for pairwise intersections.
    """

    def __init__(self) -> None:
        # container dimensions in cm
        self.container_width_cm  = 140.0
        self.container_height_cm = 200.0

        self._update_scale_factors()

        # seam allowance in cm
        self.seam_allowance_cm: float = 1.0

        # geometry stores (in cm)
        self.raw_panel_outlines: Dict[str, List[List[float]]] = {}  # as read from JSON
        self.panel_outlines:     Dict[str, List[List[float]]] = {}   # after seam allowance
        self.offset_px: Tuple[float, float] = (0.0, 0.0)            # for centering

        # UI drag bookkeeping
        self.panel_path_refs: Dict[str, Tuple[any, any]] = {}
        self.panel_transforms: Dict[str, Tuple[float, float]] = {}
        self.drag_data: Dict = {}

        self.pattern_loaded = False

        self.style_params: Dict = {}
        self.parameter_order: List[str] = []

        # Currently selected panel for style editing.
        self.selected_panel: str = ""

        self._build_layout()
        self._load_default_pattern()  # auto load default pattern

    # ---------------------------------------------------------------------------- #
    #                                  UI Builders                                 #
    # ---------------------------------------------------------------------------- #

    def _build_layout(self) -> None:
        # Layout with canvas/toolbar on the left and a style sidebar on the right.
        with ui.row().classes("w-full h-screen"):
            with ui.column().classes("flex-1"):
                self._build_canvas()
                self._build_toolbar()
            with ui.column().classes("w-1/4 p-4").style("background:#f9f9f9;"):
                self._build_sidebar()

    def _build_canvas(self) -> None:
        with ui.element("div").classes("relative").style(
            f"width:{self.container_width}px;height:{self.container_height}px"
        ):
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

    def _build_toolbar(self):
        with ui.row().classes("w-full items-center justify-center gap-4"):
            self.width_input  = ui.number(
                "Width (cm)",  value=self.container_width_cm,
                on_change=self._update_dimensions,
            )
            self.height_input = ui.number(
                "Height (cm)", value=self.container_height_cm,
                on_change=self._update_dimensions,
            )
            self.sa_input = ui.number(
                "Seam allowance (cm)", value=self.seam_allowance_cm,
                on_change=self._update_seam_allowance,
            )
            ui.button("Check intersections & boundaries", on_click=self._check_intersections)
            
            ui.upload(label="Load JSON pattern", on_upload=self._load_pattern,
                      auto_upload=True)
            
            ui.button("Auto place (Bottom‑Left)", on_click=self._auto_place)
            ui.button("Auto place (Greedy)", on_click=lambda _: self._auto_place("Greedy"))
            ui.button("Auto place (NFP)", on_click=lambda _: self._auto_place("NFP"))

            # ui.button("Gravitate", on_click=self._gravitate)

    def _build_sidebar(self) -> None:   
        with ui.column() as sb:
            ui.label("Style Parameters").classes("text-lg font-bold")
            # Panel selector - using label instead of placeholder.
            self.panel_select = ui.select(
                options=list(self.raw_panel_outlines.keys()),
                label="Select Panel",
                on_change=self._update_sidebar,
            )
            # Container to hold parameter inputs.
            self.param_container = ui.column()

    def _update_sidebar(self, e):
        selected_panel = self.panel_select.value
        self.selected_panel = selected_panel
        self.param_container.clear()
        
        ### ????? ###
        if not self.style_params:
            self.style_params = {}
        
        # currently not working, not sure where to get style_params from
        # params = self.style_params[selected_panel]
        # ui.label(f"Parameters for panel '{selected_panel}':").classes("mt-4")
        # for key, val in params.items():
        #     ui.number(
        #         label=key,
        #         value=val,
        #         on_change=lambda e, k=key, panel=selected_panel: self._on_param_change(panel, k, e),
        #     )


    # ---------------------------------------------------------------------------- #
    #                     PARAMETER CHANGE STUFF (NOT WORKING)                     #
    # ---------------------------------------------------------------------------- #

    def _on_param_change(self, panel: str, param: str, e) -> None:
        new_val = float(e.value)
        self.style_params[panel][param] = new_val
        ui.notify(f"Parameter {param} for panel {panel} updated to {new_val}")
        self._update_pattern_by_style()

    # TODO: recalculate panels
    def _update_pattern_by_style(self):
        pass

    # ---------------------------------------------------------------------------- #
    #                                  TOOLBAR                                     #
    # ---------------------------------------------------------------------------- #

    def _update_dimensions(self, _):
        self.container_width_cm  = float(self.width_input.value or self.container_width_cm)
        self.container_height_cm = float(self.height_input.value or self.container_height_cm)
        self._update_scale_factors()
        self.scene.props(
            f'width="{self.container_width}" '
            f'height="{self.container_height}" '
            f'viewBox="0 0 {self.container_width} {self.container_height}"'
        )
        if self.pattern_loaded:
            self._draw_outlines()

    def _update_seam_allowance(self, _):
        self.seam_allowance_cm = float(self.sa_input.value or 0.0)
        if self.pattern_loaded:
            self._rebuild_panel_outlines()
            self._draw_outlines()

    # ---------------------------------------------------------------------------- #
    #                                PATTERN LOADERS                               #
    # ---------------------------------------------------------------------------- #
    
    def _load_pattern(self, e: events.UploadEventArguments):
        """
        Loads pattern from the uploaded JSON file, populates necessary fields and draws its outlines
        """
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
            extractor = PatternPathExtractor(tmp_path)  # outlines in cm
            self.raw_panel_outlines = extractor.get_all_panel_outlines(samples_per_edge=20)
            ui.notify("Panels found: " + ", ".join(self.raw_panel_outlines.keys()))
            self._rebuild_panel_outlines()
            self.pattern_loaded = True
            self._draw_outlines()
            # Extract real style parameters and parameter order from the spec.
            # self.style_params = extractor.spec.get("parameters", {})
            # self.parameter_order = extractor.spec.get("parameter_order", list(self.style_params.keys()))
            # Update panel selector options.
            self.panel_select.options = [str(k) for k in self.raw_panel_outlines.keys()]
            if self.panel_select.options:
                self.panel_select.value = self.panel_select.options[0]
            
            ui.notify("Pattern loaded ✓", type="positive")
        except Exception as exc:
            ui.notify(f"Could not load pattern: {exc}", type="negative")

    def _load_default_pattern(self):
        """
        So I don't have to load something in every time I change code... 
        """
        default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification.json"
        try:
            extractor = PatternPathExtractor(default_path)
            self.raw_panel_outlines = extractor.get_all_panel_outlines(samples_per_edge=20)
            #ui.notify("Default pattern loaded: " + ", ".join(self.raw_panel_outlines.keys()))
            
            self._rebuild_panel_outlines() # add seam allowance
            self.pattern_loaded = True
            self._draw_outlines()
            # self.style_params = extractor.spec.get("parameters", {})
            # self.parameter_order = extractor.spec.get("parameter_order", list(self.style_params.keys()))

            # what does this do actually
            if hasattr(self, "panel_select"):
                self.panel_select.options = [str(k) for k in self.raw_panel_outlines.keys()]
                if self.panel_select.options:
                    self.panel_select.value = self.panel_select.options[0]
        except Exception as exc:
            ui.notify(f"Could not load default pattern: {exc}", type="negative")

    def _update_scale_factors(self):
        """
        Conversion from cm to pixels for display
        Uses the global canvas size to retain scale
        """
        self.effective_scale = min(
            MAX_CANVAS_PX_WIDTH  / self.container_width_cm,
            MAX_CANVAS_PX_HEIGHT / self.container_height_cm,
        )
        self.container_width  = int(self.container_width_cm  * self.effective_scale)
        self.container_height = int(self.container_height_cm * self.effective_scale)

    def _rebuild_panel_outlines(self):
        """
        Re‑computes `panel_outlines` from `raw_panel_outlines` + seam allowance.
        """
        sa = self.seam_allowance_cm
        self.panel_outlines = {}
        for name, outline in self.raw_panel_outlines.items():
            self.panel_outlines[name] = (
                outline if abs(sa) < 1e-9
                else utils.add_seam_allowance(outline, allowance=sa)[0]
            )

    # ---------------------------------------------------------------------------- #
    #                                  DRAWING                                     #
    # ---------------------------------------------------------------------------- #

    def _draw_outlines(self):
        self.scene.clear()
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

        offset_x = 0
        offset_y = 0
        self.offset_px = (offset_x, offset_y)

        # Optional border
        for x1, y1, x2, y2 in [
            (0, 0, self.container_width, 0),
            (self.container_width, 0, self.container_width, self.container_height),
            (self.container_width, self.container_height, 0, self.container_height),
            (0, self.container_height, 0, 0)
        ]:
            self._svg_line(x1, y1, x2, y2, stroke="#8a8a8a")

        fills = itertools.cycle(["#c3e6cb", "#bee5eb", "#ffeeba", "#f5c6cb"])
        self.panel_path_refs.clear()
        self.panel_transforms.clear()

        for name, outer_cm in self.panel_outlines.items():
            inner_cm = self.raw_panel_outlines.get(name, [])
            
            oxs, oys = zip(*outer_cm)
            ixs, iys = zip(*inner_cm)
            dx_cm = (max(oxs) - min(oxs) - (max(ixs) - min(ixs))) / 2
            dy_cm = (max(oys) - min(oys) - (max(iys) - min(iys))) / 2
            out_px = [(x*self.effective_scale + offset_x,
                       y*self.effective_scale + offset_y) for x, y in outer_cm]
            in_px = [((x + dx_cm) * self.effective_scale + offset_x, 
                      (y + dy_cm) * self.effective_scale + offset_y)
         for x, y in inner_cm]
            d_out = "M " + " L ".join(f"{x} {y}" for x, y in out_px) + " Z"
            d_in  = "M " + " L ".join(f"{x} {y}" for x, y in in_px) + " Z"

            inner_path = self._svg_path_static(
                d_in, stroke="#4b5563", stroke_width=1, stroke_dash="4 2"
            )
            outer_path = self._svg_path_draggable(
                d_out, panel_id=name,
                stroke="#ed7ea7", stroke_width=2,
                fill=next(fills), fill_opacity=0.35,
            )

            self.panel_path_refs[name] = (outer_path, inner_path)
            self.panel_transforms[name] = (0, 0)


    #------------------------- svg stuff - thank you chatgpt -----------------------#
    def _svg_line(self, x1, y1, x2, y2, *, stroke="#000") -> None:
        with self.scene:
            ui.element("line").props(
                f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="1"'
            )

    def _svg_path_static(self, d: str, *, stroke="#000", stroke_width=1, stroke_dash: str | None = None):
        props = (
            f'd="{d}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'fill="none" pointer-events="none"'
        )
        if stroke_dash:
            props += f' stroke-dasharray="{stroke_dash}"'
        with self.scene:
            element = ui.element("path").props(props)
        return element

    def _svg_path_draggable(self, d: str, *, panel_id: str,
                             stroke="#000", stroke_width=1,
                             fill="#ffffff", fill_opacity=1.0):
        with self.scene:
            element = ui.element("path").props(
                f'd="{d}" stroke="{stroke}" stroke-width="{stroke_width}" '
                f'fill="{fill}" fill-opacity="{fill_opacity}"'
            )
        element.on("pointerdown", lambda e, p=panel_id: self._on_drag_start(e, p))
        element.on("pointermove", lambda e, p=panel_id: self._on_drag_move(e, p))
        element.on("pointerup",   lambda e, p=panel_id: self._on_drag_end(e, p))
        return element

    # ---------------------------------------------------------------------------- #
    #               click and drag stuff, thx chatgpt but it works :)              #
    # ---------------------------------------------------------------------------- #

    def _on_drag_start(self, e, panel_id: str):
        self.drag_data = {
            'panel_id': panel_id,
            'start_x': e.args.get('clientX', 0),
            'start_y': e.args.get('clientY', 0),
            'orig_offset': self.panel_transforms.get(panel_id, (0, 0)),
        }

    def _on_drag_move(self, e, panel_id: str):
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

    def _on_drag_end(self, *_):
        self.drag_data = {}
        # print offsets
        # for name, (dx, dy) in self.panel_transforms.items():
        #     print(f"Panel {name} offset: ({dx}, {dy})")

    def _global_drag_move(self, e) -> None:
        if self.drag_data:
            self._on_drag_move(e, self.drag_data['panel_id'])

    def _global_drag_end(self, e) -> None:
        if self.drag_data:
            self._on_drag_end(e)

    # ---------------------------------------------------------------------------- #
    #            INTERSECTION AND CONTAINMENT BOUNDARY CHECK                       #
    # ---------------------------------------------------------------------------- #
    def _check_intersections(self) -> None:
        if not self.pattern_loaded:
            ui.notify("Load a pattern first", type="warning")
            return

        ox, oy = self.offset_px
        overlaps: List[Tuple[str, str]] = []
        names = list(self.panel_outlines.keys())

        def to_px(path_cm, dx, dy):
            return [(x * self.effective_scale + ox + dx,
                     y * self.effective_scale + oy + dy) for x, y in path_cm]

        # Check for intersections between panels.
        for i, name_i in enumerate(names):
            dx_i, dy_i = self.panel_transforms.get(name_i, (0, 0))
            poly_i = to_px(self.panel_outlines[name_i], dx_i, dy_i)
            for name_j in names[i+1:]:
                dx_j, dy_j = self.panel_transforms.get(name_j, (0, 0))
                poly_j = to_px(self.panel_outlines[name_j], dx_j, dy_j)
                if utils.polygons_overlap(poly_i, poly_j):
                    overlaps.append((name_i, name_j))

        # Check for panels that extend outside the container bounds.
        panels_outside = []
        for name in names:
            dx, dy = self.panel_transforms.get(name, (0, 0))
            poly = to_px(self.panel_outlines[name], dx, dy)
            xs_poly, ys_poly = zip(*poly)
            if min(xs_poly) < 0 or max(xs_poly) > self.container_width or min(ys_poly) < 0 or max(ys_poly) > self.container_height:
                panels_outside.append(name)

        # Build the notification message.
        messages = []
        if overlaps:
            messages.append("Overlaps:\n" + "\n".join(f"• {a} × {b}" for a, b in overlaps))
        else:
            messages.append("No intersections")
        if panels_outside:
            messages.append("Panels outside container:\n" + "\n".join(f"• {p}" for p in panels_outside))
        else:
            messages.append("All panels within container")

        ui.notify("\n".join(messages), multi_line=True)

    # ---------------------------------------------------------------------------- #
    #                                    DECODER                                   #
    # ---------------------------------------------------------------------------- #

    def _apply_placements(self, placements_cm: list[tuple[int, float, float]]):
        """
        placements_cm – list of (piece_id, dx_cm, dy_cm)
        It rewrites self.panel_transforms in pixels and updates the SVG
        paths that were created by _draw_outlines().
        """
        if not self.pattern_loaded:
            ui.notify("Load a pattern first", type="warning")
            return

        self._draw_outlines()

        cm_to_px = self.effective_scale          # 1 cm → this many CSS‑pixels
        for name, dx_cm, dy_cm in placements_cm:
            if name not in self.panel_path_refs:
                ui.notify(f"Unknown panel '{name}'", type="warning")
                continue
            
            dx_px = dx_cm * cm_to_px
            dy_px = dy_cm * cm_to_px
            self.panel_transforms[name] = (dx_px, dy_px)
            outer, inner = self.panel_path_refs[name]
            
            for path in (outer, inner):
                path.props(f'transform="translate({dx_px},{dy_px})"')

        ui.notify("Automatic placement applied", type="positive")

    def _auto_place(self, method="BL"):
        if not self.pattern_loaded:
            ui.notify('Load a pattern first', type='warning')
            return

        try:
            layout = Layout(self.panel_outlines)
                                                            
            container = Container(self.container_width_cm,
                                self.container_height_cm)

            if method == "BL":
                # Bottom-Left placement
                decoder = BottomLeftDecoder(layout, container, step=1.0)
            elif method == "Greedy":
                # Greedy placement
                decoder = GreedyBLDecoder(layout, container)
            elif method == "NFP":
                decoder = NFPDecoder(layout, container)
            else:
                raise ValueError(f"Unknown placement method: {method}")
            
            placements = decoder.decode()                     # [(name, dx, dy)]

            print(f"Auto placement ({method}) usage:")
            print(decoder.usage_BB())

            self._apply_placements(placements)

            ui.notify('Auto placement completed ', type='positive')

        except Exception as exc:
            ui.notify(f'Auto placement failed: {exc}', type='negative')

if __name__ in {"__main__", "__mp_main__"}:
    NestingGUI()
    ui.run(port=8080)
