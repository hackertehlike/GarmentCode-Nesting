from __future__ import annotations
import itertools
import tempfile
import copy
# import yaml
from typing import Dict, List, Tuple
from pathlib import Path

from nicegui import ui, events
from nicegui.events import KeyEventArguments
from nesting import utils
from nesting.evolution import Evolution  # add_seam_allowance, polygons_overlap, etc.
from .path_extractor import *
from .layout import *
from .placement_engine import *
import nesting.config as config

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

    def __init__(self, pattern_path: str | Path | None = None) -> None:
        # container dimensions in cm
        self.container_width_cm  = config.CONTAINER_WIDTH_CM
        self.container_height_cm = config.CONTAINER_HEIGHT_CM
        
        self.container = Container(self.container_width_cm, self.container_height_cm)
        self.layout = None  # will be a Layout instance once pattern is loaded

        self._update_scale_factors()

        # seam allowance in cm
        self.seam_allowance_cm: float = config.SEAM_ALLOWANCE_CM

        self.pieces : Dict[str, Piece] = {}  # for storing pieces

        # UI drag bookkeeping
        self.panel_path_refs: Dict[str, Tuple[any, any]] = {}
        # self.panel_transforms: Dict[str, Tuple[float, float]] = {}
        self.drag_data: Dict = {}

        self.pattern_loaded = False
        #self.yaml_loaded     = False

        self.design_params: Dict[str, any] = {}  # design specification from JSON

        # Currently selected panel for style editing.
        self.selected_panel: str = ""
        self.selection_mode: bool = False


        self.hull_path: ui.element | None = None

        self._build_layout()

        if pattern_path is not None:
            # user supplied a pattern → load it once
            self._load_pattern_core(Path(pattern_path))
        else:
            # fall back to the previous hard-coded default
            self._load_default_pattern()


    # ---------------------------------------------------------------------------- #
    #                                  UI Builders                                 #
    # ---------------------------------------------------------------------------- #

    def _build_layout(self) -> None:
        # Layout with canvas/toolbar on the left and a style sidebar on the right.
        with ui.row().classes("w-full h-screen"):
            with ui.column().classes("flex-1"):
                self._build_toolbar()
                self._build_sidebar()    
                self._build_canvas()


    

    def _build_sidebar(self) -> None:
        """Build the sidebar with style parameters."""

        # print the design parameters to the console
        # print("Design parameters:", self.design_params)

        def _iter_leaf_params(node: dict, prefix: str = ''):
            GATES = {'style', 'sleeveless', 'type'}

            for key, value in node.items():
                if isinstance(value, dict) and {'v', 'type'} <= value.keys():
                    if value['v'] is not None or key in GATES:
                        yield prefix + key, value
                elif isinstance(value, dict):
                    yield from _iter_leaf_params(value, prefix + key + '.')
                    
        if not hasattr(self, "sidebar"):
            self.sidebar = ui.column().classes("w-1/3 h-full overflow-y-auto p-4")

        self.sidebar.clear()

        with self.sidebar:
            ui.label("Style Parameters").classes("text-xl font-bold mb-4")
            filtered = self._filtered_design_tree()  # filter out irrelevant blocks
            for name, spec in _iter_leaf_params(filtered):
                t, val = spec['type'], spec['v']

                if t in ('float', 'int'):
                    ui.number(value=val, label=name,
                            on_change=lambda e, n=name: self._on_param_change(n, e))


    def _build_canvas(self) -> None:
        with ui.element("div").classes("relative").style(
            f"width:{self.container_width_px}px;height:{self.container_height}px"
        ):
            self.scene = (
                ui.element("svg")
                .props(
                    f'width="{self.container_width_px}" '
                    f'height="{self.container_height}" '
                    f'viewBox="0 0 {self.container_width_px} {self.container_height}"'
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

            # Label to display utilization
            self.utilization_label = ui.label("Utilization: n/a")
            self.rest_length_label = ui.label("Rest length: n/a")
            self.utilization_concave_label = ui.label("Concave hull utilization: n/a")


            ui.button("Check intersections & boundaries", on_click=self._check_intersections)
            
            ui.upload(label="Load JSON pattern", on_upload=self._load_pattern,
                      auto_upload=True)
            

            # ui.upload(label="Load YAML parameters",
            #         on_upload=self._load_yaml,
            #         auto_upload=True,
            #         multiple=False)

            
            ui.button("Auto place (Bottom‑Left)", on_click=self._auto_place)
            ui.button("Auto place (Greedy)", on_click=lambda _: self._auto_place("Greedy"))
            ui.button("Auto place (NFP)", on_click=lambda _: self._auto_place("NFP"))
            ui.button("Random Order (BL)", on_click=lambda _: self._auto_place("Random Order BL"))
            ui.button("Genetic Algorithm", on_click=lambda _: self._auto_place("Genetic Algorithm"))

            # Button to enable selection mode.
            ui.button("Select Panel (S)", on_click=self._enable_selection_mode)
            ui.button("Reset Selection", on_click=lambda _: self._select_panel(""))
            ui.button("Rotate Panel (R)", on_click=lambda _: self._rotate_panel())
            ui.button("Reset rotations", on_click=lambda _: self._reset_rotations())

            ui.button("Calculate Usage for Current Layout", on_click=self._calculate_usage)

        self._kb = ui.keyboard(on_key=self._handle_key, active=True)

    def _calculate_usage(self):
        scale = 1.0 / self.effective_scale              # px → cm
        pieces_cm: Dict[str, Piece] = {}
        for pid, p in self.pieces.items():
            q              = copy.copy(p)               # shallow copy is enough
            tx_px, ty_px   = p.translation
            q.translation  = (tx_px * scale, ty_px * scale)
            pieces_cm[pid] = q

        # 2. Feed those copies to the metric engine
        layout_cm    = Layout(pieces_cm)

        pe = PlacementEngine(layout_cm, self.container)
        pe.placed = list(layout_cm.order.values())      # mark everything “placed”

        util = pe.usage_BB()
        rest_length = pe.rest_length()
        concave_hull_usage = pe.concave_hull_utilization()
        concave_hull = pe.concave_hull_polygon
        self._draw_alpha_shape(concave_hull)
        # print(f"Calculated utilisation (current layout): {util:.2%}")
        # print(f"Calculated rest length (current layout): {rest_length:.2f} cm")
        # print(f"Calculated concave hull utilisation (current layout): {concave_hull_usage:.2%}")
        # update labels
        self.utilization_label.text = f"Utilization: {util:.2%}"
        self.rest_length_label.text = f"Rest length: {rest_length:.2f} cm"
        self.utilization_concave_label.text = f"Concave hull utilization: {concave_hull_usage:.2%}"

        return util


    def _handle_key(self, e: KeyEventArguments) -> None:
        if not e.action.keydown:      # ignore key‑up / repeats
            return
        if e.key == 'r':
            self._rotate_panel()
        elif e.key == 's':
            self._enable_selection_mode()

    # ---------------------------------------------------------------------------- #
    #                     PARAMETER CHANGE STUFF (NOT WORKING)                     #
    # ---------------------------------------------------------------------------- #
    def _filtered_design_tree(self) -> dict:
        dp_all = copy.deepcopy(self.design_params)          # never mutate source
        meta    = dp_all.get('meta', {})

        # 1 ───────────────────────── top-level meta filtering ──────────────────── #
        upper  = meta.get('upper',  {}).get('v')
        wb     = meta.get('wb',     {}).get('v')
        bottom = meta.get('bottom', {}).get('v')

        keep_top = {'meta'}                                 # always keep meta

        if wb   is not None: keep_top.add('waistband')
        if upper is not None: keep_top.update({'shirt', 'collar', 'sleeve', 'left'})

        bottom_map = {
            'SkirtCircle':      {'skirt'},
            'AsymmSkirtCircle': {'flare-skirt'},
            'GodetSkirt':       {'godet-skirt'},
            'PencilSkirt':      {'pencil-skirt'},
            'Skirt2':           {'skirt'},
            'SkirtManyPanels':  {'skirt-many-panels'},
            'SkirtLevels':      {'levels-skirt'},
            'Pants':            {'pants'},
        }
        keep_top.update(bottom_map.get(bottom, set()))

        dp = {k: v for k, v in dp_all.items() if k in keep_top}

        # 2 ────────────────────── sub-block “gate” filtering ───────────────────── #
        # Collar › component ------------------------------------------------------ #
        comp = dp.get('collar', {}).get('component')
        if comp and comp.get('style', {}).get('v') is None:
            for k in list(comp.keys()):
                print(f"Removing {k} from collar.component")
                comp.pop(k)

        # Sleeve block ----------------------------------------------------------- #
        sleeve = dp.get('sleeve')
        if sleeve:
            if sleeve.get('sleeveless', {}).get('v'):       # True  → hide sleeve details
                print("Hiding sleeve details")
                for k in list(sleeve.keys()):
                    print(f"Removing {k} from sleeve")
                    sleeve.pop(k)
            else:
                cuff = sleeve.get('cuff')
                if cuff and cuff.get('type', {}).get('v') is None:
                    for k in list(cuff.keys()):
                        cuff.pop(k)

        #print("dp:", dp)
        return dp


    # def _load_yaml(self, e: events.UploadEventArguments):
    #     import tempfile, yaml

    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
    #         tmp.write(e.content.read())
    #         tmp_path = tmp.name

    #     try:
    #         with open(tmp_path, "r", encoding="utf-8") as f:
    #             raw = yaml.safe_load(f) or {}

    #         def _flatten(node, prefix=""):
    #             flat = {}
    #             for k, v in node.items():
    #                 if isinstance(v, dict) and "v" not in v:
    #                     flat.update(_flatten(v, prefix + k + "."))
    #                 else:
    #                     flat[prefix + k] = v.get("v") if isinstance(v, dict) else v
    #             return flat

    #         self.style_params   = _flatten(raw)
    #         self.parameter_order = sorted(self.style_params)  # or any order you prefer
    #         self.yaml_loaded     = True
    #         ui.notify("YAML parameters loaded ✓", type="positive")

    #         # if self.pattern_loaded:
    #         #     self._refresh_sidebar()
    #     except Exception as exc:
    #         ui.notify(f"Could not load YAML: {exc}", type="negative")


    # def _on_param_change(self, param: str, e) -> None:
    #     try:
    #         new_val = float(e.value)
    #     except ValueError:
    #         new_val = e.value      # for string or bool parameters
    #     self.style_params[param] = new_val
    #     # ui.notify(f"{param} → {new_val}")


    # # TODO: recalculate panels
    # def _update_pattern_by_style(self):
    #     pass

    # ---------------------------------------------------------------------------- #
    #                                  TOOLBAR                                     #
    # ---------------------------------------------------------------------------- #

    def _update_dimensions(self, _):
        self.container_width_cm  = float(self.width_input.value or self.container_width_cm)
        self.container_height_cm = float(self.height_input.value or self.container_height_cm)
        self._update_scale_factors()

        self.container.width = self.container_width_cm
        self.container.height = self.container_height_cm

        self.scene.props(
            f'width="{self.container_width_px}" '
            f'height="{self.container_height}" '
            f'viewBox="0 0 {self.container_width_px} {self.container_height}"'
        )

        if self.pattern_loaded:
            self._draw_outlines()

        for piece in self.pieces.values():
            piece.translation = (0, 0)


    def _update_seam_allowance(self, _):
        self.seam_allowance_cm = float(self.sa_input.value or 0.0)
        if self.pattern_loaded:
            self._rebuild_panel_outlines()
            self._draw_outlines()

    # ---------------------------------------------------------------------------- #
    #                                PATTERN LOADERS                               #
    # ---------------------------------------------------------------------------- #
    
    
    def _load_pattern_core(self, path: Path) -> None:
        """
        Clear old state, load geometry + design parameters from *path*,
        and rebuild the sidebar.
        """
        # --- clear old GUI/scene state ---------------------------------
        self.panel_path_refs.clear()
        self.selected_panel = ""
        self.pieces.clear()
        self.drag_data.clear()
        self.scene.clear()

        # --- geometry --------------------------------------------------
        extractor = PatternPathExtractor(path)
        
        panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE,)
        # duplicate the pieces twice and add to pieces so we have 3 copies of each piece and add them to pieces
       
        # avoid modifying the original pieces
        self.pieces.update({
            f"{piece.id}": copy.deepcopy(piece) for piece in panel_pieces.values()
        })

        # split the first piece into two halves
        if self.pieces:
            first_id = next(iter(self.pieces))
            original_piece = self.pieces.pop(first_id)
            left_piece, right_piece = original_piece.split()
            self.pieces[left_piece.id] = left_piece
            self.pieces[right_piece.id] = right_piece

        self.layout = Layout(self.pieces)

        num_copies = config.NUM_COPIES
        for i in range(num_copies):
            # Add copies of each piece with updated ids
            for piece in panel_pieces.values():
                copy_piece = copy.deepcopy(piece)
                copy_piece.id = f"{piece.id}_copy{i+1}"
                self.pieces[copy_piece.id] = copy_piece

        print(f"Loaded {num_copies+1} copies, {len(self.pieces)} pieces in total.")

        # print all the pieces
        # for piece_id, piece in self.pieces.items():
        #     print(f"Loaded piece: {piece_id} with translation {piece.translation} and rotation {piece.rotation}")
        self._rebuild_panel_outlines()
        self.pattern_loaded = True
        self._draw_outlines()

        # --- design parameters ----------------------------------------
        with path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
            self.design_params = spec.get("design", {})
            print("Design parameters loaded:", self.design_params)

        # --- sidebar ---------------------------------------------------
        self._build_sidebar()          # see next section


    def _load_pattern(self, e: events.UploadEventArguments):
        """Handle file-upload event and delegate to the core loader."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(e.content.read())
            tmp_path = Path(tmp.name)

        try:
            self._load_pattern_core(tmp_path)
            ui.notify("Pattern loaded ✓", type="positive")
        except Exception as exc:
            ui.notify(f"Could not load pattern: {exc}", type="negative")
            raise                    # optional: keep traceback in server log


    def _load_default_pattern(self):
        """
        So I don't have to load something in every time I change code... 
        """
        default_path = config.DEFAULT_PATTERN_PATH
        self._load_pattern_core(Path(default_path))

    def _update_scale_factors(self):
        """
        Conversion from cm to pixels for display
        Uses the global canvas size to retain scale
        """
        self.effective_scale = min(
            MAX_CANVAS_PX_WIDTH  / self.container_width_cm,
            MAX_CANVAS_PX_HEIGHT / self.container_height_cm,
        )
        self.container_width_px  = int(self.container_width_cm  * self.effective_scale)
        self.container_height = int(self.container_height_cm * self.effective_scale)

    def _rebuild_panel_outlines(self):
        """
        Re‑computes the outer paths of every panel with the seam allowance.
        """
        sa = self.seam_allowance_cm
        for piece in self.pieces.values():
            #print("adding seam to: ", piece)
            piece.add_seam_allowance(sa)
            piece.translation = (0, 0)  # reset translation

        print ("Panel outlines rebuilt")


    # ---------------------------------------------------------------------------- #
    #                                  DRAWING                                     #
    # ---------------------------------------------------------------------------- #

    def _draw_panel(self, piece_id, fill="#bee5eb"):
        """
        Render a single pattern piece.

        Parameters
        ----------
        piece_id: Key in self.pieces
        fill : str, optional
            Fill colour for the outer path (defaults to blue)

        Notes
        -----
        * Does **not** clear the scene or update offsets.
        * Populates ``self.panel_path_refs[piece_id]``
        """

        piece = self.pieces[piece_id]

        # clear the piece's previous svg path
        if piece_id in self.panel_path_refs:
            outer_path, inner_path = self.panel_path_refs[piece_id]
            outer_path.delete()
            inner_path.delete()

        # Scale the geometry
        # outer_scaled = utils.scale(piece.get_outer_path(), self.effective_scale)
        # outer_scaled += outer_scaled[:1]                 # close path

        # inner_scaled = utils.scale(piece.get_inner_path(), self.effective_scale)
        # inner_scaled += inner_scaled[:1]                 # close path

        # # SVG path strings
        # outer_cmd = "M " + " L ".join(
        #     f"{x + piece.translation[0]} {y + piece.translation[1]}"
        #     for x, y in outer_scaled
        # )
        # inner_cmd = "M " + " L ".join(
        #     f"{x + piece.translation[0] + self.effective_scale * self.seam_allowance_cm} "
        #     f"{y + piece.translation[1] + self.effective_scale * self.seam_allowance_cm}"
        #     for x, y in inner_scaled
        # )

        outer_scaled = utils.scale(piece.get_outer_path(), self.effective_scale)
        outer_scaled += outer_scaled[:1]

        inner_scaled = utils.scale(piece.get_inner_path(), self.effective_scale)
        inner_scaled += inner_scaled[:1]

        outer_cmd = "M " + " L ".join(f"{x} {y}" for x, y in outer_scaled)
        # inner_cmd = "M " + " L ".join(f"{x} {y}" for x, y in inner_scaled)

        oxs, oys = zip(*outer_scaled)
        ixs, iys = zip(*inner_scaled)
        dx_px = (max(oxs) - min(oxs) - (max(ixs) - min(ixs))) / 2
        dy_px = (max(oys) - min(oys) - (max(iys) - min(iys))) / 2
        inner_cmd = "M " + " L ".join(
            f"{x + dx_px} {y + dy_px}"
            for x, y in inner_scaled
        )

        # Draw paths
        outer_path = self._svg_path_draggable(
            outer_cmd,
            panel_id=piece_id,
            stroke="#ed7ea7", stroke_width=2,
            fill=fill, fill_opacity=0.35,
        )
        inner_path = self._svg_path_static(
            inner_cmd,
            stroke="#4b5563", stroke_width=1, stroke_dash="4 2",
        )

        tx, ty = piece.translation           # already in CSS‑pixels
        for path in (outer_path, inner_path):
            path.props(f'transform="translate({tx},{ty})"')

        self.panel_path_refs[piece_id] = (outer_path, inner_path)


    def _draw_outlines(self):
        """Clear the scene and redraw **all** panels."""
        # Reset scene
        self.scene.clear()
        self.hull_path = None

        all_px = []
        for piece in self.pieces.values():
            piece.scale = self.effective_scale
            all_px.extend(utils.scale(piece.get_outer_path(), self.effective_scale))

        xs, ys = zip(*all_px)
        # pattern_w, pattern_h = max(xs) - min(xs), max(ys) - min(ys)

        self.offset_px = (0, 0)

        # Border
        for x1, y1, x2, y2 in [
            (0, 0, self.container_width_px, 0),
            (self.container_width_px, 0, self.container_width_px, self.container_height),
            (self.container_width_px, self.container_height, 0, self.container_height),
            (0, self.container_height, 0, 0),
        ]:
            self._svg_line(x1, y1, x2, y2, stroke="#8a8a8a")

        
        self.panel_path_refs.clear()                            
        fills = itertools.cycle(["#c3e6cb", "#bee5eb", "#ffeeba", "#f5c6cb"])

        for piece_id, piece in self.pieces.items():
            self._draw_panel(piece_id, fill=next(fills))



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
        element.on("click", lambda e, p=panel_id: self._handle_panel_click(p))
        return element


    def _draw_alpha_shape(self, polygon: Polygon, stroke: str = "#ff0000"):
        if polygon.is_empty:
            return

        # 1) Delete previous hull if present
        if self.hull_path is not None:
            self.hull_path.delete()
            self.hull_path = None

        # 2) Build the 'd' string in px
        coords_cm = list(polygon.exterior.coords)
        coords_px = [(x*self.effective_scale, y*self.effective_scale) for x,y in coords_cm]
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x,y in coords_px) + " Z"

        # 3) Append new <path> and keep a handle to it
        with self.scene:
            self.hull_path = ui.element("path").props(
                f'd="{d}" '
                f'stroke="{stroke}" stroke-width="2" '
                f'fill="none" pointer-events="none"'
            )

    # ---------------------------------------------------------------------------- #
    #               click and drag stuff                                           #
    # ---------------------------------------------------------------------------- #

    def _on_drag_start(self, e, panel_id: str):
        piece = self.pieces[panel_id]
        self.drag_data = {
            'panel_id': panel_id,
            'start_x': e.args.get('clientX', 0),
            'start_y': e.args.get('clientY', 0),
            'orig_offset': piece.translation,  # Use the piece's translation
        }
       # print(f"Drag started for {panel_id} at ({self.drag_data['start_x']}, {self.drag_data['start_y']})")

    def _on_drag_move(self, e, panel_id: str):
        if not self.drag_data or self.drag_data['panel_id'] != panel_id:
            return
        
        dx = e.args.get('clientX', 0) - self.drag_data['start_x']
        dy = e.args.get('clientY', 0) - self.drag_data['start_y']
        orig_x, orig_y = self.drag_data['orig_offset']
        new_offset = (orig_x + dx, orig_y + dy)
        self.pieces[panel_id].translation = new_offset
        outer_path, inner_path = self.panel_path_refs[panel_id]
        for path in (outer_path, inner_path):
            path.props(f'transform="translate({new_offset[0]},{new_offset[1]})"').update()
            # path.props(f'transform="translate({new_offset[0]},{new_offset[1]})"').update()

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
    def _check_intersections(self) -> bool:

        print("Checking intersections...")

        if not self.pattern_loaded:
            ui.notify("Load a pattern first", type="warning")
            return

        ox, oy = self.offset_px
        overlaps: List[Tuple[str, str]] = []
        names = list(self.pieces.keys())

        def to_px(path_cm, dx, dy):
            return [(x * self.effective_scale + ox + dx,
                     y * self.effective_scale + oy + dy) for x, y in path_cm]

        # Check for intersections between panels.
        for i, name_i in enumerate(names):
            dx_i, dy_i = self.pieces[name_i].translation
            poly_i = to_px(self.pieces[name_i].get_outer_path(), dx_i, dy_i)
            for name_j in names[i+1:]:
                dx_j, dy_j = self.pieces[name_j].translation
                poly_j = to_px(self.pieces[name_j].get_outer_path(), dx_j, dy_j)
                if utils.polygons_overlap(poly_i, poly_j):
                    overlaps.append((name_i, name_j))

        # Check for panels that extend outside the container bounds.
        panels_outside = []
        for name in names:
            dx, dy = self.pieces[name].translation
            poly = to_px(self.pieces[name].get_outer_path(), dx, dy)
            xs_poly, ys_poly = zip(*poly)
            if min(xs_poly) < 0 or max(xs_poly) > self.container_width_px or min(ys_poly) < 0 or max(ys_poly) > self.container_height:
                panels_outside.append(name)

        # Build the notification message.
        messages = []

        if overlaps:
            #messages.append(f"Intersections found: {overlaps}")
            messages.append("Overlaps:\n" + "\n".join(f"• {a} x {b}" for a, b in overlaps))
            #ui.notify("\n".join(messages), multi_line=True)
        else:
            messages.append("No intersections")
        if panels_outside:
            messages.append("Panels outside container:\n" + "\n".join(f"• {p}" for p in panels_outside))

        else:
            messages.append("All panels within container")

        ui.notify("\n".join(messages), multi_line=True)
        return bool(overlaps or panels_outside)

    # ---------------------------------------------------------------------------- #
    #                                    DECODER                                   #
    # ---------------------------------------------------------------------------- #

    # def _apply_placements(self, placements_cm: list[tuple[int, float, float]]):
    async def _apply_placements(self, placements_cm: list[tuple[int, float, float]]):
        """
        placements_cm: list of (piece_id, dx_cm, dy_cm)
        It rewrites the translation of each piece and updates the SVG
        paths that were created by _draw_outlines().
        """
        if not self.pattern_loaded:
            ui.notify("Load a pattern first", type="warning")
            return
        
        # print("Drawing outlines")

        # self._draw_outlines()

        # print("Outlines drawn")
        # print(len(self.pieces), "pieces loaded")

        cm_to_px = self.effective_scale          # 1 cm → this many CSS‑pixels
        for name, dx_cm, dy_cm, rotation in placements_cm:
            if name not in self.panel_path_refs:
                print(f"Unknown panel '{name}'")
                ui.notify(f"Unknown panel '{name}'", type="warning")
                continue
            
            # print(f"Placing {name} at ({dx_cm:.2f}, {dy_cm:.2f}) cm")
            dx_px = dx_cm * cm_to_px
            dy_px = dy_cm * cm_to_px
            # self.panel_transforms[name] = (dx_px, dy_px)

            piece = self.pieces[name]
            piece.translation = (dx_px, dy_px)  # update translation in cm
            #piece.rotate(rotation)
            outer, inner = self.panel_path_refs[name]
            #outer = piece.get_outer_path()
            #inner = piece.get_inner_path()

            if piece.rotation != rotation:               # avoid double‑rotating
                piece.rotate((rotation - piece.rotation) % 360)

            self._draw_panel(piece.id)
            for path in (outer, inner):
                # print(rotation)
                # path.props(f'transform="translate({dx_px},{dy_px}),rotate({rotation})"').update()
                path.props(f'transform="translate({dx_px},{dy_px})"').update()

        ui.notify("Automatic placement applied", type="positive")

    async def _auto_place(self, method="BL"):
        if not self.pattern_loaded:
            ui.notify('Load a pattern first', type='warning')
            return
        
        print(f"Auto placing with method: {method}")

        try:
            #layout = self.layout
            layout = copy.deepcopy(self.layout)
            container = self.container

            print (f"Container: {container}")
            print (f"Layout: {layout}")

            if method == "BL":
                # Bottom-Left placement
                decoder = BottomLeftDecoder(layout, container, step=config.GRAVITATE_STEP)
            elif method == "Greedy":
                # Greedy placement
                decoder = GreedyBLDecoder(layout, container)   
            elif method == "NFP":
                decoder = NFPDecoder(layout, container)
            elif method == "Random Order BL":
                decoder = RandomDecoder(layout, container)
            elif method == "Genetic Algorithm":
                evo = Evolution(
                    self.pieces,
                    container,
                    num_generations=config.NUM_GENERATIONS,
                    population_size=config.POPULATION_SIZE,
                    mutation_rate=config.MUTATION_RATE,
                    enable_dynamic_stopping=config.ENABLE_DYNAMIC_STOPPING,
                    early_stop_window=config.EARLY_STOP_WINDOW,
                    early_stop_tolerance=config.EARLY_STOP_TOLERANCE,
                    enable_extension=config.ENABLE_EXTENSION,
                    extend_window=config.EXTEND_WINDOW,
                    extend_threshold=config.EXTEND_THRESHOLD,
                    max_generations=config.MAX_GENERATIONS,
                    crossover_method= config.SELECTED_CROSSOVER,
                )
                best_chromosome = evo.run()
                if best_chromosome is None:
                    ui.notify("No valid solution found by the Genetic Algorithm", type="negative")
                    return
                view    = LayoutView(best_chromosome.genes)
                decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)
  
            else:
                raise ValueError(f"Unknown placement method: {method}")
            
            print("Now decoding...")
            placements = decoder.decode()  # [(name, dx, dy)]
            print("Decoding done")

            utilization = decoder.usage_BB()

            print(f"Utilization: {utilization:.2%}")
            rest_length = decoder.rest_length()
            print(f"Rest length: {rest_length:.2f} cm")
            
            # print(f"Concave hull utilization: {concave_hull_usage:.2%}")

            # print(f"Auto placement ({method}) usage:")
            
            self.utilization_label.text = f"Utilization: {utilization:.2%}"
            self.rest_length_label.text = f"Rest length: {rest_length:.2f} cm"

            print(f"Auto placement ({method}) usage: {utilization:.2%}")
            print(f"Rest length: {rest_length:.2f} cm")

            # print placements
            # for name, dx, dy, rot in placements:
            #      print(f"Placing {name} at ({dx:.2f}, {dy:.2f}) cm with rotation {rot}")

            await self._apply_placements(placements)
            concave_hull_usage = decoder.concave_hull_utilization()
            self._draw_alpha_shape(decoder._last_hull, stroke="#ff0000")

            self.utilization_concave_label.text = f"Concave hull utilization: {concave_hull_usage:.2%}"

            ui.notify('Auto placement completed ', type='positive')

            # if the BL runs out of space and there are intersections, still draw it but notify the user
            if self._check_intersections():
                ui.notify("Auto placement has intersections", type="negative")
                print("Auto placement has intersections")

        except Exception as exc:
            ui.notify(f'Auto placement failed: {exc}', type='negative')

    def _enable_selection_mode(self):
        self.selection_mode = True
        ui.notify("Select a panel by clicking on it", type="info")

    # Called on clicking a panel while selection mode is enabled.
    def _handle_panel_click(self, panel_id: str):
        if self.selection_mode:
            self._select_panel(panel_id)
            # Optionally, disable selection mode after selection.
            self.selection_mode = False

    # Updates selected panel appearance.
    def _select_panel(self, panel_id: str):
        # Revert previously selected panel to its original style.
        if self.selected_panel and self.selected_panel in self.panel_path_refs:
            outer, _ = self.panel_path_refs[self.selected_panel]
            # Reset stroke to the default
            outer.props('stroke="#ed7ea7"')
        # Set new selection.
        self.selected_panel = panel_id
        if panel_id in self.panel_path_refs:
            outer, _ = self.panel_path_refs[panel_id]
            # Change stroke to red to indicate selection.
            outer.props('stroke="#FF0000"')
            ui.notify(f"Panel '{panel_id}' selected", type="info")

    def _rotate_panel(self):
        if not self.selected_panel:
            ui.notify("No panel selected", type="warning")
            return

        # Rotate the selected panel by 90 degrees.
        piece = self.pieces[self.selected_panel]
        piece.rotate(90)
        
        # refresh the outlines
        self._draw_panel(self.selected_panel)
        # self.selected_panel = ""

        ui.notify(f"Panel '{self.selected_panel}' rotated", type="info")

    def _reset_rotations(self):
        for piece in self.pieces.values():
            piece.reset_rotation()
        self._draw_outlines()
        ui.notify("All panel rotations reset", type="positive")

if __name__ in {"__main__", "__mp_main__"}:
    # run_gui.py
    
    gui = NestingGUI(pattern_path=None)
    ui.run(port=8082)
