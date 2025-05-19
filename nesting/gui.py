from __future__ import annotations
import itertools
import tempfile
import copy
import yaml
from typing import Dict, List, Tuple

from nicegui import ui, events
from nicegui.events import KeyEventArguments
from nesting import utils
from nesting.evolution import Evolution  # add_seam_allowance, polygons_overlap, etc.
from .path_extractor import *
from .layout import *
from .placement_engine import *

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
        self.container_width_cm  = 400.0
        self.container_height_cm = 100.0

        self._update_scale_factors()

        # seam allowance in cm
        self.seam_allowance_cm: float = 1.0

        # geometry stores (in cm)
        #self.raw_panel_outlines: Dict[str, List[List[float]]] = {}  # as read from JSON
        #self.panel_outlines:     Dict[str, List[List[float]]] = {}   # after seam allowance
        #self.offset_px: Tuple[float, float] = (0.0, 0.0)            # for centering
        #self.panel_rotations: Dict[str, float] = {}  # for rotation

        self.pieces : Dict[str, Piece] = {}  # for storing pieces

        # UI drag bookkeeping
        self.panel_path_refs: Dict[str, Tuple[any, any]] = {}
        # self.panel_transforms: Dict[str, Tuple[float, float]] = {}
        self.drag_data: Dict = {}

        self.pattern_loaded = False
        self.yaml_loaded     = False

        self.style_params: Dict = {}
        self.parameter_order: List[str] = []

        # Currently selected panel for style editing.
        self.selected_panel: str = ""
        self.selection_mode: bool = False

        self._build_layout()
        self._load_default_pattern()  # auto load default pattern

    # ---------------------------------------------------------------------------- #
    #                                  UI Builders                                 #
    # ---------------------------------------------------------------------------- #

    def _build_layout(self) -> None:
        # Layout with canvas/toolbar on the left and a style sidebar on the right.
        with ui.row().classes("w-full h-screen"):
            with ui.column().classes("flex-1"):
                self._build_toolbar()
                self._build_canvas()

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
            

            ui.upload(label="Load YAML parameters",
                    on_upload=self._load_yaml,
                    auto_upload=True,
                    multiple=False)

            
            ui.button("Auto place (Bottom‑Left)", on_click=self._auto_place)
            ui.button("Auto place (Greedy)", on_click=lambda _: self._auto_place("Greedy"))
            ui.button("Auto place (NFP)", on_click=lambda _: self._auto_place("NFP"))
            ui.button("Random Order (BL)", on_click=lambda _: self._auto_place("Random Order BL"))
            ui.button("Genetic Algorithm", on_click=lambda _: self._auto_place("Genetic Algorithm"))

            # Button to enable selection mode.
            ui.button("Select Panel (S)", on_click=self._enable_selection_mode)
            ui.button("Reset Selection", on_click=lambda _: self._select_panel(""))
            ui.button("Rotate Panel (R)", on_click=lambda _: self._rotate_panel())

        self._kb = ui.keyboard(on_key=self._handle_key, active=True)
        
    def _handle_key(self, e: KeyEventArguments) -> None:
        if not e.action.keydown:      # ignore key‑up / repeats
            return
        if e.key == 'r':
            self._rotate_panel()
        elif e.key == 's':
            self._enable_selection_mode()
                
    # def randomize_order_and_autoplace(self):
    #     """
    #     Randomizes the order of the pieces and then calls the
    #     Bottom‑Left auto placement.
    #     """
    #     from collections import OrderedDict
    #     import random

    #     piece_ids = list(self.pieces.keys())
    #     random.shuffle(piece_ids)
    #     new_order = OrderedDict((pid, self.pieces[pid]) for pid in piece_ids)
    #     self.pieces = new_order
    #     ui.notify("Random order applied; now auto placing...", type="positive")

    # ---------------------------------------------------------------------------- #
    #                     PARAMETER CHANGE STUFF (NOT WORKING)                     #
    # ---------------------------------------------------------------------------- #

    def _load_yaml(self, e: events.UploadEventArguments):
        import tempfile, yaml, os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
            tmp.write(e.content.read())
            tmp_path = tmp.name

        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            def _flatten(node, prefix=""):
                flat = {}
                for k, v in node.items():
                    if isinstance(v, dict) and "v" not in v:
                        flat.update(_flatten(v, prefix + k + "."))
                    else:
                        flat[prefix + k] = v.get("v") if isinstance(v, dict) else v
                return flat

            self.style_params   = _flatten(raw)
            self.parameter_order = sorted(self.style_params)  # or any order you prefer
            self.yaml_loaded     = True
            ui.notify("YAML parameters loaded ✓", type="positive")

            # if self.pattern_loaded:
            #     self._refresh_sidebar()
        except Exception as exc:
            ui.notify(f"Could not load YAML: {exc}", type="negative")


    def _on_param_change(self, param: str, e) -> None:
        try:
            new_val = float(e.value)
        except ValueError:
            new_val = e.value      # for string or bool parameters
        self.style_params[param] = new_val
        # ui.notify(f"{param} → {new_val}")


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
            f'width="{self.container_width_px}" '
            f'height="{self.container_height}" '
            f'viewBox="0 0 {self.container_width_px} {self.container_height}"'
        )
        if self.pattern_loaded:
            self._draw_outlines()

        # reset the translation of all pieces
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
    

    def _load_pattern(self, e: events.UploadEventArguments):
        """
        Loads pattern from the uploaded JSON file, populates necessary fields and draws its outlines
        """
        # self.raw_panel_outlines.clear()
        # self.panel_outlines.clear()
        self.panel_path_refs.clear()
        # self.panel_transforms.clear()
        self.selected_panel = ""
        self.pieces.clear()
        self.drag_data.clear()
        self.scene.clear()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(e.content.read())
            tmp_path = tmp.name

        try:
            extractor = PatternPathExtractor(tmp_path)  # outlines in cm
            self.pieces = extractor.get_all_panel_pieces(samples_per_edge=7)

            # ui.notify("Panels found: " + ", ".join(self.raw_panel_outlines.keys()))
            self._rebuild_panel_outlines()
            self.pattern_loaded = True

            self._draw_outlines()
            # for(name, _) in self.panel_outlines.items():
            #     self.panel_rotations[name] = 0.0
            #     print(f"Panel {name} rotation: {self.panel_rotations[name]}")
            
            ui.notify("Pattern loaded ✓", type="positive")
        except Exception as exc:
            ui.notify(f"Could not load pattern: {exc}", type="negative")

    def _load_default_pattern(self):
        """
        So I don't have to load something in every time I change code... 
        """
        default_path = "/Users/aysegulbarlas/codestuff/GarmentCode/nesting-assets/Configured_design_specification_asym_dress.json"
        
        try:
            extractor = PatternPathExtractor(default_path)  # outlines in cm
            self.pieces = extractor.get_all_panel_pieces(samples_per_edge=7)
            print("Default pattern loaded: ", self.pieces)
            
            # print (f"Default pattern loaded: {self.pieces}")

            # for name, piece in self.pieces.items():
            #     self.panel_transforms[name] = (0, 0)
            #     self.panel_rotations[name] = 0.0
                # print(f"Panel {name} rotation: {self.panel_rotations[name]}")
        

            # ui.notify("Panels found: " + ", ".join(self.raw_panel_outlines.keys()))
            self._rebuild_panel_outlines()
            self.pattern_loaded = True
            self._draw_outlines()

            ui.notify("Default pattern loaded ✓", type="positive")
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
        self.container_width_px  = int(self.container_width_cm  * self.effective_scale)
        self.container_height = int(self.container_height_cm * self.effective_scale)

    def _rebuild_panel_outlines(self):
        """
        Re‑computes the outer paths of every panel with the seam allowance.
        """
        sa = self.seam_allowance_cm
        for piece in self.pieces.values():
            print("adding seam to: ", piece)
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
        print(f"Drag started for {panel_id} at ({self.drag_data['start_x']}, {self.drag_data['start_y']})")

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
            messages.append("Overlaps:\n" + "\n".join(f"• {a} × {b}" for a, b in overlaps))
            return True
        else:
            messages.append("No intersections")
        if panels_outside:
            messages.append("Panels outside container:\n" + "\n".join(f"• {p}" for p in panels_outside))
            return True
        else:
            messages.append("All panels within container")

        ui.notify("\n".join(messages), multi_line=True)
        return False

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
    # def _auto_place(self, method="BL"):
        if not self.pattern_loaded:
            ui.notify('Load a pattern first', type='warning')
            return
        
        print(f"Auto placing with method: {method}")

        try:
            layout = Layout(self.pieces)
            container = Container(self.container_width_cm,
                                  self.container_height_cm)
            
            print (f"Container: {container}")
            print (f"Layout: {layout}")

            if method == "BL":
                # Bottom-Left placement
                decoder = BottomLeftDecoder(layout, container, step=1.0)
                # self.panel_rotations = {name: 0 for name in self.panel_outlines.keys()}
            elif method == "Greedy":
                # Greedy placement
                decoder = GreedyBLDecoder(layout, container)
                # self.panel_rotations = {name: 0 for name in self.panel_outlines.keys()}
            elif method == "NFP":
                decoder = NFPDecoder(layout, container)
                # self.panel_rotations = {name: 0 for name in self.panel_outlines.keys()}
            elif method == "Random Order BL":
                # Random‑order Bottom‑Left placement
                decoder = RandomDecoder(layout, container)
            elif method == "Genetic Algorithm":
                evo = Evolution(
                    self.pieces,
                    container,
                    num_generations=20,
                    population_size=10,
                    elite_population_size=5,
                    mutation_rate=0.1,
                    pmx=True,
                    allow_duplicate_genes=True,
                    early_stop_tolerance=0.01,
                    early_stop_window=5,
                )
                # evo.generate_population()
                best_chromosome = evo.run()
                decoder = BottomLeftDecoder(best_chromosome, container)
            else:
                raise ValueError(f"Unknown placement method: {method}")
            
            print("Now decoding...")
            placements = decoder.decode()  # [(name, dx, dy)]
            print("Decoding done")
            

            utilization = decoder.usage_BB()
            
            print(f"Utilization: {utilization:.2%}")
            rest_length = decoder.rest_length()
            print(f"Rest length: {rest_length:.2f} cm")
            try:
                concave_hull_usage = decoder.concave_hull_utilization()
            except Exception as err:
                ui.notify(f"Concave hull utilization skipped: {err}", type="warning")
                concave_hull_usage = "n/a"

            # print(f"Concave hull utilization: {concave_hull_usage:.2%}")

            # print(f"Auto placement ({method}) usage:")
            
            self.utilization_label.text = f"Utilization: {utilization:.2%}"
            self.rest_length_label.text = f"Rest length: {rest_length:.2f} cm"
            if concave_hull_usage != "n/a":
                self.utilization_concave_label.text = f"Concave hull utilization: {concave_hull_usage:.2%}"

            print(f"Auto placement ({method}) usage: {utilization:.2%}")
            print(f"Rest length: {rest_length:.2f} cm")

            # print placements
            for name, dx, dy, rot in placements:
                 print(f"Placing {name} at ({dx:.2f}, {dy:.2f}) cm with rotation {rot}")

            await self._apply_placements(placements)

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


if __name__ in {"__main__"}:

    NestingGUI()
    ui.run(port=8080)