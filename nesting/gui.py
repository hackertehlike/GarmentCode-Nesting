from __future__ import annotations
import itertools
import tempfile
import json
import copy
import traceback
# import yaml
# from typing import Dict, List, Tuple, Set, Optional
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from datetime import datetime

from nicegui import ui, events
from nicegui.events import KeyEventArguments
from nesting import utils
from nesting import config
from nesting.evolution import Evolution  # add_seam_allowance, polygons_overlap, etc.
from nesting.simulated_annealing import SimulatedAnnealing
import matplotlib.pyplot as plt
# import io
# import base64

from pygarment.garmentcode.params import DesignSampler
from .path_extractor import *
from .layout import *
from .placement_engine import *
# from .pattern_loader import load_pattern_bundle  # added import
from nesting.metastatistics import MetaStatistics

# from nesting.panel_mapping import affected_panels, select_genes
# from shapely.errors import GEOSException as TopologyException

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

    def __init__(self, pattern_path: str | Path | None = None, use_default_params: bool = False) -> None:
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
        # Ensure these attributes always exist before any optional creation/assignment paths
        self.body_params = None
        self.design_sampler = None
        self.meta_garment = None  # keeps MetaGarment instance for incremental operations

        # Currently selected panel for style editing.
        self.selected_panel: str = ""
        self.selection_mode: bool = False

        # Pattern regeneration path
        self.pattern_path: Optional[Path] = None
        if pattern_path:
            self.pattern_path = Path(pattern_path)
            
        # Flag to determine if we should use default parameters
        self.use_default_params = use_default_params

        self.hull_path: ui.element | None = None

        # GA parameters (initialize with config defaults)
        self.ga_population_size = config.POPULATION_SIZE
        self.ga_num_generations = config.NUM_GENERATIONS
        self.ga_mutation_rate = config.MUTATION_RATE
        # Initialize population weights with config defaults
        self.ga_population_weights = config.POPULATION_WEIGHTS.copy()
        # Initialize allowed rotations with config defaults
        self.ga_allowed_rotations = config.ALLOWED_ROTATIONS.copy()

        self._build_layout()

        if pattern_path is not None:
            # user supplied a pattern -> load it once
            self._load_pattern_core(Path(pattern_path))
        else:
            # fall back to the previous hard-coded default
            self._load_default_pattern()


    # ---------------------------------------------------------------------------- #
    #                                  UI Builders                                 #
    # ---------------------------------------------------------------------------- #

    def _build_layout(self) -> None:
        # Three-column layout: parameters on left, canvas in center, toolbar on right
        with ui.row().classes("w-full h-screen"):
            # Left column for parameters
            with ui.column().classes("w-1/5 h-full overflow-y-auto p-4 border-r"):
                self._build_sidebar()
            
            # Center column for canvas
            with ui.column().classes("flex-1 h-full"):
                self._build_canvas()
            
            # Right column for toolbar/buttons
            with ui.column().classes("w-1/5 h-full overflow-y-auto p-4 border-l"):
                self._build_toolbar()


    

    def _build_sidebar(self) -> None:
        """Build the sidebar with style parameters."""

        def _iter_all_params(node: dict, prefix: str = ''):
            """Yield all leaf parameters for display, with their types."""
            for key, value in node.items():
                if isinstance(value, dict) and 'v' in value:
                    param_type = value.get('type', 'unknown')
                    if value['v'] is not None:
                        yield prefix + key, value, param_type
                elif isinstance(value, dict):
                    yield from _iter_all_params(value, prefix + key + '.')
                    
        if not hasattr(self, "sidebar"):
            self.sidebar = ui.column().classes("w-full h-full overflow-y-auto")

        self.sidebar.clear()

        with self.sidebar:
            ui.label("Style Parameters").classes("text-xl font-bold mb-4")
            
            # Only build sidebar content if we have design parameters or pieces loaded
            if not self.design_params and not self.pieces:
                ui.label("Load a pattern to see parameters").classes("text-gray-500 italic")
                return
                
            filtered = self._filtered_design_tree()  # filter out irrelevant blocks

            # Display read-only meta parameters
            meta_params = filtered.get("meta", {})
            if meta_params:
                with ui.expansion("Meta", value=True).classes("w-full mb-2"):
                    for m_name, m_spec in meta_params.items():
                        ui.label(f"{m_name.replace('_', ' ').capitalize()}: {m_spec.get('v')}")

            # Remove meta block for editable parameters
            filtered_numeric = {k: v for k, v in filtered.items() if k != "meta"}

            # Group parameters into collapsible sections
            sections = {}
            for name, spec, param_type in _iter_all_params(filtered_numeric):
                section_name = name.split('.')[0] if '.' in name else 'General'
                if section_name not in sections:
                    sections[section_name] = []
                sections[section_name].append((name, spec, param_type))
            
            # Create each section
            for section_name, params in sections.items():
                with ui.expansion(section_name, value=True).classes("w-full mb-2"):
                    for name, spec, param_type in params:
                        display_name = name.split('.')[-1] if '.' in name else name
                        val = spec['v']
                        
                        # Different UI elements based on parameter type
                        if param_type in ('float', 'int'):
                            # Editable numeric parameters
                            ui.number(
                                value=val,
                                label=display_name,
                                on_change=lambda e, n=name: self._on_param_change(n, e),
                            )
                        elif param_type == 'bool':
                            # Read-only boolean parameters with appropriate style
                            bool_val = "Yes" if val else "No"
                            ui.label(f"{display_name}: {bool_val}").classes("text-sm px-2 py-1 bg-gray-100 rounded")
                        else:
                            # Read-only categorical/string parameters with appropriate style
                            ui.label(f"{display_name}: {val}").classes("text-sm px-2 py-1 bg-gray-100 rounded")


    def _build_canvas(self) -> None:
        with ui.element("div").classes("w-full h-full flex items-center justify-center"):
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
        # Title for the toolbar
        ui.label("Tools").classes("text-xl font-bold mb-4")
        
        # Container dimensions section
        with ui.card().classes("w-full mb-4 p-2"):
            ui.label("Container Size").classes("font-bold mb-2")
            with ui.column().classes("gap-2"):
                self.width_input = ui.number("Width (cm)", value=self.container_width_cm, on_change=self._update_dimensions).classes("w-full")
                self.height_input = ui.number("Height (cm)", value=self.container_height_cm, on_change=self._update_dimensions).classes("w-full")
                self.sa_input = ui.number("Seam Allowance (cm)", value=self.seam_allowance_cm, on_change=self._update_seam_allowance).classes("w-full")
        
        # File operations
        with ui.card().classes("w-full mb-4 p-2"):
            ui.label("File Operations").classes("font-bold mb-2")
            ui.upload(label="Load Pattern", on_upload=self._load_pattern, auto_upload=True).classes("w-full")
        
        # Placement Methods
        with ui.card().classes("w-full mb-4 p-2"):
            ui.label("Placement Methods").classes("font-bold mb-2")
            with ui.column().classes("gap-1 w-full"):
                ui.button("Bottom-Left", on_click=self._auto_place).classes("w-full")
                # ui.button("Greedy", on_click=lambda _: self._auto_place("Greedy")).classes("w-full")
                ui.button("NFP", on_click=lambda _: self._auto_place("NFP")).classes("w-full")
                # ui.button("BLF", on_click=lambda _: self._auto_place("BLF")).classes("w-full")
                # ui.button("TOPOS", on_click=lambda _: self._auto_place("TOPOS")).classes("w-full")
                # ui.button("Random BL", on_click=lambda _: self._auto_place("RandomBL")).classes("w-full")
                # ui.button("Random NFP", on_click=lambda _: self._auto_place("RandomNFP")).classes("w-full")
                ui.button("Genetic Algorithm", on_click=lambda _: self._run_genetic_algorithm()).classes("w-full")
                ui.button("Simulated Annealing", on_click=lambda _: self._run_simulated_annealing()).classes("w-full")
                ui.button("2-Exchange Local Search", on_click=lambda _: self._run_two_exchange()).classes("w-full")
                ui.button("Random Search", on_click=lambda _: self._run_random_search()).classes("w-full")
                ui.button("Random → SA", on_click=lambda _: self._run_random_then_sa()).classes("w-full")
                ui.button("Random → 2-Exchange", on_click=lambda _: self._run_random_then_two_exchange()).classes("w-full")
        
        # GA Parameters
        with ui.card().classes("w-full mb-4 p-2"):
            ui.label("GA Parameters").classes("font-bold mb-2")
            with ui.column().classes("gap-2"):
                self.ga_pop_size_input = ui.number("Population Size", value=self.ga_population_size, 
                                                   on_change=self._update_ga_params).classes("w-full")
                self.ga_generations_input = ui.number("Generations", value=self.ga_num_generations, 
                                                      on_change=self._update_ga_params).classes("w-full")
                self.ga_mutation_rate_input = ui.number("Mutation Rate", value=self.ga_mutation_rate, 
                                                        min=0.0, max=1.0, step=0.01,
                                                        on_change=self._update_ga_params).classes("w-full")
                
                # Population composition weights
                ui.label("Population Weights").classes("font-semibold mt-2")
                self.ga_elite_weight = ui.number("Elites", value=self.ga_population_weights['elites'], 
                                                 min=0.0, max=1.0, step=0.05,
                                                 on_change=self._update_ga_weights).classes("w-full")
                self.ga_offspring_weight = ui.number("Offspring", value=self.ga_population_weights['offspring'], 
                                                     min=0.0, max=1.0, step=0.05,
                                                     on_change=self._update_ga_weights).classes("w-full")
                self.ga_mutant_weight = ui.number("Mutants", value=self.ga_population_weights['mutants'], 
                                                  min=0.0, max=1.0, step=0.05,
                                                  on_change=self._update_ga_weights).classes("w-full")
                self.ga_random_weight = ui.number("Randoms", value=self.ga_population_weights['randoms'], 
                                                  min=0.0, max=1.0, step=0.05,
                                                  on_change=self._update_ga_weights).classes("w-full")
                
                # Allowed rotations
                ui.label("Allowed Rotations (comma-separated)").classes("font-semibold mt-2")
                rotations_str = ",".join(map(str, self.ga_allowed_rotations))
                self.ga_rotations_input = ui.input("Rotations", value=rotations_str, 
                                                   on_change=self._update_ga_rotations).classes("w-full")
        
        # Piece Operations
        with ui.card().classes("w-full mb-4 p-2"):
            ui.label("Piece Operations").classes("font-bold mb-2")
            with ui.column().classes("gap-1 w-full"):
                ui.button("Select (S)", on_click=self._enable_selection_mode).classes("w-full")
                ui.button("Reset Selection", on_click=lambda _: self._select_panel("")).classes("w-full")
                ui.button("Rotate (R)", on_click=lambda _: self._rotate_panel()).classes("w-full")
                ui.button("Reset Rotation", on_click=lambda _: self._reset_rotations()).classes("w-full")
                ui.button("Split", on_click=lambda _: self._split_panel()).classes("w-full")
        
        # Analysis Operations
        with ui.card().classes("w-full mb-4 p-2"):
            ui.label("Analysis").classes("font-bold mb-2")
            with ui.column().classes("gap-1 w-full"):
                ui.button("Check Intersections", on_click=self._check_intersections).classes("w-full")
                ui.button("Calculate Usage", on_click=self._calculate_usage).classes("w-full")
                ui.button("Run Heuristics", on_click=self._run_heuristics).classes("w-full")
            
            # Metrics display
            with ui.card().classes("w-full mt-2 p-2 bg-gray-100"):
                with ui.column().classes("gap-1"):
                    self.utilization_label = ui.label("Utilization: n/a")
                    self.rest_length_label = ui.label("Rest length: n/a")
                    self.utilization_concave_label = ui.label("Hull utilization: n/a")
                    self.rest_height_label = ui.label("Rest height: n/a")
        
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
        rest_height = pe.rest_height()
        concave_hull_usage = pe.concave_hull_utilization()
        concave_hull = pe._last_hull
        self._draw_alpha_shape(concave_hull)
        # print(f"Calculated utilisation (current layout): {util:.2%}")
        # print(f"Calculated rest length (current layout): {rest_length:.2f} cm")
        # print(f"Calculated concave hull utilisation (current layout): {concave_hull_usage:.2%}")
        # update labels
        self.utilization_label.text = f"Utilization: {util:.2%}"
        self.rest_length_label.text = f"Rest length: {rest_length:.2f} cm"
        self.rest_height_label.text = f"Rest height: {rest_height:.2f} cm"
        self.utilization_concave_label.text = f"Concave hull utilization: {concave_hull_usage:.2%}"

        return util

    def _update_metrics_from_decoder(self, decoder: "PlacementEngine"):
        """Update all metric labels using a decoder after placements.
        Safely handles failures so UI labels don't stay 'n/a'."""
        try:
            util = decoder.usage_BB()
        except Exception as e:
            print(f"[GUI][METRICS] usage_BB failed: {e}")
            util = None
        try:
            rest_len = decoder.rest_length()
        except Exception as e:
            print(f"[GUI][METRICS] rest_length failed: {e}")
            rest_len = None
        try:
            rest_ht = decoder.rest_height()
        except Exception as e:
            print(f"[GUI][METRICS] rest_height failed: {e}")
            rest_ht = None
        try:
            hull_util = decoder.concave_hull_utilization()
            hull = getattr(decoder, '_last_hull', None)
        except Exception as e:
            print(f"[GUI][METRICS] concave hull failed: {e}")
            hull_util = None
            hull = None

        if util is not None:
            self.utilization_label.text = f"Utilization: {util:.2%}"
        if rest_len is not None:
            self.rest_length_label.text = f"Rest length: {rest_len:.2f} cm"
        if rest_ht is not None:
            self.rest_height_label.text = f"Rest height: {rest_ht:.2f} cm"
        if hull_util is not None:
            self.utilization_concave_label.text = f"Concave hull utilization: {hull_util:.2%}"
        if hull is not None:
            self._draw_alpha_shape(hull, stroke="#ff0000")


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
        
        # If we don't have any pieces loaded yet, do basic filtering
        if not self.pieces:
            # Always include meta
            filtered_dp = {'meta': dp_all.get('meta', {})}
            # Use the meta parameters to determine what to include
            meta = dp_all.get('meta', {})
            upper = meta.get('upper', {}).get('v')
            wb = meta.get('wb', {}).get('v')
            bottom = meta.get('bottom', {}).get('v')
            
            if wb is not None: 
                filtered_dp['waistband'] = dp_all.get('waistband', {})
            if upper is not None:
                filtered_dp['shirt'] = dp_all.get('shirt', {})
                
                # Add collar with proper bezier parameter filtering
                collar = dp_all.get('collar', {})
                filtered_collar = copy.deepcopy(collar)
                
                # Filter bezier parameters based on collar type
                f_collar_type = collar.get('f_collar', {}).get('v')
                b_collar_type = collar.get('b_collar', {}).get('v')
                
                # Remove front bezier parameters if front collar type is not Bezier2NeckHalf
                if f_collar_type and f_collar_type != 'Bezier2NeckHalf':
                    if 'f_bezier_x' in filtered_collar:
                        del filtered_collar['f_bezier_x']
                    if 'f_bezier_y' in filtered_collar:
                        del filtered_collar['f_bezier_y']
                    if 'f_flip_curve' in filtered_collar:
                        del filtered_collar['f_flip_curve']
                
                # Remove back bezier parameters if back collar type is not Bezier2NeckHalf
                if b_collar_type and b_collar_type != 'Bezier2NeckHalf':
                    if 'b_bezier_x' in filtered_collar:
                        del filtered_collar['b_bezier_x']
                    if 'b_bezier_y' in filtered_collar:
                        del filtered_collar['b_bezier_y']
                    if 'b_flip_curve' in filtered_collar:
                        del filtered_collar['b_flip_curve']
                
                filtered_dp['collar'] = filtered_collar
                filtered_dp['sleeve'] = dp_all.get('sleeve', {})
                filtered_dp['left'] = dp_all.get('left', {})
                
            # Handle bottom garments
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
            if bottom in bottom_map:
                for key in bottom_map[bottom]:
                    if key in dp_all:
                        filtered_dp[key] = dp_all[key]
        else:
            # Extract panel IDs from loaded pieces
            panel_ids = set()
            for piece_id in self.pieces.keys():
                # Remove any "_copy#" suffix to get the original panel ID
                base_id = piece_id.split("_copy")[0]
                panel_ids.add(base_id)
            
            print(f"Loaded panel IDs: {panel_ids}")
            
            # Use the new panel_mapping filter_parameters function
            from nesting.panel_mapping import filter_parameters
            filtered_dp = filter_parameters(dp_all, panel_ids)
            
            # Additional manual filtering for bezier parameters in the GUI
            # This ensures the UI doesn't show irrelevant bezier parameters
            if 'collar' in filtered_dp:
                collar = filtered_dp['collar']
                f_collar_type = collar.get('f_collar', {}).get('v')
                b_collar_type = collar.get('b_collar', {}).get('v')
                
                # Filter front bezier parameters
                if f_collar_type and f_collar_type != 'Bezier2NeckHalf':
                    if 'f_bezier_x' in collar:
                        del collar['f_bezier_x']
                    if 'f_bezier_y' in collar:
                        del collar['f_bezier_y']
                    if 'f_flip_curve' in collar:
                        del collar['f_flip_curve']
                
                # Filter back bezier parameters
                if b_collar_type and b_collar_type != 'Bezier2NeckHalf':
                    if 'b_bezier_x' in collar:
                        del collar['b_bezier_x']
                    if 'b_bezier_y' in collar:
                        del collar['b_bezier_y']
                    if 'b_flip_curve' in collar:
                        del collar['b_flip_curve']
            
        return filtered_dp


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


    def _on_param_change(self, param: str, e) -> None:
        """Handle parameter changes by updating the design and regenerating the pattern."""
        # Update the nested design parameter dictionary
        value = getattr(e, "value", None)
        node = self.design_params
        parts = param.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        leaf = node.setdefault(parts[-1], {})
        if isinstance(leaf, dict) and "v" in leaf:
            leaf["v"] = value
        else:
            node[parts[-1]] = {"v": value}

        # Handle case when using default params
        if self.use_default_params:
            # If using default params, we need to ensure we have a body_params object
            if self.body_params is None:
                try:
                    from assets.bodies.body_params import BodyParameters
                    default_body_path = Path(config.DEFAULT_BODY_PARAM_PATH)
                    if default_body_path.exists():
                        self.body_params = BodyParameters(default_body_path)
                        print("Loaded default body parameters for dynamic design update")
                    else:
                        print(f"Cannot find default body parameters at {default_body_path}")
                        return
                except Exception as exc:
                    print(f"Failed to load default body parameters: {exc}")
                    return

            # Ensure we have a valid pattern_path
            if self.pattern_path is None:
                self.pattern_path = Path(config.DEFAULT_PATTERN_PATH)
            
            # Create a temporary directory for regenerated pattern
            import tempfile
            out_dir = Path(tempfile.mkdtemp())
            
            # Save updated design parameters to the temporary directory
            import yaml
            yaml_path = out_dir / "design_params.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump({"design": self.design_params}, f, default_flow_style=False, sort_keys=False)
        else:
            # Original behavior for non-default parameters
            if self.pattern_path is None or self.body_params is None:
                return
                
            # Use the original pattern directory
            out_dir = self.pattern_path.parent
            
            # Save updated design parameters
            import yaml
            yaml_path = out_dir / "design_params.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump({"design": self.design_params}, f, default_flow_style=False, sort_keys=False)

        # Regenerate the sewing pattern using MetaGarment
        try:
            from assets.garment_programs.meta_garment import MetaGarment
            mg = MetaGarment("Configured_design", self.body_params, self.design_params)
            pattern = mg.assembly()

            pattern.serialize(out_dir, to_subfolder=False, with_3d=False, with_text=False,
                            view_ids=False, empty_ok=True)
            new_pattern_path = out_dir / f"{pattern.name}_specification.json"
            
            # Store the new pattern path
            self.pattern_path = new_pattern_path
            
            # Keep track that we're not using default params anymore since we're
            # now working with a dynamically generated pattern
            self.use_default_params = False
            
            # Store current parameters before loading the new pattern
            original_design_params = self.design_params
            original_body_params = self.body_params
            
            # Reload pieces from the regenerated pattern without updating parameters
            self._load_pattern_core(self.pattern_path, update_params=False)
            
            # Ensure parameters are preserved
            self.design_params = original_design_params
            self.body_params = original_body_params
            
            # Rebuild the sidebar to ensure UI reflects the current parameters
            self._build_sidebar()
            
            # Notify success
            ui.notify("Pattern updated successfully", type="positive")
        except Exception as exc:
            print(f"Error regenerating pattern: {exc}")
            ui.notify(f"Failed to update pattern: {exc}", type="negative")
    
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

    def _update_ga_params(self, _):
        """Update basic GA parameters."""
        self.ga_population_size = int(self.ga_pop_size_input.value or config.POPULATION_SIZE)
        self.ga_num_generations = int(self.ga_generations_input.value or config.NUM_GENERATIONS)
        self.ga_mutation_rate = float(self.ga_mutation_rate_input.value or config.MUTATION_RATE)

    def _update_ga_weights(self, _):
        """Update population composition weights and normalize them."""
        # Get current values
        elites = float(self.ga_elite_weight.value or 0.25)
        offspring = float(self.ga_offspring_weight.value or 0.25)
        mutants = float(self.ga_mutant_weight.value or 0.25)
        randoms = float(self.ga_random_weight.value or 0.25)
        
        # Normalize to sum to 1.0
        total = elites + offspring + mutants + randoms
        if total > 0:
            self.ga_population_weights = {
                'elites': elites / total,
                'offspring': offspring / total,
                'mutants': mutants / total,
                'randoms': randoms / total,
            }
            
            # Update the UI inputs to show normalized values
            self.ga_elite_weight.value = self.ga_population_weights['elites']
            self.ga_offspring_weight.value = self.ga_population_weights['offspring']
            self.ga_mutant_weight.value = self.ga_population_weights['mutants']
            self.ga_random_weight.value = self.ga_population_weights['randoms']

    def _update_ga_rotations(self, _):
        """Update allowed rotations from comma-separated input."""
        try:
            rotations_str = self.ga_rotations_input.value or "0,180"
            self.ga_allowed_rotations = [int(x.strip()) for x in rotations_str.split(',') if x.strip()]
            if not self.ga_allowed_rotations:
                self.ga_allowed_rotations = [0, 180]  # fallback
        except ValueError:
            ui.notify("Invalid rotation values. Using default: 0,180", type="warning")
            self.ga_allowed_rotations = [0, 180]
            self.ga_rotations_input.value = "0,180"

    # ---------------------------------------------------------------------------- #
    #                                PATTERN LOADERS                               #
    # ---------------------------------------------------------------------------- #
    
    
    # def _load_pattern_core(self, path: Path, update_params: bool = True) -> None:
    #     """
    #     Clear old state, load geometry + design/body parameters from *path* using shared loader,
    #     and rebuild the sidebar.
    #     """
    #     self.pattern_path = path
    #     current_use_default_params = self.use_default_params

    #     # reset GUI state
    #     self.panel_path_refs.clear()
    #     self.selected_panel = ""
    #     self.pieces.clear()
    #     self.drag_data.clear()
    #     self.scene.clear()

    #     # --- shared loader -----------------------------------------------------
    #     try:
    #         pieces_dict, design_params, body_params, pattern_name = load_pattern_bundle(path)
    #     except Exception as exc:
    #         print(f"Failed to load pattern bundle: {exc}")
    #         ui.notify(f"Failed to load pattern: {exc}", type="negative")
    #         return

    #     # apply NUM_COPIES logic
    #     self.pieces.update(pieces_dict)
    #     panel_pieces = list(pieces_dict.values())
    #     num_copies = config.NUM_COPIES
    #     for i in range(num_copies):
    #         for piece in panel_pieces:
    #             cp = copy.deepcopy(piece)
    #             cp.id = f"{piece.id}_copy{i+1}"
    #             self.pieces[cp.id] = cp
    #     if num_copies:
    #         print(f"Loaded {num_copies+1} copies, {len(self.pieces)} pieces in total.")

    #     self.layout = Layout(self.pieces)
    #     self._rebuild_panel_outlines()
    #     self.pattern_loaded = True
    #     self._draw_outlines()

    #     # --- params / body (respect existing flags) ----------------------------
    #     if update_params:
    #         if not current_use_default_params and design_params is not None:
    #             self.design_params = design_params
    #         elif current_use_default_params:
    #             # keep existing or fall back to default file if provided
    #             if not self.design_params:
    #                 default_design_path = Path(config.DEFAULT_DESIGN_PARAM_PATH)
    #                 if default_design_path.exists():
    #                     try:
    #                         import yaml as _yaml
    #                         with open(default_design_path, 'r', encoding='utf-8') as f:
    #                             self.design_params = _yaml.safe_load(f).get('design', {})
    #                     except Exception as _exc:
    #                         print(f"Could not load default design params: {_exc}")
    #         if not current_use_default_params and body_params is not None:
    #             self.body_params = body_params
    #         elif current_use_default_params and not getattr(self, 'body_params', None):
    #             default_body_path = Path(config.DEFAULT_BODY_PARAM_PATH)
    #             if default_body_path.exists():
    #                 try:
    #                     from assets.bodies.body_params import BodyParameters as _BP
    #                     self.body_params = _BP(default_body_path)
    #                 except Exception as _exc:
    #                     print(f"Could not load default body params: {_exc}")

    #         # MetaGarment instance
    #         try:
    #             from assets.garment_programs.meta_garment import MetaGarment
    #             if self.body_params and self.design_params:
    #                 self.meta_garment = MetaGarment("GUI_base", self.body_params, self.design_params)
    #         except Exception as exc:
    #             print(f"Failed to create MetaGarment: {exc}")

    #     # rebuild sidebar
    #     self._build_sidebar()
    #     self.use_default_params = current_use_default_params


    # def _load_pattern(self, e: events.UploadEventArguments):
    #     """Handle file-upload event and delegate to the core loader."""
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    #         tmp.write(e.content.read())
    #         tmp_path = Path(tmp.name)

    #     try:
    #         # --- CLEAR ALL PARAM/STATE WHEN A NEW JSON IS UPLOADED ---
    #         # Do NOT carry over params/samplers/meta from previous session
    #         self.design_params = {}
    #         self.body_params = None
    #         self.design_sampler = None
    #         self.meta_garment = None
    #         self.use_default_params = False  # ensure no defaults get injected

    #         # Load only geometry; do not (re)load any params
    #         self._load_pattern_core(tmp_path, update_params=False)

    #         # Rebuild the sidebar so it reflects empty/cleared params
    #         self._build_sidebar()

    #         ui.notify("Pattern loaded ✓", type="positive")
    #     except Exception as exc:
    #         ui.notify(f"Could not load pattern: {exc}", type="negative")
    #         raise  # keep traceback in server log

    def _load_pattern_core(self, path: Path, update_params: bool = True) -> None:
        """
        Clear old state, load geometry + design parameters from *path*,
        and rebuild the sidebar.
        """
        # Store the pattern path for regeneration
        self.pattern_path = path
        
        # Store current use_default_params state
        current_use_default_params = self.use_default_params
        
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

        # # split the first piece into two halves
        # if self.pieces:
        #     first_id = next(iter(self.pieces))
        #     original_piece = self.pieces.pop(first_id)
        #     left_piece, right_piece = original_piece.split()
        #     self.pieces[left_piece.id] = left_piece
        #     self.pieces[right_piece.id] = right_piece

        # split every piece into two halves
        # for piece_id, piece in panel_pieces.items():
        #     original_piece = self.pieces.pop(piece_id)
        #     left_piece, right_piece = original_piece.split()
        #     self.pieces[left_piece.id] = left_piece
        #     self.pieces[right_piece.id] = right_piece

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

        if update_params:
            # --- design parameters ----------------------------------------
            yaml_path = path.parent / "design_params.yaml"
            try:
                import yaml
                if yaml_path.exists() and not self.use_default_params:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        self.design_params = yaml.safe_load(f).get("design", {})
                    #print("Design parameters loaded from pattern folder:", self.design_params)
                elif not self.use_default_params:
                    with path.open("r", encoding="utf-8") as f:
                        spec = json.load(f)
                        self.design_params = spec.get("design", {})
                    #print("Design parameters loaded from pattern JSON:", self.design_params)
                else:
                    default_design_path = Path(config.DEFAULT_DESIGN_PARAM_PATH)
                    if default_design_path.exists():
                        with open(default_design_path, "r", encoding="utf-8") as f:
                            self.design_params = yaml.safe_load(f).get("design", {})
                        #print("Default design parameters loaded:", self.design_params)
                    else:
                        #print(f"Default design params file not found: {default_design_path}")
                        self.design_params = {}
            except Exception as exc:
                print(f"Failed to load design parameters: {exc}")
                self.design_params = {}

        if update_params:
            # --- body parameters -----------------------------------------
            body_path = path.parent / "body_measurements.yaml"
            default_body_path = Path(config.DEFAULT_BODY_PARAM_PATH)
            try:
                from assets.bodies.body_params import BodyParameters
                if body_path.exists() and not self.use_default_params:
                    self.body_params = BodyParameters(body_path)
                    print("Body parameters loaded from pattern folder")
                elif default_body_path.exists() and self.use_default_params:
                    self.body_params = BodyParameters(default_body_path)
                    print("Default body parameters loaded")
                else:
                    print(f"No body parameters found")
                    self.body_params = None
            except Exception as exc:
                print(f"Failed to load body parameters: {exc}")
                self.body_params = None

        if update_params:
            # --- design sampler -----------------------------------------
            try:
                from pygarment.garmentcode.params import DesignSampler
                import tempfile

                if self.design_params:
                    print("Creating design sampler from design parameters...")
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
                        import yaml
                        yaml_content = {'design': self.design_params}
                        yaml.dump(yaml_content, tmp, default_flow_style=False)
                        tmp_path = tmp.name
                        print(f"Created temporary YAML file: {tmp_path}")

                    try:
                        print(f"Creating DesignSampler with file: {tmp_path}")
                        self.design_sampler = DesignSampler(tmp_path)
                        print("DesignSampler created successfully!")
                    except Exception as e:
                        print(f"Error creating DesignSampler: {e}")
                        self.design_sampler = None

                    print(f"Design sampler state: {self.design_sampler is not None}")
                else:
                    self.design_sampler = None
                    print("No design parameters available for sampler")
            except Exception as exc:
                print(f"Failed to create design sampler: {exc}")
                self.design_sampler = None

            # Keep a MetaGarment instance for successive operations
            try:
                from assets.garment_programs.meta_garment import MetaGarment
                if self.body_params and self.design_params:
                    self.meta_garment = MetaGarment("GUI_base", self.body_params, self.design_params)
            except Exception as e:
                print(f"Failed to create MetaGarment: {e}")

        # --- sidebar ---------------------------------------------------
        self._build_sidebar()          # see next section
        
        # Restore use_default_params state unless explicitly set in the method call
        self.use_default_params = current_use_default_params

    def _load_pattern(self, e: events.UploadEventArguments):
        """Handle file-upload event and delegate to the core loader."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(e.content.read())
            tmp_path = Path(tmp.name)

        try:
            # Clear all previous parameters when uploading a new JSON file
            self.design_params = {}
            self.body_params = None
            self.design_sampler = None
            self.meta_garment = None
            self.use_default_params = False
            
            # Load only the pattern geometry, don't update parameters from non-existent files
            self._load_pattern_core(tmp_path, update_params=False)
            
            # Rebuild the sidebar to reflect the cleared parameters
            self._build_sidebar()
            
            ui.notify("Pattern loaded ✓", type="positive")
        except Exception as exc:
            ui.notify(f"Could not load pattern: {exc}", type="negative")
            raise                    # optional: keep traceback in server log




    def _load_default_pattern(self):
        """
        So I don't have to load something in every time I change code... 
        """
        default_path = config.DEFAULT_PATTERN_PATH
        # Always use default params when loading the default pattern
        self.use_default_params = True
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

        # Draw canvas boundaries with more visible lines
        for x1, y1, x2, y2 in [
            (0, 0, self.container_width_px, 0),
            (self.container_width_px, 0, self.container_width_px, self.container_height),
            (self.container_width_px, self.container_height, 0, self.container_height),
            (0, self.container_height, 0, 0),
        ]:
            self._svg_line(x1, y1, x2, y2, stroke="#4a90e2")
        
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
        if polygon is None or polygon.is_empty:
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
        #     print(f"Panel {name} offset: ({dx, dy})")

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
                decoder = BottomLeftDecoder(layout, container, gravitate_once=config.GRAVITATE_ONCE, step=config.GRAVITATE_STEP)
                print("[GUI][AUTO_PLACE] Using BottomLeftDecoder")
            # elif method == "Greedy":
            #     # Greedy placement using sorting + BL
            #     sorted_order = get_piece_order_by_criteria(layout, config.SORT_BY, reverse=True)
            #     decoder = BottomLeftDecoder(layout, container, gravitate_once=config.GRAVITATE_ONCE, step=config.GRAVITATE_STEP)
            #     placements = decoder.decode_in_order(sorted_order)
            #     await self._apply_placements(placements)
            #     concave_hull_usage = decoder.concave_hull_utilization()
            #     self._draw_alpha_shape(decoder._last_hull, stroke="#ff0000")
            #     self.utilization_concave_label.text = f"Concave hull utilization: {concave_hull_usage:.2%}"
            #     ui.notify('Auto placement completed ', type='positive')
            elif method == "NFP":
                decoder = NFPDecoder(layout, container, gravitate_once = config.GRAVITATE_ONCE, step=config.GRAVITATE_STEP)
            # elif method == "RandomBL":
            #     decoder = RandomDecoder(layout, container, decoder="BL")
            # elif method == "RandomNFP":
            #     decoder = RandomDecoder(layout, container, decoder="NFP")
            elif method == "Genetic Algorithm":
                # Use the parameters that were loaded during pattern loading
                print("Using design parameters from pattern loading")
                
                # Debug: Print what we're passing to Evolution
                print(f"Design params: {self.design_params is not None}")
                print(f"Body params: {getattr(self, 'body_params', None) is not None}")
                print(f"Design sampler: {getattr(self, 'design_sampler', None) is not None}")
                
                # Get the pattern name from the current specification if available
                pattern_name = None
                if hasattr(self, 'specification_path') and self.specification_path:
                    pattern_name = Path(self.specification_path).stem
                
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
                    # crossover_method=config.SELECTED_CROSSOVER,
                    design_params=self.design_params if hasattr(self, 'design_params') else None,
                    body_params=self.body_params if hasattr(self, 'body_params') else None,
                    pattern_name=pattern_name or "",
                )
                best_chromosome = evo.run()
                if best_chromosome is None:
                    ui.notify("No valid solution found by the Genetic Algorithm", type="negative")
                    return

                print("Genetic Algorithm completed")
                
                self.pieces = {p.id: copy.deepcopy(p) for p in best_chromosome.genes}
                self.layout = Layout(self.pieces)          # keep layout in sync
                self._rebuild_panel_outlines()             # add seam allowance etc.
                self._draw_outlines()                      # paint them on the canvas
            
                view    = LayoutView(best_chromosome.genes)
                decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)
  
            # elif method == "Jostle":
            #     # Use the default decoder instead of JostleDecoder
            #     view    = LayoutView(layout)
            #     decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)

            # elif method == "BLF":
                #decoder = BottomLeftFill(layout, container, step=config.GRAVITATE_STEP)
            # elif method == "TOPOS":
            #     decoder = TOPOSDecoder(layout, container, eval_terms=("distance",))
            else:
                raise ValueError(f"Unknown placement method: {method}")
            
            print("Now decoding...")
            placements = decoder.decode()  # [(name, dx, dy)]
            print("Decoding done")

            # print placements
            # for name, dx, dy, rot in placements:
            #      print(f"Placing {name} at ({dx:.2f}, {dy:.2f}) cm with rotation {rot}")

            await self._apply_placements(placements)
            self._update_metrics_from_decoder(decoder)

            ui.notify('Auto placement completed ', type='positive')

            # if the BL runs out of space and there are intersections, still draw it but notify the user
            # if self._check_intersections():
            #     ui.notify("Auto placement has intersections", type="negative")
            #     print("Auto placement has intersections")

        except Exception as exc:
            ui.notify(f'Auto placement failed: {exc}', type='negative')

    def _run_heuristics(self):
        """Run Random+BL heuristic 100x before and after manual split, then save plot to log path."""
        ui.notify('Running heuristics...', type='info')
        print('Run heuristics: starting')

        from pathlib import Path
        import copy
        # import io
        # import matplotlib.pyplot as plt

        # Load original pieces
        extractor = PatternPathExtractor(Path(config.DEFAULT_PATTERN_PATH))
        panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)

        # Metrics before split
        print('Run heuristics: before-split runs start')
        before_bb = []
        before_hull = []
        for _ in range(100):
            # Prepare a fresh copy of all pieces for each run
            pieces_copy = {pid: copy.deepcopy(piece) for pid, piece in panel_pieces.items()}
            layout = Layout(pieces_copy)
            dec = RandomDecoder(layout, self.container)
            dec.decode()
            before_bb.append(dec.usage_BB())
            before_hull.append(dec.concave_hull_utilization())
        print('Run heuristics: before-split runs complete')

        # Manual split of every piece
        print('Run heuristics: splitting pieces')
        split_pieces = {}
        for pid, piece in panel_pieces.items():
            left, right = copy.deepcopy(piece).split()
            split_pieces[left.id] = left
            split_pieces[right.id] = right

        # Metrics after split
        print('Run heuristics: after-split runs start')
        after_bb = []
        after_hull = []
        for _ in range(200):
            # Again, fresh copy of all split pieces for each run
            pieces_copy = {pid: copy.deepcopy(piece) for pid, piece in split_pieces.items()}
            layout = Layout(pieces_copy)
            dec = RandomDecoder(layout, self.container)
            dec.decode()
            after_bb.append(dec.usage_BB())
            after_hull.append(dec.concave_hull_utilization())
        print('Run heuristics: after-split runs complete')

        # Plot and save
        print('Run heuristics: plotting results')
        plt.figure(figsize=(6, 6))
        plt.scatter(before_bb, before_hull, c='blue', alpha=0.5, label='before split')
        plt.scatter(after_bb, after_hull, c='red', alpha=0.5, label='after split')
        plt.xlabel('Bounding-box utilization')
        plt.ylabel('Concave-hull utilization')
        plt.legend()
        plt.title('Heuristic: split vs original')
        plt.tight_layout()

        # Ensure the log directory exists, then save the PNG there with timestamp
        log_dir = Path(config.SAVE_LOGS_PATH)
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f'heuristic_split_vs_original_{ts}.png'
        output_path = log_dir / filename
        plt.savefig(output_path, format='png')
        plt.close()

        print(f'Run heuristics: plot saved to {output_path} (timestamp: {ts})')
        ui.notify(f'Heuristics run complete. Plot saved to: {output_path} (timestamp: {ts})', type='positive')

    async def _run_simulated_annealing(self):
        """Run simulated annealing optimization with current pieces and parameters."""
        if not self.pattern_loaded:
            ui.notify("Please load a pattern first", type="warning")
            return
            
        ui.notify("Starting simulated annealing optimization...", type="info")
        
        try:
            # Prepare pieces for simulated annealing
            pieces_list = list(self.pieces.values())
            if not pieces_list:
                ui.notify("No pieces available for optimization", type="warning")
                return
            
            # Create copies of pieces to avoid modifying the originals
            pieces_copy = [copy.deepcopy(piece) for piece in pieces_list]
            
            # Configure simulated annealing parameters
            # These could be made configurable via GUI inputs in the future
            initial_temperature = 0.05
            cooling_rate = 0.9
            iterations_per_temp = 5

            # Create simulated annealing instance
            sa = SimulatedAnnealing(
                pieces=pieces_copy,
                container=self.container,
                cooling_rate=cooling_rate,
                initial_temperature=initial_temperature,
                iterations_per_temp=iterations_per_temp,
                design_params=copy.deepcopy(self.design_params) if hasattr(self, 'design_params') else None,
                body_params=self.body_params if hasattr(self, 'body_params') else None
            )
            
            # Get initial fitness
            initial_fitness = sa.fitness(sa.current_state)
            print(f"Initial fitness: {initial_fitness:.4f}")
            
            # Run optimization
            print("Running simulated annealing optimization...")
            sa.run()
            
            # Get final results
            final_fitness = sa.best_fitness
            improvement = final_fitness - initial_fitness
            
            print(f"Final fitness: {final_fitness:.4f}")
            print(f"Improvement: {improvement:.4f}")
            
            # Update the GUI with the optimized layout
            if sa.best_state:
                print("Simulated Annealing completed, applying best solution...")
                
                # Debug: Check what's in the best state
                best_piece_ids = [p.id for p in sa.best_state]
                split_pieces_in_best = [pid for pid in best_piece_ids if "_split_" in pid]
                print(f"[GUI] DEBUG: Best state has {len(split_pieces_in_best)} split pieces out of {len(best_piece_ids)} total: {split_pieces_in_best}")
                
                # Update pieces with the best state found
                self.pieces = {p.id: copy.deepcopy(p) for p in sa.best_state}
                self.layout = Layout(self.pieces)          # keep layout in sync
                self._rebuild_panel_outlines()             # add seam allowance etc.
                self._draw_outlines()                      # paint them on the canvas
                
                # Decode the best solution to get actual placements
                view = LayoutView(sa.best_state)
                decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, self.container, step=config.GRAVITATE_STEP)
                
                print("Now decoding best SA solution...")
                placements = decoder.decode()  # [(name, dx, dy)]
                print("Decoding done")
                
                # Apply the placements to update piece positions on canvas
                await self._apply_placements(placements)
                self._update_metrics_from_decoder(decoder)
                
                ui.notify(
                    f"Simulated annealing complete! Fitness improved by {improvement:.4f} "
                    f"(from {initial_fitness:.4f} to {final_fitness:.4f})",
                    type="positive"
                )
            else:
                ui.notify("Simulated annealing completed but no improvement found", type="info")
                
        except Exception as exc:
            print(f"Simulated annealing error: {exc}")
            import traceback
            traceback.print_exc()
            ui.notify(f"Simulated annealing failed: {exc}", type="negative")

    async def _run_two_exchange(self):
        """Run 2-exchange (local swap) hill-climbing using the current pieces.

        Mirrors the flow of simulated annealing UI integration but uses only
        deterministic/improving local swap moves within a window (default 3).
        """
        if not self.pattern_loaded:
            ui.notify("Please load a pattern first", type="warning")
            return

        ui.notify("Starting 2-Exchange local search...", type="info")
        try:
            from .two_exchange_search import TwoExchangeSearch
            # Prepare initial state
            pieces_list = list(self.pieces.values())
            if not pieces_list:
                ui.notify("No pieces available for optimization", type="warning")
                return

            # Strategy selection: could be made user-configurable later.
            # For now choose 'best' (steepest ascent) as default.
            strategy = "best"
            max_distance = 3

            # Run search
            search = TwoExchangeSearch(
                pieces=pieces_list,
                container=self.container,
                strategy=strategy,
                max_distance=max_distance,
                verbose=True,
            )
            initial_fitness = search.best_fitness
            print(f"[2-Exchange] Initial fitness: {initial_fitness:.4f}")
            best_pieces, best_fitness = search.run()
            improvement = best_fitness - initial_fitness
            print(f"[2-Exchange] Final fitness: {best_fitness:.4f} (improvement {improvement:.4f})")

            # Apply best solution if improved (or even if equal to reflect normalization)
            if best_pieces and best_fitness >= initial_fitness:
                self.pieces = {p.id: copy.deepcopy(p) for p in best_pieces}
                self.layout = Layout(self.pieces)
                self._rebuild_panel_outlines()
                self._draw_outlines()

                # Decode placements for visualization
                view = LayoutView(best_pieces)
                decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, self.container, step=config.GRAVITATE_STEP)
                print("[2-Exchange] Decoding best layout...")
                placements = decoder.decode()
                print("[2-Exchange] Decode complete")
                await self._apply_placements(placements)
                self._update_metrics_from_decoder(decoder)

                ui.notify(
                    f"2-Exchange complete! Fitness improved by {improvement:.4f} (to {best_fitness:.4f})",
                    type="positive" if improvement > 0 else "info"
                )
            else:
                ui.notify("2-Exchange search found no improving swap", type="info")
        except Exception as exc:
            print(f"2-Exchange error: {exc}")
            traceback.print_exc()
            ui.notify(f"2-Exchange failed: {exc}", type="negative")

    async def _run_random_search(self):
        """Run pure random search over piece order (and optional rotations)."""
        if not self.pattern_loaded:
            ui.notify("Please load a pattern first", type="warning")
            return

        ui.notify("Starting Random Search...", type="info")
        try:
            from .random_search import RandomSearch

            pieces_list = list(self.pieces.values())
            if not pieces_list:
                ui.notify("No pieces available for optimization", type="warning")
                return

            # Parameters could be exposed via GUI later
            num_samples = 300
            shuffle_order = True
            randomize_rotations = True  # could tie to config.ENABLE_ROTATIONS

            rs = RandomSearch(
                pieces=pieces_list,
                container=self.container,
                num_samples=num_samples,
                shuffle_order=shuffle_order,
                randomize_rotations=randomize_rotations,
                track_history=False,
                verbose=True,
            )
            initial = rs.metric_fn(pieces_list, config.SELECTED_DECODER, self.container)
            best_pieces, best_fitness = rs.run()
            improvement = best_fitness - initial
            print(f"[RandomSearch] Initial fitness: {initial:.4f}")
            print(f"[RandomSearch] Best fitness: {best_fitness:.4f} (Δ {improvement:.4f})")

            # Apply best state
            self.pieces = {p.id: copy.deepcopy(p) for p in best_pieces}
            self.layout = Layout(self.pieces)
            self._rebuild_panel_outlines()
            self._draw_outlines()

            # Decode best layout for placements
            view = LayoutView(best_pieces)
            decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, self.container, step=config.GRAVITATE_STEP)
            placements = decoder.decode()
            await self._apply_placements(placements)
            self._update_metrics_from_decoder(decoder)

            ui.notify(
                f"Random Search complete! Δ {improvement:.4f} (best {best_fitness:.4f})",
                type="positive" if improvement > 0 else "info"
            )
        except Exception as exc:
            print(f"Random Search error: {exc}")
            traceback.print_exc()
            ui.notify(f"Random Search failed: {exc}", type="negative")

    async def _run_random_then_sa(self):
        """Run Random Search first, then start SA from the best random sample."""
        if not self.pattern_loaded:
            ui.notify("Please load a pattern first", type="warning")
            return
        ui.notify("Random Search pre-initialization for SA...", type="info")
        try:
            from .random_search import RandomSearch
            pieces_list = list(self.pieces.values())
            if not pieces_list:
                ui.notify("No pieces available", type="warning")
                return

            rs = RandomSearch(
                pieces=pieces_list,
                container=self.container,
                num_samples=200,
                shuffle_order=True,
                randomize_rotations=True,
                track_history=False,
                verbose=False,
            )
            best_pieces, best_fit = rs.run()
            print(f"[Random→SA] Seed fitness after RS: {best_fit:.4f}")

            # Seed current state with RS result
            self.pieces = {p.id: copy.deepcopy(p) for p in best_pieces}
            self.layout = Layout(self.pieces)
            self._rebuild_panel_outlines()
            self._draw_outlines()

            # Now run SA using the seeded arrangement
            await self._run_simulated_annealing()
        except Exception as exc:
            print(f"Random→SA chain error: {exc}")
            traceback.print_exc()
            ui.notify(f"Random→SA failed: {exc}", type="negative")

    async def _run_random_then_two_exchange(self):
        """Run Random Search then refine with 2-Exchange local search."""
        if not self.pattern_loaded:
            ui.notify("Please load a pattern first", type="warning")
            return
        ui.notify("Random Search pre-initialization for 2-Exchange...", type="info")
        try:
            from .random_search import RandomSearch
            from .two_exchange_search import TwoExchangeSearch
            pieces_list = list(self.pieces.values())
            if not pieces_list:
                ui.notify("No pieces available", type="warning")
                return

            rs = RandomSearch(
                pieces=pieces_list,
                container=self.container,
                num_samples=200,
                shuffle_order=True,
                randomize_rotations=True,
                track_history=False,
                verbose=False,
            )
            base_pieces, base_fit = rs.run()
            print(f"[Random→2E] Seed fitness after RS: {base_fit:.4f}")

            # Apply RS best state
            self.pieces = {p.id: copy.deepcopy(p) for p in base_pieces}
            self.layout = Layout(self.pieces)
            self._rebuild_panel_outlines()
            self._draw_outlines()

            # Run 2-Exchange refinement
            search = TwoExchangeSearch(
                pieces=base_pieces,
                container=self.container,
                strategy="best",
                max_distance=3,
                verbose=True,
            )
            best_pieces, best_fitness = search.run()
            improvement = best_fitness - base_fit
            print(f"[Random→2E] Post-refinement fitness {best_fitness:.4f} (Δ {improvement:.4f})")

            # Apply refined solution
            self.pieces = {p.id: copy.deepcopy(p) for p in best_pieces}
            self.layout = Layout(self.pieces)
            self._rebuild_panel_outlines()
            self._draw_outlines()

            # Decode for placement
            view = LayoutView(best_pieces)
            decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, self.container, step=config.GRAVITATE_STEP)
            placements = decoder.decode()
            await self._apply_placements(placements)
            self._update_metrics_from_decoder(decoder)

            ui.notify(
                f"Random→2-Exchange complete Δ {improvement:.4f} (final {best_fitness:.4f})",
                type="positive" if improvement > 0 else "info"
            )
        except Exception as exc:
            print(f"Random -> 2E chain error: {exc}")
            traceback.print_exc()
            ui.notify(f"Random -> 2-Exchange failed: {exc}", type="negative")

    async def _run_genetic_algorithm(self):
        """Run genetic algorithm with GUI-configured parameters."""
        if not self.pattern_loaded:
            ui.notify("Please load a pattern first", type="warning")
            return
            
        ui.notify("Starting genetic algorithm optimization...", type="info")
        
        try:
            # Prepare a copy of the layout for GA
            layout = copy.deepcopy(self.layout)
            container = self.container
            
            print(f"Running GA with parameters:")
            print(f"  Population size: {self.ga_population_size}")
            print(f"  Generations: {self.ga_num_generations}")
            print(f"  Mutation rate: {self.ga_mutation_rate}")
            print(f"  Population weights: {self.ga_population_weights}")
            print(f"  Allowed rotations: {self.ga_allowed_rotations}")
            
            # Temporarily override config values with GUI parameters
            import nesting.config as config
            original_population_size = config.POPULATION_SIZE
            original_num_generations = config.NUM_GENERATIONS
            original_mutation_rate = config.MUTATION_RATE
            original_population_weights = config.POPULATION_WEIGHTS.copy()
            original_allowed_rotations = config.ALLOWED_ROTATIONS.copy()
            
            # Set GUI parameters in config
            config.POPULATION_SIZE = self.ga_population_size
            config.NUM_GENERATIONS = self.ga_num_generations
            config.MUTATION_RATE = self.ga_mutation_rate
            config.POPULATION_WEIGHTS = self.ga_population_weights.copy()
            config.ALLOWED_ROTATIONS = self.ga_allowed_rotations.copy()
            
            # Also update the ACTIVE config instance
            config.ACTIVE.population_size = self.ga_population_size
            config.ACTIVE.num_generations = self.ga_num_generations
            config.ACTIVE.mutation_rate = self.ga_mutation_rate
            config.ACTIVE.population_weights = self.ga_population_weights.copy()
            config.ACTIVE.allowed_rotations = self.ga_allowed_rotations.copy()
            
            try:
                # Get the pattern name from the current specification if available
                pattern_name = None
                if hasattr(self, 'pattern_path') and self.pattern_path:
                    pattern_name = Path(self.pattern_path).stem
                
                # Create and run Evolution instance
                evo = Evolution(
                    self.pieces,
                    container,
                    num_generations=self.ga_num_generations,
                    population_size=self.ga_population_size,
                    mutation_rate=self.ga_mutation_rate,
                    enable_dynamic_stopping=config.ENABLE_DYNAMIC_STOPPING,
                    early_stop_window=config.EARLY_STOP_WINDOW,
                    early_stop_tolerance=config.EARLY_STOP_TOLERANCE,
                    enable_extension=config.ENABLE_EXTENSION,
                    extend_window=config.EXTEND_WINDOW,
                    extend_threshold=config.EXTEND_THRESHOLD,
                    max_generations=config.MAX_GENERATIONS,
                    design_params=self.design_params if hasattr(self, 'design_params') else None,
                    body_params=self.body_params if hasattr(self, 'body_params') else None,
                    pattern_name=pattern_name or "",
                )
                
                # Run the GA
                best_chromosome = evo.run()
                
                if best_chromosome is None:
                    ui.notify("No valid solution found by the Genetic Algorithm", type="negative")
                    return

                print("Genetic Algorithm completed")
                
                # Update the GUI with the best solution
                self.pieces = {p.id: copy.deepcopy(p) for p in best_chromosome.genes}
                self.layout = Layout(self.pieces)
                self._rebuild_panel_outlines()
                self._draw_outlines()
                
                # Calculate final placement for display
                view = LayoutView(best_chromosome.genes)
                decoder = DECODER_REGISTRY[config.SELECTED_DECODER](view, container, step=config.GRAVITATE_STEP)
                placements = decoder.decode()
                
                await self._apply_placements(placements)
                
                # Calculate and display final metrics
                self._update_metrics_from_decoder(decoder)
                
                ui.notify(f"GA completed: {self.ga_num_generations} generations, best fitness: {best_chromosome.fitness:.4f}", type="positive")
                
            finally:
                # Restore original config values
                config.POPULATION_SIZE = original_population_size
                config.NUM_GENERATIONS = original_num_generations
                config.MUTATION_RATE = original_mutation_rate
                config.POPULATION_WEIGHTS = original_population_weights
                config.ALLOWED_ROTATIONS = original_allowed_rotations
                
                config.ACTIVE.population_size = original_population_size
                config.ACTIVE.num_generations = original_num_generations
                config.ACTIVE.mutation_rate = original_mutation_rate
                config.ACTIVE.population_weights = original_population_weights
                config.ACTIVE.allowed_rotations = original_allowed_rotations
                
        except Exception as exc:
            print(f"Genetic algorithm error: {exc}")
            import traceback
            traceback.print_exc()
            ui.notify(f"Genetic algorithm failed: {exc}", type="negative")


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

    def _split_panel(self, proportion=0.5):
        """Split the currently selected panel into two and redraw.
        # """
        if not self.selected_panel:
            ui.notify("No panel selected to split", type="warning")
            return
            
        pid = self.selected_panel
        piece = self.pieces.get(pid)
        if piece is None:
            ui.notify(f"Panel '{pid}' not found", type="negative")
            return
            
        print(f"[GUI] Splitting panel '{pid}'")
        
        # Check if we have design and body params needed for regeneration
        if not self.design_params or not self.body_params:
            print(f"[GUI] Missing design or body params")
            ui.notify("Design or body parameters are missing", type="warning")

            # use the Piece.split() method to split the panel
            try:
                left, right = piece.split()
                # Remove the original piece and add the two new split pieces
                self.pieces[left.id] = left
                self.pieces[right.id] = right 
                self.pieces.pop(pid)

                
                #self._draw_panel(left.id)
                #self._draw_panel(right.id)
                self.selected_panel = None # Clear selection after split
                ui.notify(f"Panel '{pid}' split into '{left.id}' and '{right.id}'", type="positive")
            except Exception as e:
                print(f"Error splitting panel: {e}")
                ui.notify(f"Error splitting panel: {e}", type="negative")
            self._rebuild_panel_outlines()  # Rebuild outlines after split
            self._draw_outlines()            # Redraw the outlines
            return

        # # Follow the chromosome's split mutation pattern
        #from nesting.panel_mapping import get_panel_type
        import tempfile
        from pathlib import Path
        
        # Identify the panel type
        #panel_type = get_panel_type(pid)
        #if not panel_type:
        #    ui.notify(f"Cannot determine panel type for '{pid}'", type="warning")
        #    return
            
        #ui.notify(f"[GUI] Panel type identified as: '{panel_type}'")
        
        # Use or create a MetaGarment instance
        from assets.garment_programs.meta_garment import MetaGarment

        if self.meta_garment is None:
            mg = MetaGarment("GUI_base", self.body_params, self.design_params)
        else:
            mg = self.meta_garment

        # Create a temporary directory for the regenerated pattern
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "regenerated_pattern.json"
            
        
            # Split the panel using our new method
            new_panel_names = mg.split_panel(pid, proportion)
            
            # Generate the pattern with the split panels
            pattern = mg.assembly()
            
            # Save the pattern to a temporary file
            pattern.serialize(Path(tmpdir), to_subfolder=False, with_3d=False, with_text=False,
                            view_ids=False, empty_ok=True)
            
            # Get the correct file path - pattern.name_specification.json
            pattern_path = Path(tmpdir) / f"{pattern.name}_specification.json"
            print(f"Looking for pattern at: {pattern_path}")
            
            # Check if the file exists
            if not pattern_path.exists():
                print(f"Pattern file not found at {pattern_path}")
                # List all files in the directory to help debug
                print(f"Files in directory: {list(Path(tmpdir).glob('*'))}")
                raise FileNotFoundError(f"Pattern file not found at {pattern_path}")
            
            # Store current parameters and UI state
            original_design_params = self.design_params
            original_body_params = self.body_params
            #original_use_default_params = self.use_default_params
            
            # Save design params to temporary file to ensure they're loaded properly
            if original_design_params:
                import yaml
                with open(Path(tmpdir) / "design_params.yaml", "w", encoding="utf-8") as f:
                    yaml.dump({"design": original_design_params}, f, default_flow_style=False)
            
            # Save body params to temporary file if they exist
            if original_body_params:
                try:
                    # If body_params has a save method, use it
                    if hasattr(original_body_params, "save"):
                        original_body_params.save(Path(tmpdir) / "body_measurements.yaml")
                    # Otherwise, just copy the original file
                    elif hasattr(self, "pattern_path") and self.pattern_path:
                        body_path = self.pattern_path.parent / "body_measurements.yaml"
                        if body_path.exists():
                            import shutil
                            shutil.copy(body_path, Path(tmpdir) / "body_measurements.yaml")
                except Exception as e:
                    print(f"Warning: Unable to save body parameters to temp directory: {e}")
            
            # Temporarily set use_default_params to False to ensure our saved params are used
            #self.use_default_params = False
            
            # Load the new pattern into the GUI without resetting parameters
            self._load_pattern_core(pattern_path, update_params=False)
            # preserve updated MetaGarment instance
            self.meta_garment = mg
            
            # If parameters weren't properly loaded, restore them manually
            if not self.design_params and original_design_params:
                print("Restoring original design parameters manually")
                self.design_params = original_design_params
                
                # Recreate design sampler with restored parameters
                if self.design_params:
                    self.design_sampler = self._create_design_sampler(self.design_params)
            
            if not self.body_params and original_body_params:
                print("Restoring original body parameters manually")
                self.body_params = original_body_params
            
            # Restore UI state
            #self.use_default_params = original_use_default_params
            
            # Rebuild the sidebar to ensure UI reflects the restored parameters
            self._build_sidebar()
            
            # Select one of the new panels
            if new_panel_names and len(new_panel_names) > 0:
                self._select_panel(new_panel_names[0])
            
            ui.notify(f"Panel '{pid}' split into {', '.join(new_panel_names)}", type="positive")
                
            # except Exception as e:
            #     print(f"Error splitting panel: {str(e)}")
            #     ui.notify(f"Error splitting panel: {str(e)}", type="negative")

            # find the panel in the regenerated pattern

        

    def _create_design_sampler(self, design_params):
        """
        Create a DesignSampler instance from a dictionary of design parameters.
        
        Args:
            design_params: Dictionary of design parameters
            
        Returns:
            A DesignSampler instance, or None if creation fails
        """
        try:
            from pygarment.garmentcode.params import DesignSampler
            import yaml
            import tempfile
            from pathlib import Path
            
            # Create a temporary YAML file with the design parameters
            with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as tmp:
                # Format the content as required by DesignSampler.load
                yaml_content = {'design': design_params}
                yaml.dump(yaml_content, tmp, default_flow_style=False)
                tmp_path = tmp.name
            
            # Create a DesignSampler with the temporary file
            sampler = DesignSampler(tmp_path)
            
            # Clean up the temporary file
            Path(tmp_path).unlink()
            
            print(f"Design sampler created with {len(design_params)} parameters")
            return sampler
        
        except Exception as exc:
            print(f"Failed to create design sampler: {exc}")
            return None

if __name__ in {"__main__", "__mp_main__"}:
    # run_gui.py
    
    gui = NestingGUI(pattern_path=None)
    ui.run(port=8082)
