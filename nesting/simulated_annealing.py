import math
import time
import random
import copy
from typing import Optional, Union, List
from pathlib import Path
import threading

from .layout import Piece, Container, Layout
from .operations import METRIC_REGISTRY, Operators, weighted_choice
import nesting.config as config

class SimulatedAnnealing:
    def __init__(
            self, 
            pieces: List[Piece], 
            container: Container,
            cooling_rate, 
            initial_temperature,
            iterations_per_temp: int = 10,
            design_params: Optional[dict] = None,
            body_params: Optional[object] = None,
            # initial_design_params: Optional[dict] = None,
        ):
        # Normalize parent_id and root_id for initial pieces
        normalized_pieces = []
        for piece in pieces:
            if "_split_" in piece.id:
                # Split pieces: parent_id = original piece, root_id = root piece
                original_root_id = piece.id.split("_split_")[0]
                piece.parent_id = original_root_id
                piece.root_id = original_root_id
            else:
                # Original unsplit pieces: no parent, root is self
                piece.parent_id = None
                piece.root_id = piece.id
            normalized_pieces.append(piece)

        self.current_state = normalized_pieces
        self.container = container
        self.cooling_rate = cooling_rate
        self.temperature = initial_temperature
        self.iterations_per_temp = iterations_per_temp
        import copy
        self.design_params = design_params
        self.body_params = body_params
        self.initial_design_params = copy.deepcopy(design_params) if design_params else None

        self.last_operation = None
        
        # best snapshot (deep copies)
        self.best_fitness = float('-inf')
        self.best_state = copy.deepcopy(self.current_state)
        self.split_history = []
        self.split_set = set()  # Track IDs of split pieces, not objects
        self.best_split_history = list(self.split_history)
        self.best_split_set = set(self.split_set)
        self.best_design_params = copy.deepcopy(self.design_params)

        # Initialize logging
        self.log_lines = []
        self._log_lock = threading.Lock()

        # Setup log file
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_dir = getattr(config, 'SAVE_LOGS_PATH', './logs')
        if hasattr(config, 'PATTERN_NAME') and config.PATTERN_NAME:
            log_dir = Path(log_dir) / config.PATTERN_NAME
        log_dir = f"{log_dir}_sa_{ts}"
        if getattr(config, 'SAVE_LOGS', True):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / f"simulated_annealing_log_{ts}.txt"
        
        # Initialize CSV logging for convergence tracking
        self.csv_path = Path(log_dir) / f"sa_convergence_{ts}.csv"
        self.csv_data = []
        self.iteration_count = 0
        self.temperature_level = 0
        
        # Setup SVG directory for saving state visualizations
        self.svg_dir = Path(log_dir) / "svgs"
        if getattr(config, 'SAVE_GENERATION_SVGS', True) and getattr(config, 'SAVE_LOGS', True):
            self.svg_dir.mkdir(exist_ok=True)
            self.log(f"SVG state visualization directory: {self.svg_dir}")
        else:
            self.svg_dir = None
        
        # Create CSV file with header if logging is enabled
        if getattr(config, 'SAVE_LOGS', True):
            try:
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    import csv
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        'iteration', 'temperature_level', 'temperature', 'operation', 
                        'old_fitness', 'new_fitness', 'acceptance_probability', 
                        'accepted', 'best_fitness_so_far', 'time_elapsed'
                    ])
                self.log(f"CSV convergence tracking initialized: {self.csv_path}")
            except Exception as e:
                self.log(f"Warning: Could not initialize CSV logging: {e}")
                self.csv_path = None

        # Auto-calibrate temperature and cooling rate based on fitness metric
        # (Must be called AFTER logging setup)
        self._calibrate_temperature_parameters()

        self.meta_garment = None
        if self.design_params and self.body_params:
            try:
                from assets.garment_programs.meta_garment import MetaGarment
                self.meta_garment = MetaGarment("metagarment", self.body_params, self.design_params)
            except Exception as exc:
                self.log(f"[SimulatedAnnealing] Failed to create MetaGarment: {exc}")
        
        # Save initial state SVG
        self._save_state_svg(self.current_state, 0, suffix="_initial")

    def log(self, msg: str = "", divider: bool = False):
        """Log message to both console and file"""
        lines_to_write = []
        if divider:
            line = "-" * 60
            print(line)
            self.log_lines.append(line)
            lines_to_write.append(line)
        if msg:
            print(msg)
            self.log_lines.append(msg)
            lines_to_write.append(msg)

        # Continuous flush to disk for each log call
        if getattr(config, 'SAVE_LOGS', True) and lines_to_write:
            with self._log_lock:
                with self.log_path.open("a", encoding="utf-8") as f:
                    for ln in lines_to_write:
                        f.write(ln + "\n")

    def log_csv_data(self, operation: str, old_fitness: float, new_fitness: float, 
                     acceptance_prob: float, accepted: bool, time_elapsed: float):
        """Log convergence data to CSV file"""
        if not getattr(config, 'SAVE_LOGS', True) or self.csv_path is None:
            return
            
        try:
            import csv
            row_data = [
                self.iteration_count,
                self.temperature_level, 
                self.temperature,
                operation,
                old_fitness,
                new_fitness,
                acceptance_prob,
                accepted,
                self.best_fitness,
                time_elapsed
            ]
            
            # Store in memory for potential batch writing
            self.csv_data.append(row_data)
            
            # Write immediately to disk for real-time tracking
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
                
        except Exception as e:
            self.log(f"Warning: Could not write CSV data: {e}")

    def _flush_log(self) -> None:
        """Write accumulated log lines to disk (append or create)."""
        # Ensure the log directory exists before writing
        if not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()

        with self._log_lock:
            with self.log_path.open("w", encoding="utf-8") as f:
                for line in self.log_lines:
                    f.write(line + "\n")

    def _save_state_svg(self, state: List[Piece], iteration: int, 
                       decoder=None, suffix: str = "") -> None:
        """Save the current state as an SVG visualization.
        
        Args:
            state: List of pieces to visualize
            iteration: Current iteration number for filename
            decoder: Optional pre-computed decoder to avoid redundant calculations
            suffix: Optional suffix for filename (e.g., "_best", "_current")
        """
        if not (getattr(config, 'SAVE_LOGS', True) and getattr(config, 'SAVE_GENERATION_SVGS', True) and self.svg_dir is not None):
            return
            
        try:
            import copy
            import svgwrite
            from .layout import LayoutView
            from .placement_engine import DECODER_REGISTRY
            
            # Use provided decoder or create a new one
            if decoder is None:
                view = LayoutView([copy.deepcopy(p) for p in state])
                decoder = DECODER_REGISTRY[config.SELECTED_DECODER](
                    view, self.container, step=getattr(config, 'GRAVITATE_STEP', 1.0)
                )
                decoder.decode()
            
            # Create filename with optional suffix
            filename = f"iter_{iteration:06d}{suffix}.svg"
            
            dwg = svgwrite.Drawing(
                filename=str(self.svg_dir / filename),
                size=(f"{self.container.width}cm", f"{self.container.height}cm"),
                viewBox=f"0 0 {self.container.width} {self.container.height}"
            )
            
            # Add container boundary
            dwg.add(dwg.rect(
                insert=(0, 0),
                size=(self.container.width, self.container.height),
                fill="none", stroke="red", stroke_width=0.2
            ))
            
            # Add placed pieces
            for piece in decoder.placed:
                pts = [(x + piece.translation[0], y + piece.translation[1]) 
                       for x, y in piece.get_outer_path()]
                dwg.add(dwg.polygon(
                    points=pts, 
                    fill="lightblue", 
                    fill_opacity=0.3,
                    stroke="black", 
                    stroke_width=0.5
                ))
                
                # Add piece ID as text at center
                if pts:
                    center_x = sum(x for x, y in pts) / len(pts)
                    center_y = sum(y for x, y in pts) / len(pts)
                    dwg.add(dwg.text(
                        piece.id,
                        insert=(center_x, center_y),
                        font_size="2px",
                        text_anchor="middle",
                        fill="black"
                    ))
            
            dwg.save()
            
        except Exception as e:
            self.log(f"Warning: Could not save SVG for iteration {iteration}: {e}")

    def _finalize_csv_logging(self):
        """Finalize CSV logging with summary statistics."""
        if not getattr(config, 'SAVE_LOGS', True) or self.csv_path is None:
            return
            
        try:
            # Calculate summary statistics
            if self.csv_data:
                accepted_count = sum(1 for row in self.csv_data if row[7])  # accepted column
                total_iterations = len(self.csv_data)
                acceptance_rate = accepted_count / total_iterations if total_iterations > 0 else 0
                
                # Get final fitness values
                final_fitness = self.csv_data[-1][5] if self.csv_data else 0  # new_fitness from last row
                initial_fitness = self.csv_data[0][4] if self.csv_data else 0  # old_fitness from first row
                improvement = self.best_fitness - initial_fitness
                
                # Add summary to the log
                self.log(divider=True)
                self.log("SIMULATED ANNEALING CONVERGENCE SUMMARY")
                self.log(f"Total iterations: {total_iterations}")
                self.log(f"Accepted moves: {accepted_count}")
                self.log(f"Acceptance rate: {acceptance_rate:.3f}")
                self.log(f"Initial fitness: {initial_fitness:.6f}")
                self.log(f"Final fitness: {final_fitness:.6f}")
                self.log(f"Best fitness achieved: {self.best_fitness:.6f}")
                self.log(f"Total improvement: {improvement:.6f}")
                self.log(f"CSV convergence data saved to: {self.csv_path}")
                self.log(divider=True)
                
        except Exception as e:
            self.log(f"Warning: Could not finalize CSV logging: {e}")

    def _calibrate_temperature_parameters(self):
        """Auto-calibrate temperature and cooling rate based on fitness metric scale."""
        metric_name = config.SELECTED_FITNESS_METRIC
        current_fitness = self.fitness(self.current_state)
        
        # Define metric characteristics and suggested parameters
        metric_configs = {
            # Percentage-based metrics (0-1 range)
            "usage_bb": {"scale": "percentage", "temp_factor": 0.1, "cooling": 0.95},
            "concave_hull": {"scale": "percentage", "temp_factor": 0.1, "cooling": 0.95},
            "bb_cc": {"scale": "percentage", "temp_factor": 0.1, "cooling": 0.95},
            
            # Area-based metrics (large values, inverted)
            "concave_hull_area": {"scale": "area_inverted", "temp_factor": 1000, "cooling": 0.98},
            "bb_area": {"scale": "area_inverted", "temp_factor": 1000, "cooling": 0.98},
            "bb_cc_area": {"scale": "area_inverted", "temp_factor": 1000, "cooling": 0.98},
            
            # Length-based metrics (cm scale)
            "rest_length": {"scale": "length", "temp_factor": 10, "cooling": 0.97},
            "rest_height": {"scale": "length", "temp_factor": 10, "cooling": 0.97},
            
            # Combined metrics (mixed scales)
            "cc_with_rest_height": {"scale": "combined", "temp_factor": 1, "cooling": 0.96},
            "cc_with_rest_length": {"scale": "combined", "temp_factor": 1, "cooling": 0.96},
            "bb_with_rest_length": {"scale": "combined", "temp_factor": 1, "cooling": 0.96},
        }
        
        # Get config for current metric or use defaults
        config_data = metric_configs.get(metric_name, {"scale": "unknown", "temp_factor": 1, "cooling": 0.95})
        
        # Sample a few random mutations to estimate fitness variance
        sample_deltas = []
        original_state = copy.deepcopy(self.current_state)
        
        for _ in range(5):  # Sample 5 mutations
            try:
                # Apply a simple rotation mutation for sampling
                mutated_state = Operators.rotate(self.current_state)
                mutated_fitness = self.fitness(mutated_state)
                delta = abs(mutated_fitness - current_fitness)
                if delta > 0:  # Only consider actual changes
                    sample_deltas.append(delta)
            except Exception:
                continue  # Skip failed mutations
        
        # Restore original state
        self.current_state = original_state
        
        # Calculate calibrated parameters
        if sample_deltas:
            avg_delta = sum(sample_deltas) / len(sample_deltas)
            # Temperature should be roughly 2-5x the average fitness change for good acceptance
            calibrated_temp = max(avg_delta * 3, config_data["temp_factor"] * 0.1)
        else:
            # Fallback if no deltas found
            calibrated_temp = config_data["temp_factor"]
        
        calibrated_cooling = config_data["cooling"]
        
        # Apply calibration
        original_temp = self.temperature
        original_cooling = self.cooling_rate
        
        self.temperature = calibrated_temp
        self.cooling_rate = calibrated_cooling
        
        self.log(f"[CALIBRATION] Metric: {metric_name} (scale: {config_data['scale']})")
        self.log(f"[CALIBRATION] Current fitness: {current_fitness:.6f}")
        if sample_deltas:
            self.log(f"[CALIBRATION] Sample fitness deltas: {sample_deltas}")
            self.log(f"[CALIBRATION] Average delta: {avg_delta:.6f}")
        self.log(f"[CALIBRATION] Temperature: {original_temp:.4f} -> {self.temperature:.4f}")
        self.log(f"[CALIBRATION] Cooling rate: {original_cooling:.4f} -> {self.cooling_rate:.4f}")

    # evaluate fitness of a state
    def fitness(self, state: List[Piece], return_decoder: bool = False) -> Union[float, tuple[float, object]]:
        """Evaluate fitness of a state, optionally returning the decoder for reuse.
        
        Args:
            state: List of pieces to evaluate
            return_decoder: If True, return (fitness, decoder) tuple for reuse
            
        Returns:
            Either fitness value or (fitness, decoder) tuple
        """
        import copy
        from .layout import LayoutView
        from .placement_engine import DECODER_REGISTRY
        
        view = LayoutView([copy.deepcopy(p) for p in state])
        decoder = DECODER_REGISTRY[config.SELECTED_DECODER](
            view, self.container, step=getattr(config, 'GRAVITATE_STEP', 1.0)
        )
        decoder.decode()
        
        metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        fitness_value = metric_fn(state, config.SELECTED_DECODER, self.container)
        
        if return_decoder:
            return fitness_value, decoder
        return fitness_value
        
    # generate a neighboring state
    # analog of mutation
    def neighbor(self):

        start = time.time() if config.LOG_TIME else None
        start_time = time.time()  # Always track time for CSV logging

        chosen_operation = weighted_choice(config.MUTATION_WEIGHTS)
        self.last_operation = chosen_operation

        # Save original MetaGarment state before neighbor generation
        original_design_params = None
        if self.meta_garment:
            original_design_params = copy.deepcopy(self.design_params)

        handler = {
            "split": self._split,
            "rotate": self._rotate,
            "swap": self._swap,
            "inversion": self._inversion,
            "insertion": self._insertion,
            "scramble": self._scramble,
            "design_params": self._design_params,
        }.get(chosen_operation)

        if handler is None:
            raise ValueError(f"Unknown operation: {chosen_operation}")

        # All handlers now return tuples (pieces, tracking_changes)
        generated_neighbor, tracking_changes = handler()

        # Evaluate fitness and get decoder for potential SVG saving
        new_fitness, new_decoder = self.fitness(generated_neighbor, return_decoder=True)
        curr_fitness = self.fitness(self.current_state)
        acceptance_prob = self._acceptance_probability(curr_fitness, new_fitness)
        accepted = self.accept(curr_fitness, new_fitness)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Log convergence data to CSV
        self.log_csv_data(
            operation=chosen_operation,
            old_fitness=curr_fitness,
            new_fitness=new_fitness, 
            acceptance_prob=acceptance_prob,
            accepted=accepted,
            time_elapsed=elapsed_time
        )
        
        # Increment iteration counter for tracking
        self.iteration_count += 1
        
        if accepted:
            self.current_state = generated_neighbor
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_state = copy.deepcopy(self.current_state)
                self.best_split_history = list(self.split_history)
                self.best_split_set = set(self.split_set)
                self.best_design_params = copy.deepcopy(self.design_params)
                self.log(f"[BEST] {self.best_fitness:.4f} with split roots: {sorted(self.best_split_set)}")
                
                # Save SVG of new best state (reusing the decoder for efficiency)
                self._save_state_svg(self.current_state, self.iteration_count, 
                                   decoder=new_decoder, suffix="_best")

            # Apply tracking changes only after acceptance
            self._apply_tracking_changes(tracking_changes)
        else:
            # Neighbor rejected - restore original MetaGarment state if it was mutated
            if original_design_params is not None and self.design_params != original_design_params:
                self.log(f"[TRACKING DEBUG] Restoring original design params after rejection")
                self.design_params = original_design_params
                # Recreate MetaGarment with original parameters and regenerate pieces
                if self.design_params and self.body_params:
                    try:
                        from assets.garment_programs.meta_garment import MetaGarment
                        from nesting.path_extractor import PatternPathExtractor
                        from pathlib import Path
                        import tempfile

                        self.meta_garment = MetaGarment("sa_restored", self.body_params, self.design_params)
                        # Reapply accepted splits from split_history
                        for panel_name, proportion in self.split_history:
                            try:
                                self.meta_garment.split_panel(panel_name, proportion)
                            except Exception as e:
                                self.log(f"[TRACKING DEBUG] Warning: Could not restore split {panel_name}: {e}")

                        # CRITICAL: Regenerate pieces from the restored MetaGarment to match its state
                        pattern = self.meta_garment.assembly()
                        with tempfile.TemporaryDirectory() as td:
                            spec_file = Path(td) / f"{pattern.name}_specification.json"
                            pattern.serialize(Path(td), to_subfolder=False, with_3d=False,
                                            with_text=False, view_ids=False)
                            extractor = PatternPathExtractor(spec_file)
                            all_pieces = extractor.get_all_panel_pieces(
                                samples_per_edge=getattr(config, 'SAMPLES_PER_EDGE', 10)
                            )

                        # Update current_state to match the restored MetaGarment
                        new_pieces = []
                        for piece_id, piece in all_pieces.items():
                            # Set parent_id and root_id correctly
                            if "_split_" in piece_id:
                                # Split pieces: parent_id = original piece, root_id = root piece
                                original_root_id = piece_id.split("_split_")[0]
                                piece.parent_id = original_root_id  # Parent is the original unsplit piece
                                piece.root_id = original_root_id    # Root is also the original piece
                            else:
                                # Original unsplit pieces: no parent, root is self
                                piece.parent_id = None
                                piece.root_id = piece_id
                            new_pieces.append(piece)

                        self.current_state = new_pieces
                        self.log(f"[TRACKING DEBUG] Restored {len(new_pieces)} pieces to match MetaGarment state")

                    except Exception as exc:
                        self.log(f"[TRACKING DEBUG] Failed to restore MetaGarment: {exc}")

        if config.LOG_TIME:
            end = time.time()
            elapsed = end - start
            self.log(f"Neighbor generation ({chosen_operation}) took {elapsed:.6f} seconds")

    def accept(self, old_fitness, new_fitness):
        ap = self._acceptance_probability(old_fitness, new_fitness)
        self.log(f"Old fitness: {old_fitness:.4f}, New fitness: {new_fitness:.4f}, Temp: {self.temperature:.4f}, Acceptance Probability: {ap:.4f}")
        return ap > random.random()

    # acceptance probability
    def _acceptance_probability(self, old_fitness, new_fitness):
        if new_fitness > old_fitness:
            return 1.0
        return math.exp((new_fitness - old_fitness) / self.temperature)


    def _apply_tracking_changes(self, tracking_changes):
        """Apply tracking changes after a solution is accepted"""
        if not tracking_changes:
            return

        self.log(f"[TRACKING DEBUG] Applying tracking changes: {tracking_changes}")
        self.log(f"[TRACKING DEBUG] Before changes - split_set: {self.split_set}")

        # Handle split_set additions
        if 'split_set_additions' in tracking_changes:
            for piece_id in tracking_changes['split_set_additions']:
                self.log(f"[TRACKING DEBUG] Adding {piece_id} to split_set")
                self.split_set.add(piece_id)

        # Handle split_set replacement (used by design_params)
        if 'split_set_replacement' in tracking_changes:
            old_split_set = self.split_set.copy()
            self.log(f"[TRACKING DEBUG] REPLACING split_set entirely")
            self.log(f"[TRACKING DEBUG] Old split_set: {old_split_set}")
            self.log(f"[TRACKING DEBUG] New split_set will be: {tracking_changes['split_set_replacement']}")

            self.split_set.clear()
            for piece_id in tracking_changes['split_set_replacement']:
                self.split_set.add(piece_id)

            lost_items = old_split_set - self.split_set
            gained_items = self.split_set - old_split_set
            if lost_items:
                self.log(f"[TRACKING DEBUG] WARNING: Lost items in replacement: {lost_items}")
            if gained_items:
                self.log(f"[TRACKING DEBUG] INFO: Gained items in replacement: {gained_items}")

        # Handle split_history additions
        if 'split_history_additions' in tracking_changes:
            for entry in tracking_changes['split_history_additions']:
                self.log(f"[TRACKING DEBUG] Adding to split_history: {entry}")
                self.split_history.append(entry)

        # Handle pending MetaGarment splits - apply them to the live MetaGarment on acceptance
        if 'pending_mg_splits' in tracking_changes and self.meta_garment:
            self.log(f"[TRACKING DEBUG] Applying {len(tracking_changes['pending_mg_splits'])} splits to live MetaGarment")
            for piece_id, root_id, proportion in tracking_changes['pending_mg_splits']:
                try:
                    self.log(f"[TRACKING DEBUG] Applying split: {piece_id} (root: {root_id}) with proportion {proportion}")
                    # Try to split using the piece.id first, then root_id if that fails
                    try:
                        self.meta_garment.split_panel(piece_id, proportion=proportion)
                        self.log(f"[TRACKING DEBUG] Successfully applied split using piece_id: {piece_id}")
                    except Exception as e1:
                        self.log(f"[TRACKING DEBUG] Split failed with piece_id '{piece_id}': {e1}")
                        try:
                            self.meta_garment.split_panel(root_id, proportion=proportion)
                            self.log(f"[TRACKING DEBUG] Successfully applied split using root_id: {root_id}")
                        except Exception as e2:
                            self.log(f"[TRACKING DEBUG] Split also failed with root_id '{root_id}': {e2}")
                            # Don't fail the entire operation - just log and continue
                except Exception as e:
                    self.log(f"[TRACKING DEBUG] Error applying split {piece_id}: {e}")

        self.log(f"[TRACKING DEBUG] After changes - split_set: {self.split_set}")
        self.log(f"[TRACKING DEBUG] After changes - split_history length: {len(self.split_history)}")
    
    def termination_condition(self):
        return self.temperature < 1e-3
    
    def cool_down(self):
        old_temp = self.temperature
        self.temperature *= self.cooling_rate
        self.temperature_level += 1  # Track temperature level for CSV logging
        self.log(f"Cooling down: Temp {old_temp:.4f} -> {self.temperature:.4f}")

    # operations to generate neighbors

    def _split(self) -> tuple[List[Piece], dict]:
        import copy
        import random
        
        # Create a copy of current state to modify
        new_state = copy.deepcopy(self.current_state)
        
        def _unsplit_roots():
            # Only return pieces that:
            # 1. Have no parent (are root pieces)
            # 2. Haven't been split before (check by ID)
            # 3. Don't have "_split_" in their ID (not already a split piece)
            candidates = [g for g in new_state
                         if g.parent_id is None
                         and g.id not in self.split_set
                         and "_split_" not in g.id]
            self.log(f"[SimulatedAnnealing] DEBUG: Split candidates: {[c.id for c in candidates]}")
            self.log(f"[SimulatedAnnealing] DEBUG: Split_set has {len(self.split_set)} piece IDs: {self.split_set}")

            # Debug candidate selection
            all_root_pieces = [g for g in new_state if g.parent_id is None]
            self.log(f"[TRACKING DEBUG] All root pieces: {[g.id for g in all_root_pieces]}")
            for g in all_root_pieces:
                in_split_set = g.id in self.split_set
                has_split_in_id = "_split_" in g.id
                self.log(f"[TRACKING DEBUG] Piece {g.id}: in_split_set={in_split_set}, has_split_in_id={has_split_in_id}, root_id={getattr(g, 'root_id', 'NO_ROOT_ID')}")

            return candidates
        
        candidates = _unsplit_roots()
        
        # If no candidates for splitting, return current state
        if not candidates and not getattr(config, 'ALLOW_RECURSIVE_SPLITS', False):
            return self.current_state, {}

        if not candidates:
            return self.current_state, {}
            
        # Randomly pick a piece from the candidates
        if getattr(config, 'WEIGHT_BY_BBOX', False):
            weights = [c.bbox_area for c in candidates]
            piece = random.choices(candidates, weights=weights, k=1)[0]
        else:
            piece = random.choice(candidates)

        # Track changes to return instead of applying directly
        tracking_changes = {'split_set_additions': [piece.id], 'split_history_additions': []}
        self.log(f"[TRACKING DEBUG] Regular split will add to split_set: {piece.id}")
        self.log(f"[TRACKING DEBUG] piece.id={piece.id}, piece.root_id={getattr(piece, 'root_id', 'NO_ROOT_ID')}")
        
        # Mirror lookup (basic left/right root swap)
        piece_mirror = next((p for p in new_state
                              if p.parent_id is None and p is not piece and (
                                  p.id == piece.id.replace("left", "right") or
                                  p.id == piece.id.replace("right", "left"))), None)
        
        if not self.meta_garment:  # no parameter pipeline
            left, right = piece.split()
            new_state = self._replace_piece(piece, [left, right], new_state)
            if piece_mirror and getattr(config, 'SYMMETRIC_SPLITS', False):
                tracking_changes['split_set_additions'].append(piece_mirror.id)
                left_mirror, right_mirror = piece_mirror.split()
                new_state = self._replace_piece(piece_mirror, [left_mirror, right_mirror], new_state)
        else:  # pipeline with design and body parameters
            try:
                success, meta_tracking = self._meta_split(piece, new_state.index(piece), piece_mirror if getattr(config, 'SYMMETRIC_SPLITS', False) else None, new_state)
                if not success:
                    return self.current_state, {}
                # Merge tracking changes from meta_split
                tracking_changes['split_history_additions'].extend(meta_tracking.get('split_history_additions', []))
                tracking_changes['split_set_additions'].extend(meta_tracking.get('split_set_additions', []))
            except Exception as e:
                self.log(f"[SimulatedAnnealing] MetaGarment split failed for {piece.id}: {e}")
                return self.current_state, {}
        
        return new_state, tracking_changes
        
    def _replace_piece(self, old_piece, new_pieces, state):
        """Replace a piece with new pieces in the state list"""
        try:
            idx = state.index(old_piece)
            return state[:idx] + new_pieces + state[idx+1:]
        except ValueError:
            return state
            
    def _meta_split(self, piece, idx, piece_mirror, state):
        """Adapted from chromosome.meta_split method"""
        from nesting.path_extractor import PatternPathExtractor
        from assets.garment_programs.meta_garment import MetaGarment
        from pathlib import Path
        import tempfile
        import random
        import copy

        # Create temporary MetaGarment for proposal - don't mutate the live one
        temp_design_params = copy.deepcopy(self.design_params)
        temp_mg = MetaGarment('temp_meta_garment', self.body_params, temp_design_params)
        
        # choose split proportion and perform
        proportion = random.uniform(getattr(config, 'SPLIT_LOWER_BOUND', 0.3), getattr(config, 'SPLIT_UPPER_BOUND', 0.7))
        
        self.log(f"[SimulatedAnnealing] Attempting to split panel ID: '{piece.id}' with proportion {proportion}")
        self.log(f"[SimulatedAnnealing] Piece root_id: '{piece.root_id}'")

        # Defensive validation - check if panel already appears to be split
        try:
            current_pattern = temp_mg.assembly()
            panel_names = list(current_pattern.pattern['panels'].keys()) if hasattr(current_pattern, 'pattern') else []
            already_split_variants = [p for p in panel_names if p.startswith(f"{piece.id}_split_") or p.startswith(f"{piece.root_id}_split_")]
            if already_split_variants:
                self.log(f"[TRACKING DEBUG] WARNING: Panel appears already split in MetaGarment: {already_split_variants}")
                self.log(f"[TRACKING DEBUG] But piece.id={piece.id} not in split_set: {self.split_set}")
                self.log(f"[TRACKING DEBUG] This suggests a tracking sync issue!")
        except Exception as debug_e:
            self.log(f"[TRACKING DEBUG] Could not check for existing splits: {debug_e}")

        # Debug: Check what panels are available in the MetaGarment
        try:
            # Get the pattern to see what panels exist
            current_pattern = temp_mg.assembly()
            panel_names = list(current_pattern.pattern['panels'].keys()) if hasattr(current_pattern, 'pattern') else []
            self.log(f"[SimulatedAnnealing] Available panels in MetaGarment: {panel_names}")
        except Exception as debug_e:
            self.log(f"[SimulatedAnnealing] Could not get panel list: {debug_e}")

        # Try to split using the piece.id first
        try:
            new_panel_names = temp_mg.split_panel(piece.id, proportion=proportion)
        except Exception as e:
            self.log(f"[SimulatedAnnealing] Split failed with piece.id '{piece.id}': {e}")
            # If that fails, try with root_id
            try:
                self.log(f"[SimulatedAnnealing] Retrying split with root_id '{piece.root_id}'")
                new_panel_names = temp_mg.split_panel(piece.root_id, proportion=proportion)
            except Exception as e2:
                self.log(f"[SimulatedAnnealing] Split also failed with root_id '{piece.root_id}': {e2}")
                return False, {}
        
        if not new_panel_names:
            self.log(f"[SimulatedAnnealing] MetaGarment split failed for {piece.id}: no new panels returned")
            return False, {}
        
        # Track changes to return instead of applying directly
        # Include the split operations so they can be applied to live MetaGarment on acceptance
        pending_splits = [(piece.id, piece.root_id, proportion)]
        tracking_changes = {
            'split_history_additions': [(piece.root_id, proportion)],
            'split_set_additions': [],
            'pending_mg_splits': pending_splits  # New field for MetaGarment operations
        }
        self.log(f"[SimulatedAnnealing] Split {piece.id} into {new_panel_names} with proportion {proportion}")
        self.log(f"[TRACKING DEBUG] Will add to split_history: ({piece.root_id}, {proportion})")
        self.log(f"[TRACKING DEBUG] piece.id={piece.id}, piece.root_id={piece.root_id}")

        # Rebuild pattern and extract pieces from temporary MetaGarment
        pattern = temp_mg.assembly()
        with tempfile.TemporaryDirectory() as td:
            spec_file = Path(td) / f"{pattern.name}_specification.json"
            pattern.serialize(Path(td), to_subfolder=False, with_3d=False,
                            with_text=False, view_ids=False)
            extractor = PatternPathExtractor(spec_file)
            all_pieces = extractor.get_all_panel_pieces(
                samples_per_edge=getattr(config, 'SAMPLES_PER_EDGE', 10)
            )
        
        # Build new Piece objects for the split panels
        left_piece = all_pieces.get(new_panel_names[0]) if new_panel_names else None
        right_piece = all_pieces.get(new_panel_names[1]) if new_panel_names else None

        if left_piece is None or right_piece is None:
            self.log(f"[SimulatedAnnealing] Split pieces not found for {piece.id} in regenerated pattern")
            return False, {}

        # CRITICAL FIX: Set parent_id and root_id correctly for split pieces
        # Split pieces should have parent_id pointing to the original piece and root_id to the root
        original_piece_id = piece.id
        original_root_id = piece.root_id if piece.root_id else piece.id

        # Set parent-child relationships
        left_piece.parent_id = original_piece_id
        right_piece.parent_id = original_piece_id
        left_piece.root_id = original_root_id
        right_piece.root_id = original_root_id
        self.log(f"[TRACKING DEBUG] Set parent_id={original_piece_id}, root_id={original_root_id} for split pieces {left_piece.id} and {right_piece.id}")

        # Replace the original piece with the split pieces
        new_pieces = [left_piece, right_piece]
        state[:] = self._replace_piece(piece, new_pieces, state)
        
        return True, tracking_changes

    def _rotate(self) -> tuple[List[Piece], dict]:
        return Operators.rotate(self.current_state), {}

    def _swap(self, k=None) -> tuple[List[Piece], dict]:
        k = k or getattr(config, 'SWAP_MUTATION_K', None)
        return Operators.swap(self.current_state, k), {}

    def _inversion(self) -> tuple[List[Piece], dict]:
        return Operators.inversion(self.current_state), {}

    def _insertion(self) -> tuple[List[Piece], dict]:
        return Operators.insertion(self.current_state), {}

    def _scramble(self) -> tuple[List[Piece], dict]:
        return Operators.scramble(self.current_state), {}

    def _design_params(self) -> tuple[List[Piece], dict]:
        if not (self.design_params and self.body_params):
            return self.current_state, {}

        self.log(f"[SimulatedAnnealing] Starting design parameter mutation")
        
        # # Debug: Check current MetaGarment state before design param change
        # if self.meta_garment:
        #     try:
        #         current_pattern = self.meta_garment.assembly()
        #         panel_names = [panel.name for panel in current_pattern.patterns]
        #         print(f"[SimulatedAnnealing] Current panels before design param change: {panel_names}")
        #     except Exception as debug_e:
        #         print(f"[SimulatedAnnealing] Could not get panel list before design param change: {debug_e}")
        
        # Create fitness function for the shared operation
        def fitness_fn(pieces):
            metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
            return metric_fn(pieces, config.SELECTED_DECODER, self.container)

        new_pieces, new_design_params, success = Operators.design_params(
            self.current_state,
            self.design_params,
            self.body_params,
            self.initial_design_params,
            getattr(self, 'split_history', []),
            fitness_fn
        )

        if success:
            # Update design params - identical to chromosome behavior
            self.design_params = new_design_params
            
            # CRITICAL: Update SA's internal state to match the new garment structure
            # The design parameter operation has already regenerated the garment and reapplied splits
            # We just need to create a fresh MetaGarment that matches the returned pieces structure
            if self.design_params and self.body_params:
                try:
                    from assets.garment_programs.meta_garment import MetaGarment
                    # Create a new MetaGarment with updated design params
                    self.meta_garment = MetaGarment("sa_updated", self.body_params, self.design_params)
                    
                    # Apply the same splits to SA's MetaGarment to match the returned pieces
                    # (The design parameter operation already applied these splits to create the new_pieces)
                    for panel_name, proportion in self.split_history:
                        try:
                            self.meta_garment.split_panel(panel_name, proportion)
                        except Exception as e:
                            self.log(f"[SimulatedAnnealing] Warning: Could not sync split {panel_name}: {e}")
                    
                    # Prepare tracking changes for split_set update
                    split_panel_names = {panel_name for panel_name, _ in self.split_history}

                    self.log(f"[TRACKING DEBUG] Building split_set_replacement from split_history")
                    self.log(f"[TRACKING DEBUG] Current split_history: {self.split_history}")
                    self.log(f"[TRACKING DEBUG] Extracted panel names: {split_panel_names}")
                    self.log(f"[TRACKING DEBUG] Current split_set before replacement: {self.split_set}")

                    # Check what pieces we actually have
                    piece_ids = {p.id for p in new_pieces}
                    split_piece_ids = {pid for pid in piece_ids if "_split_" in pid}
                    self.log(f"[TRACKING DEBUG] Actual split pieces in new_pieces: {split_piece_ids}")

                    # Check which root panels actually have splits
                    actually_split_roots = set()
                    for panel_name in split_panel_names:
                        has_splits = any(pid.startswith(f"{panel_name}_split_") for pid in piece_ids)
                        self.log(f"[TRACKING DEBUG] Panel {panel_name} has splits: {has_splits}")
                        if has_splits:
                            actually_split_roots.add(panel_name)

                    tracking_changes = {'split_set_replacement': list(split_panel_names)}

                    self.log(f"[TRACKING DEBUG] Would track {len(split_panel_names)} split root panel IDs: {split_panel_names}")
                    self.log(f"[TRACKING DEBUG] Actually split roots found in pieces: {actually_split_roots}")
                    self.log(f"[TRACKING DEBUG] All pieces after sync: {[p.id for p in new_pieces]}")
                    
                    # Debug: Check final state
                    final_pattern = self.meta_garment.assembly()
                    final_panels = list(final_pattern.pattern['panels'].keys())
                    split_panels = [p for p in final_panels if "_split_" in p]
                    self.log(f"[SimulatedAnnealing] Synced MetaGarment with current structure ({len(self.split_history)} splits applied)")
                    self.log(f"[SimulatedAnnealing] DEBUG: Final MetaGarment has {len(split_panels)} split panels: {split_panels}")
                except Exception as exc:
                    self.log(f"[SimulatedAnnealing] Warning: Could not sync MetaGarment: {exc}")

            # CRITICAL: Normalize parent_id and root_id for all pieces from design_params
            for piece in new_pieces:
                if "_split_" in piece.id:
                    # Split pieces: parent_id = original piece, root_id = root piece
                    original_root_id = piece.id.split("_split_")[0]
                    piece.parent_id = original_root_id
                    piece.root_id = original_root_id
                else:
                    # Original unsplit pieces: no parent, root is self
                    piece.parent_id = None
                    piece.root_id = piece.id

            return new_pieces, tracking_changes

        return self.current_state, {}

    def apply_best(self):
        # Restore trackers
        self.current_state = copy.deepcopy(getattr(self, "best_state", self.current_state))
        self.split_history = list(getattr(self, "best_split_history", self.split_history))
        self.split_set = set(getattr(self, "best_split_set", self.split_set))
        self.design_params = copy.deepcopy(getattr(self, "best_design_params", self.design_params))

        self.log(f"[BEST RESTORE] Restoring best state with {len(self.split_history)} splits")
        self.log(f"[BEST RESTORE] Split history: {self.split_history}")
        self.log(f"[BEST RESTORE] Split set: {self.split_set}")

        # Rebuild MetaGarment from the best snapshot
        if self.design_params and self.body_params:
            from assets.garment_programs.meta_garment import MetaGarment
            from nesting.path_extractor import PatternPathExtractor
            from pathlib import Path
            import tempfile

            mg = MetaGarment("best", self.body_params, self.design_params)
            self.log(f"[BEST RESTORE] Applying {len(self.split_history)} splits: {self.split_history}")
            for i, (root, prop) in enumerate(self.split_history):
                try:
                    self.log(f"[BEST RESTORE] Split {i+1}/{len(self.split_history)}: {root} with proportion {prop}")
                    mg.split_panel(root, prop)
                    self.log(f"[BEST RESTORE] Successfully applied split {root}")
                except Exception as e:
                    self.log(f"[BEST RESTORE] Could not reapply split {root}: {e}")
                    # Continue with remaining splits even if one fails
            self.meta_garment = mg

            # Re-extract pieces to ensure perfect consistency with MG
            pattern = mg.assembly()
            with tempfile.TemporaryDirectory() as td:
                spec = Path(td) / f"{pattern.name}_specification.json"
                pattern.serialize(Path(td), to_subfolder=False, with_3d=False,
                                  with_text=False, view_ids=False)
                extractor = PatternPathExtractor(spec)
                all_pieces = extractor.get_all_panel_pieces(
                    samples_per_edge=getattr(config, 'SAMPLES_PER_EDGE', 10)
                )

            # Normalize metadata
            restored = []
            for pid, p in all_pieces.items():
                if "_split_" in pid:
                    base = pid.split("_split_")[0]
                    p.root_id = base
                    p.parent_id = base
                else:
                    p.root_id = pid
                    p.parent_id = None
                restored.append(p)
            self.current_state = restored

    def run(self):
        while not self.termination_condition():
            # Run multiple iterations at current temperature
            for _ in range(self.iterations_per_temp):
                self.neighbor()
                # Break early if we've reached termination condition during iterations
                if self.termination_condition():
                    break
            # Cool down after all iterations at this temperature
            self.cool_down()
        
        self.log(f"Final fitness: {self.best_fitness:.4f}")
        self.apply_best()   # <— make best the live state shown to GUI
        
        # Save final best state SVG
        # self._save_state_svg(self.current_state, self.iteration_count, suffix="_final")
        
        # Final log flush to ensure all logs are written to disk
        self._flush_log()
        
        # Final CSV summary
        self._finalize_csv_logging()