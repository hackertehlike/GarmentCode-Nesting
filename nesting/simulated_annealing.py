import math
import time
import random
import copy
from typing import Optional, Union, List

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
        self.best_state = pieces
        self.best_fitness = float('-inf')

        self.split_history = []
        self.split_set = set()  # Track IDs of split pieces, not objects

        self.meta_garment = None
        if self.design_params and self.body_params:
            try:
                from assets.garment_programs.meta_garment import MetaGarment
                self.meta_garment = MetaGarment("metagarment", self.body_params, self.design_params)
            except Exception as exc:
                print(f"[Chromosome] Failed to create MetaGarment: {exc}")

    # evaluate fitness of a state
    def fitness(self, state: List[Piece]) -> float:
        metric_fn = METRIC_REGISTRY[config.SELECTED_FITNESS_METRIC]
        return metric_fn(state, config.SELECTED_DECODER, self.container)
        
    # generate a neighboring state
    # analog of mutation
    def neighbor(self):

        start = time.time() if config.LOG_TIME else None

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

        new_fitness = self.fitness(generated_neighbor)
        curr_fitness = self.fitness(self.current_state)
        if self.accept(curr_fitness, new_fitness):
            self.current_state = generated_neighbor
            if new_fitness > self.best_fitness:
                self.best_fitness = new_fitness
                self.best_state = self.current_state

            # Apply tracking changes only after acceptance
            self._apply_tracking_changes(tracking_changes)
        else:
            # Neighbor rejected - restore original MetaGarment state if it was mutated
            if original_design_params is not None and self.design_params != original_design_params:
                print(f"[TRACKING DEBUG] Restoring original design params after rejection")
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
                                print(f"[TRACKING DEBUG] Warning: Could not restore split {panel_name}: {e}")

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
                        print(f"[TRACKING DEBUG] Restored {len(new_pieces)} pieces to match MetaGarment state")

                    except Exception as exc:
                        print(f"[TRACKING DEBUG] Failed to restore MetaGarment: {exc}")

        if config.LOG_TIME:
            end = time.time()
            elapsed = end - start
            print(f"Neighbor generation ({chosen_operation}) took {elapsed:.6f} seconds")

    def accept(self, old_fitness, new_fitness):
        ap = self._acceptance_probability(old_fitness, new_fitness)
        print(f"Old fitness: {old_fitness:.4f}, New fitness: {new_fitness:.4f}, Temp: {self.temperature:.4f}, Acceptance Probability: {ap:.4f}")
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

        print(f"[TRACKING DEBUG] Applying tracking changes: {tracking_changes}")
        print(f"[TRACKING DEBUG] Before changes - split_set: {self.split_set}")

        # Handle split_set additions
        if 'split_set_additions' in tracking_changes:
            for piece_id in tracking_changes['split_set_additions']:
                print(f"[TRACKING DEBUG] Adding {piece_id} to split_set")
                self.split_set.add(piece_id)

        # Handle split_set replacement (used by design_params)
        if 'split_set_replacement' in tracking_changes:
            old_split_set = self.split_set.copy()
            print(f"[TRACKING DEBUG] REPLACING split_set entirely")
            print(f"[TRACKING DEBUG] Old split_set: {old_split_set}")
            print(f"[TRACKING DEBUG] New split_set will be: {tracking_changes['split_set_replacement']}")

            self.split_set.clear()
            for piece_id in tracking_changes['split_set_replacement']:
                self.split_set.add(piece_id)

            lost_items = old_split_set - self.split_set
            gained_items = self.split_set - old_split_set
            if lost_items:
                print(f"[TRACKING DEBUG] WARNING: Lost items in replacement: {lost_items}")
            if gained_items:
                print(f"[TRACKING DEBUG] INFO: Gained items in replacement: {gained_items}")

        # Handle split_history additions
        if 'split_history_additions' in tracking_changes:
            for entry in tracking_changes['split_history_additions']:
                print(f"[TRACKING DEBUG] Adding to split_history: {entry}")
                self.split_history.append(entry)

        print(f"[TRACKING DEBUG] After changes - split_set: {self.split_set}")
        print(f"[TRACKING DEBUG] After changes - split_history length: {len(self.split_history)}")
    
    def termination_condition(self):
        return self.temperature < 1e-3
    
    def cool_down(self):
        print(f"Cooling down: Temp {self.temperature:.4f} -> ", end="")
        self.temperature *= self.cooling_rate
        print(f"{self.temperature:.4f}")

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
            print(f"[SimulatedAnnealing] DEBUG: Split candidates: {[c.id for c in candidates]}")
            print(f"[SimulatedAnnealing] DEBUG: Split_set has {len(self.split_set)} piece IDs: {self.split_set}")

            # Debug candidate selection
            all_root_pieces = [g for g in new_state if g.parent_id is None]
            print(f"[TRACKING DEBUG] All root pieces: {[g.id for g in all_root_pieces]}")
            for g in all_root_pieces:
                in_split_set = g.id in self.split_set
                has_split_in_id = "_split_" in g.id
                print(f"[TRACKING DEBUG] Piece {g.id}: in_split_set={in_split_set}, has_split_in_id={has_split_in_id}, root_id={getattr(g, 'root_id', 'NO_ROOT_ID')}")

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
        print(f"[TRACKING DEBUG] Regular split will add to split_set: {piece.id}")
        print(f"[TRACKING DEBUG] piece.id={piece.id}, piece.root_id={getattr(piece, 'root_id', 'NO_ROOT_ID')}")
        
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
                print(f"[SimulatedAnnealing] MetaGarment split failed for {piece.id}: {e}")
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
        
        mg = self.meta_garment
        
        # choose split proportion and perform
        proportion = random.uniform(getattr(config, 'SPLIT_LOWER_BOUND', 0.3), getattr(config, 'SPLIT_UPPER_BOUND', 0.7))
        
        print(f"[SimulatedAnnealing] Attempting to split panel ID: '{piece.id}' with proportion {proportion}")
        print(f"[SimulatedAnnealing] Piece root_id: '{piece.root_id}'")

        # Defensive validation - check if panel already appears to be split
        try:
            current_pattern = mg.assembly()
            panel_names = list(current_pattern.pattern['panels'].keys()) if hasattr(current_pattern, 'pattern') else []
            already_split_variants = [p for p in panel_names if p.startswith(f"{piece.id}_split_") or p.startswith(f"{piece.root_id}_split_")]
            if already_split_variants:
                print(f"[TRACKING DEBUG] WARNING: Panel appears already split in MetaGarment: {already_split_variants}")
                print(f"[TRACKING DEBUG] But piece.id={piece.id} not in split_set: {self.split_set}")
                print(f"[TRACKING DEBUG] This suggests a tracking sync issue!")
        except Exception as debug_e:
            print(f"[TRACKING DEBUG] Could not check for existing splits: {debug_e}")
        
        # Debug: Check what panels are available in the MetaGarment
        try:
            # Get the pattern to see what panels exist
            current_pattern = mg.assembly()
            panel_names = list(current_pattern.pattern['panels'].keys()) if hasattr(current_pattern, 'pattern') else []
            print(f"[SimulatedAnnealing] Available panels in MetaGarment: {panel_names}")
        except Exception as debug_e:
            print(f"[SimulatedAnnealing] Could not get panel list: {debug_e}")
        
        # Try to split using the piece.id first
        try:
            new_panel_names = mg.split_panel(piece.id, proportion=proportion)
        except Exception as e:
            print(f"[SimulatedAnnealing] Split failed with piece.id '{piece.id}': {e}")
            # If that fails, try with root_id
            try:
                print(f"[SimulatedAnnealing] Retrying split with root_id '{piece.root_id}'")
                new_panel_names = mg.split_panel(piece.root_id, proportion=proportion)
            except Exception as e2:
                print(f"[SimulatedAnnealing] Split also failed with root_id '{piece.root_id}': {e2}")
                return False, {}
        
        if not new_panel_names:
            print(f"[SimulatedAnnealing] MetaGarment split failed for {piece.id}: no new panels returned")
            return False, {}
        
        # Track changes to return instead of applying directly
        tracking_changes = {'split_history_additions': [(piece.root_id, proportion)], 'split_set_additions': []}
        print(f"[SimulatedAnnealing] Split {piece.id} into {new_panel_names} with proportion {proportion}")
        print(f"[TRACKING DEBUG] Will add to split_history: ({piece.root_id}, {proportion})")
        print(f"[TRACKING DEBUG] piece.id={piece.id}, piece.root_id={piece.root_id}")
        
        new_panel_names_mirror = None
        if piece_mirror and getattr(config, 'SYMMETRIC_SPLITS', False):
            # If there's a mirror piece, also split it with complementary proportion
            mirror_proportion = 1 - proportion
            new_panel_names_mirror = mg.split_panel(piece_mirror.id, proportion=mirror_proportion)
            tracking_changes['split_set_additions'].append(piece_mirror.id)  # track mirror piece as split
            if not new_panel_names_mirror:
                print(f"[SimulatedAnnealing] MetaGarment split failed for mirror {piece_mirror.id}")
            else:
                # record mirror split using ROOT id
                tracking_changes['split_history_additions'].append((piece_mirror.root_id, mirror_proportion))
                print(f"[SimulatedAnnealing] Split {piece_mirror.id} into {new_panel_names_mirror} with proportion {mirror_proportion}")
        
        # Rebuild pattern and extract pieces
        pattern = mg.assembly()
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
            print(f"[SimulatedAnnealing] Split pieces not found for {piece.id} in regenerated pattern")
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
        print(f"[TRACKING DEBUG] Set parent_id={original_piece_id}, root_id={original_root_id} for split pieces {left_piece.id} and {right_piece.id}")

        # Replace the original piece with the split pieces
        new_pieces = [left_piece, right_piece]
        state[:] = self._replace_piece(piece, new_pieces, state)
        
        # Handle mirror piece if it exists
        if piece_mirror and new_panel_names_mirror:
            left_piece_mirror = all_pieces.get(new_panel_names_mirror[0])
            right_piece_mirror = all_pieces.get(new_panel_names_mirror[1])
            if left_piece_mirror and right_piece_mirror:
                # Set parent_id and root_id for mirror pieces too
                original_mirror_piece_id = piece_mirror.id
                original_mirror_root_id = piece_mirror.root_id if piece_mirror.root_id else piece_mirror.id
                left_piece_mirror.parent_id = original_mirror_piece_id
                right_piece_mirror.parent_id = original_mirror_piece_id
                left_piece_mirror.root_id = original_mirror_root_id
                right_piece_mirror.root_id = original_mirror_root_id
                print(f"[TRACKING DEBUG] Set parent_id={original_mirror_piece_id}, root_id={original_mirror_root_id} for mirror split pieces {left_piece_mirror.id} and {right_piece_mirror.id}")

                new_pieces_mirror = [left_piece_mirror, right_piece_mirror]
                state[:] = self._replace_piece(piece_mirror, new_pieces_mirror, state)
        
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

        print(f"[SimulatedAnnealing] Starting design parameter mutation")
        
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
                            print(f"[SimulatedAnnealing] Warning: Could not sync split {panel_name}: {e}")
                    
                    # Prepare tracking changes for split_set update
                    split_panel_names = {panel_name for panel_name, _ in self.split_history}

                    print(f"[TRACKING DEBUG] Building split_set_replacement from split_history")
                    print(f"[TRACKING DEBUG] Current split_history: {self.split_history}")
                    print(f"[TRACKING DEBUG] Extracted panel names: {split_panel_names}")
                    print(f"[TRACKING DEBUG] Current split_set before replacement: {self.split_set}")

                    # Check what pieces we actually have
                    piece_ids = {p.id for p in new_pieces}
                    split_piece_ids = {pid for pid in piece_ids if "_split_" in pid}
                    print(f"[TRACKING DEBUG] Actual split pieces in new_pieces: {split_piece_ids}")

                    # Check which root panels actually have splits
                    actually_split_roots = set()
                    for panel_name in split_panel_names:
                        has_splits = any(pid.startswith(f"{panel_name}_split_") for pid in piece_ids)
                        print(f"[TRACKING DEBUG] Panel {panel_name} has splits: {has_splits}")
                        if has_splits:
                            actually_split_roots.add(panel_name)

                    tracking_changes = {'split_set_replacement': list(split_panel_names)}

                    print(f"[TRACKING DEBUG] Would track {len(split_panel_names)} split root panel IDs: {split_panel_names}")
                    print(f"[TRACKING DEBUG] Actually split roots found in pieces: {actually_split_roots}")
                    print(f"[TRACKING DEBUG] All pieces after sync: {[p.id for p in new_pieces]}")
                    
                    # Debug: Check final state
                    final_pattern = self.meta_garment.assembly()
                    final_panels = list(final_pattern.pattern['panels'].keys())
                    split_panels = [p for p in final_panels if "_split_" in p]
                    print(f"[SimulatedAnnealing] Synced MetaGarment with current structure ({len(self.split_history)} splits applied)")
                    print(f"[SimulatedAnnealing] DEBUG: Final MetaGarment has {len(split_panels)} split panels: {split_panels}")
                except Exception as exc:
                    print(f"[SimulatedAnnealing] Warning: Could not sync MetaGarment: {exc}")

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