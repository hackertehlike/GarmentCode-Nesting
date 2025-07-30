from copy import deepcopy
import numpy as np

import pygarment as pyg
from assets.garment_programs.base_classes import BaseBottoms
from assets.garment_programs import bands


class PantPanel(pyg.Panel):
    def __init__(
            self, name, body, design, 
            length,
            waist, 
            hips,
            hips_depth,
            crotch_width,
            dart_position,
            match_top_int_to=None,
            hipline_ext=1,
            double_dart=False) -> None:
        """
            Basic pant panel with option to be fitted (with darts)
        """
        super().__init__(name)

        flare = body['leg_circ'] * (design['flare']['v']  - 1) / 4 
        hips_depth = hips_depth * hipline_ext

        hip_side_incl = np.deg2rad(body['_hip_inclination'])
        dart_depth = hips_depth * 0.8 

        # Crotch cotrols
        crotch_depth_diff =  body['crotch_hip_diff']
        crotch_extention = crotch_width

        # eval pants shape
        # TODO Return ruffle opportunity?

        # amount of extra fabric at waist
        w_diff = hips - waist   # Assume its positive since waist is smaller then hips
        # We distribute w_diff among the side angle and a dart 
        hw_shift = np.tan(hip_side_incl) * hips_depth
        # Small difference
        if hw_shift > w_diff:
            hw_shift = w_diff

        # --- Edges definition ---
        # Right
        if pyg.utils.close_enough(design['flare']['v'], 1):  # skip optimization
            right_bottom = pyg.Edge(    
                [-flare, 0], 
                [0, length]
            )
        else:
            right_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                [-flare, 0], 
                [0, length],
                target_tan1=np.array([0, 1]), 
                # initial guess places control point closer to the hips 
                initial_guess=[0.75, 0]
            )
        right_top = pyg.CurveEdgeFactory.curve_from_tangents(
            right_bottom.end,    
            [hw_shift, length + hips_depth],
            target_tan0=np.array([0, 1]),
            initial_guess=[0.5, 0]
        )
       
        top = pyg.Edge(
            right_top.end, 
            [w_diff + waist, length + hips_depth] 
        )

        crotch_top = pyg.Edge(
            top.end, 
            [hips, length + 0.45 * hips_depth]  # A bit higher than hip line
            # NOTE: The point should be lower than the minimum rise value (0.5)
        )
        crotch_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
            crotch_top.end,
            [hips + crotch_extention, length - crotch_depth_diff], 
            target_tan0=np.array([0, -1]),
            target_tan1=np.array([1, 0]),
            initial_guess=[0.5, -0.5] 
        )

        left = pyg.CurveEdgeFactory.curve_from_tangents(
            crotch_bottom.end,    
            [
                # NOTE "Magic value" (-2 cm) which we use to define default width:
                #   just a little behing the crotch point
                # NOTE: Ensuring same distance from the crotch point in both 
                #   front and back for matching curves
                crotch_bottom.end[0] - 2 + flare, 
                # NOTE: The inside edge either matches the length of the outside (0, normal case)
                # or when the inteded length is smaller than crotch depth,
                # inside edge covers of the inside leg a bit below the crotch (panties-like shorts)
                y:=min(0, length - crotch_depth_diff * 1.5)
            ], 
            target_tan1=[flare, y - crotch_bottom.end[1]],
            initial_guess=[0.3, 0]
        )

        self.edges = pyg.EdgeSequence(
            right_bottom, right_top, top, crotch_top, crotch_bottom, left
            ).close_loop()
        bottom = self.edges[-1]

        right_bottom.add_semantic_label('right_bottom')
        right_top.add_semantic_label('right_top')
        top.add_semantic_label('top')
        crotch_top.add_semantic_label('crotch_top')
        crotch_bottom.add_semantic_label('crotch_bottom')
        left.add_semantic_label('left')

        # Default placement
        self.set_pivot(crotch_bottom.end)
        self.translation = [-0.5, - hips_depth - crotch_depth_diff + 5, 0] 

        # Out interfaces (easier to define before adding a dart)
        self.interfaces = {
            'outside': pyg.Interface(
                self, 
                pyg.EdgeSequence(right_bottom, right_top), 
                ruffle=[1, hipline_ext]),
            'crotch': pyg.Interface(self, pyg.EdgeSequence(crotch_top, crotch_bottom)),
            'inside': pyg.Interface(self, left),
            'bottom': pyg.Interface(self, bottom)
        }

        # Add top dart
        # NOTE: Ruffle indicator to match to waistline proportion for correct balance line matching
        dart_width = w_diff - hw_shift  
        if w_diff > hw_shift:
            top_edges, int_edges = self.add_darts(
                top, dart_width, dart_depth, dart_position, double_dart=double_dart)
            self.interfaces['top'] = pyg.Interface(
                self, int_edges, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
            ) 
            self.edges.substitute(top, top_edges)
        else:
            self.interfaces['top'] = pyg.Interface(
                self, top, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
        ) 
        
        

    def add_darts(self, top, dart_width, dart_depth, dart_position, double_dart=False):
        
        if double_dart:
            # TODOLOW Avoid hardcoding for matching with the top?
            dist = dart_position * 0.5  # Dist between darts -> dist between centers
            offsets_mid = [
                - (dart_position + dist / 2 + dart_width / 2 + dart_width / 4),   
                - (dart_position - dist / 2) - dart_width / 4,
            ]

            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth * 0.9), # smaller
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth)
            ]
        else:
            offsets_mid = [
                - dart_position - dart_width / 2,
            ]
            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width, dart_depth)
            ]
        top_edges, int_edges = pyg.EdgeSequence(top), pyg.EdgeSequence(top)

        # Keep track of dart indices to add semantic labels later
        dart_indices = []
        
        import os
        debug_log_path = os.path.expanduser("~/codestuff/GarmentCode/dart_creation.log")
        with open(debug_log_path, "w") as debug_file:
            debug_file.write(f"Adding darts to panel {self.name}\n")
            debug_file.write(f"double_dart={double_dart}, dart_width={dart_width}, dart_depth={dart_depth}\n")
            debug_file.write(f"Initial top edge: {top.start} -> {top.end}\n")
        
        for i, (off, dart) in enumerate(zip(offsets_mid, darts)):
            left_edge_len = top_edges[-1].length()
            original_edges_count = len(top_edges)
            
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"\nAdding dart #{i+1}\n")
                debug_file.write(f"Offset: {off}, Left edge length: {left_edge_len}\n")
                debug_file.write(f"Original edges count: {original_edges_count}\n")
            
            # Add the dart
            top_edges, int_edges = self.add_dart(
                dart,
                top_edges[-1],
                offset=left_edge_len + off,
                edge_seq=top_edges, 
                int_edge_seq=int_edges
            )
            
            # Calculate the indices of the dart edges in the new sequence
            # The dart edges will be the newly added edges that replaced the last edge
            dart_start_idx = original_edges_count - 1  # Index of the edge that was replaced
            dart_end_idx = dart_start_idx + (len(top_edges) - original_edges_count + 1)
            dart_indices.append((dart_start_idx, dart_end_idx, i+1))
            
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"After adding dart, total edges: {len(top_edges)}\n")
                debug_file.write(f"Dart indices: start={dart_start_idx}, end={dart_end_idx}\n")

        # Add semantic labels to dart edges
        with open(debug_log_path, "a") as debug_file:
            debug_file.write("\n--- Adding semantic labels to dart edges ---\n")
            
        for start_idx, end_idx, dart_num in dart_indices:
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"Processing dart #{dart_num}: indices {start_idx} to {end_idx}\n")
                
            for j in range(start_idx, end_idx):
                # Skip interface edges (these are part of the dart perimeter)
                if top_edges[j] in int_edges:
                    #top_edges[j].add_semantic_label(f'dart_{dart_num}_interface')
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"  Edge {j} is an interface edge, skipping\n")
                    continue
                else:
                    # These are the inner edges of the dart
                    top_edges[j].add_semantic_label(f'dart_{dart_num}')
                    # Add more specific labels if needed
                    if j == start_idx:
                        top_edges[j].add_semantic_label(f'dart_{dart_num}_left')
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(f"Added label 'dart_{dart_num}_left' to edge {j}\n")
                            debug_file.write(f"Edge now has labels: {top_edges[j].semantic_labels}\n")
                    elif j == end_idx - 1:
                        top_edges[j].add_semantic_label(f'dart_{dart_num}_right')
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(f"Added label 'dart_{dart_num}_right' to edge {j}\n")
                            debug_file.write(f"Edge now has labels: {top_edges[j].semantic_labels}\n")
                    else:
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(f"Edge {j} labeled as dart_{dart_num} (middle edge)\n")

        return top_edges, int_edges

    def split(self, proportion=0.5):
        # check if back or front
        if self.name.startswith('pant_b_'):
            import numpy as np
            import os
            
            # Create a debug log file in a known location
            debug_log_path = os.path.expanduser("~/codestuff/GarmentCode/debug_split.log")
            with open(debug_log_path, "w") as debug_file:  # Use "w" to overwrite old logs
                debug_file.write(f"\n\n--- NEW SPLIT OPERATION ---\n")
                debug_file.write(f"DEBUG: PantPanel.split called with proportion={proportion}\n")
                debug_file.write(f"DEBUG: Panel name: {self.name}\n")
                
                # Comprehensive check of all edges and their semantic labels
                debug_file.write("\n--- COMPREHENSIVE EDGE INSPECTION ---\n")
                for i, edge in enumerate(self.edges):
                    debug_file.write(f"Edge {i}: {edge.start} -> {edge.end}, Labels: {edge.semantic_labels}\n")
                    for label in edge.semantic_labels:
                        if 'dart' in label:
                            debug_file.write(f"  Dart label breakdown: {label.split('_')}\n")
                debug_file.write("--- END EDGE INSPECTION ---\n\n")
                
                # Inspect interfaces
                debug_file.write("\n--- INTERFACE INSPECTION ---\n")
                for interface_name, interface in self.interfaces.items():
                    debug_file.write(f"Interface '{interface_name}' contains {len(interface.edges)} edges:\n")
                    for i, edge in enumerate(interface.edges):
                        debug_file.write(f"  Edge {i}: {edge.start} -> {edge.end}, Labels: {edge.semantic_labels}\n")
                debug_file.write("--- END INTERFACE INSPECTION ---\n\n")
            
            # Find all top edges (interface edges) and dart tips
            dart_tips = []
                
            # Find all top edges including both interface edges and dart edges
            all_top_edges = []
            with open(debug_log_path, "a") as debug_file:
                for edge in self.edges:
                    # Check if this is a top edge (either interface or dart)
                    if 'top' in edge.semantic_labels or any('dart' in label for label in edge.semantic_labels):
                        debug_file.write(f"DEBUG: Found top edge with labels: {edge.semantic_labels}\n")
                        all_top_edges.append(edge)
                
                # Find dart edges to get dart tips
                # Look for any edge with dart_1 or dart_2 AND left or right in their labels
                dart_left_edges = []
                dart_right_edges = []
                
                debug_file.write("\n--- DETAILED DART EDGE DETECTION ---\n")
                for edge in self.edges:
                    # Look for different patterns of dart labels
                    has_dart_1 = any('dart_1' in label for label in edge.semantic_labels)
                    has_dart_2 = any('dart_2' in label for label in edge.semantic_labels)
                    has_left = any('left' in label for label in edge.semantic_labels)
                    has_right = any('right' in label for label in edge.semantic_labels)
                    
                    # Also check for specific combined labels
                    has_dart_1_left = any('dart_1_left' in label for label in edge.semantic_labels)
                    has_dart_1_right = any('dart_1_right' in label for label in edge.semantic_labels)
                    has_dart_2_left = any('dart_2_left' in label for label in edge.semantic_labels)
                    has_dart_2_right = any('dart_2_right' in label for label in edge.semantic_labels)
                    
                    debug_file.write(f"Edge {edge.start} -> {edge.end}:\n")
                    debug_file.write(f"  Labels: {edge.semantic_labels}\n")
                    debug_file.write(f"  has_dart_1: {has_dart_1}, has_dart_2: {has_dart_2}\n")
                    debug_file.write(f"  has_left: {has_left}, has_right: {has_right}\n")
                    debug_file.write(f"  Specific: dart_1_left: {has_dart_1_left}, dart_1_right: {has_dart_1_right}\n")
                    debug_file.write(f"  Specific: dart_2_left: {has_dart_2_left}, dart_2_right: {has_dart_2_right}\n")
                    
                    # Use specific combined labels if available
                    if has_dart_1_left or has_dart_2_left or ((has_dart_1 or has_dart_2) and has_left):
                        dart_left_edges.append(edge)
                        debug_file.write(f"  Added to dart_left_edges\n")
                    elif has_dart_1_right or has_dart_2_right or ((has_dart_1 or has_dart_2) and has_right):
                        dart_right_edges.append(edge)
                        debug_file.write(f"  Added to dart_right_edges\n")
                    else:
                        debug_file.write(f"  Not a dart left/right edge\n")
                
                debug_file.write(f"Found {len(dart_left_edges)} left dart edges and {len(dart_right_edges)} right dart edges\n")
                debug_file.write("--- END DETAILED DART EDGE DETECTION ---\n\n")
            
            # Find dart tips based on dart edges that form the darts
            with open(debug_log_path, "a") as debug_file:
                debug_file.write("\n--- FINDING DART TIPS DIRECTLY FROM GEOMETRY ---\n")
                
                # Get all unique dart numbers from edge labels
                dart_numbers = set()
                for edge in all_top_edges:
                    for label in edge.semantic_labels:
                        if label.startswith('dart_'):
                            parts = label.split('_')
                            if len(parts) >= 2 and parts[1].isdigit():
                                dart_numbers.add(parts[1])
                
                debug_file.write(f"Found dart numbers: {dart_numbers}\n")
                
                # Group edges by dart number
                for dart_num in dart_numbers:
                    # Find all edges for this dart
                    dart_edges = [edge for edge in all_top_edges if any(label.startswith(f'dart_{dart_num}') for label in edge.semantic_labels)]
                    
                    debug_file.write(f"Found {len(dart_edges)} edges for dart_{dart_num}\n")
                    
                    if len(dart_edges) >= 2:  # Need at least 2 edges to form a dart
                        debug_file.write(f"Processing dart #{dart_num} with {len(dart_edges)} edges\n")
                        
                        # Find the common point (where the edges meet)
                        # Typically, this is where the edges connect at the dart tip
                        found_tip = False
                        for i, edge1 in enumerate(dart_edges):
                            for j, edge2 in enumerate(dart_edges):
                                if i >= j:  # Only check unique pairs (and avoid comparing an edge with itself)
                                    continue
                                    
                                debug_file.write(f"Checking edges {i} and {j} for common points\n")
                                
                                # Check if any endpoints match
                                if np.allclose(edge1.start, edge2.start, atol=1e-6):
                                    debug_file.write(f"Found common point between edges at: {edge1.start}\n")
                                    dart_tips.append(edge1.start)
                                    found_tip = True
                                    break
                                elif np.allclose(edge1.start, edge2.end, atol=1e-6):
                                    debug_file.write(f"Found common point between edges at: {edge1.start}\n")
                                    dart_tips.append(edge1.start)
                                    found_tip = True
                                    break
                                elif np.allclose(edge1.end, edge2.start, atol=1e-6):
                                    debug_file.write(f"Found common point between edges at: {edge1.end}\n")
                                    dart_tips.append(edge1.end)
                                    found_tip = True
                                    break
                                elif np.allclose(edge1.end, edge2.end, atol=1e-6):
                                    debug_file.write(f"Found common point between edges at: {edge1.end}\n")
                                    dart_tips.append(edge1.end)
                                    found_tip = True
                                    break
                            
                            # If we found a dart tip, break out of the outer loop too
                            if found_tip:
                                break
                        
                        # If we couldn't find a common point, use the point with minimum Y as the dart tip
                        if not found_tip:
                            # Collect all points from this dart's edges
                            all_points = []
                            for edge in dart_edges:
                                all_points.append(edge.start)
                                all_points.append(edge.end)
                            
                            # Find the point with the minimum Y coordinate (lowest point)
                            min_y_point = min(all_points, key=lambda p: p[1])
                            dart_tips.append(min_y_point)
                            debug_file.write(f"No common point found, using lowest point as dart tip: {min_y_point}\n")
                
                debug_file.write(f"Found {len(dart_tips)} dart tips\n")
                for i, tip in enumerate(dart_tips):
                    debug_file.write(f"Dart tip #{i+1}: {tip}\n")
                debug_file.write("--- END FINDING DART TIPS ---\n")
            
            # Determine split point based on proportion
            split_point = None
            
            # Find the total length of all top edges (including interface and dart edges)
            total_top_length = 0
            edge_lengths = []
            
            with open(debug_log_path, "a") as debug_file:
                if all_top_edges:
                    for i, edge in enumerate(all_top_edges):
                        length = edge.length()
                        edge_lengths.append(length)
                        total_top_length += length
                        debug_file.write(f"DEBUG: Edge {i} length: {length:.2f}, labels: {edge.semantic_labels}\n")
                    
                    debug_file.write(f"DEBUG: Total top length: {total_top_length:.2f}\n")
                
                # Check if the proportion corresponds to a point on any top edge
                if all_top_edges and total_top_length > 0:
                    # Scale proportion to the total length of all top edges
                    target_length = proportion * total_top_length
                    debug_file.write(f"DEBUG: Target length along top edges: {target_length:.2f}\n")
                    
                    # Find which edge contains the point at this proportion
                    current_length = 0
                    for i, edge in enumerate(all_top_edges):
                        debug_file.write(f"DEBUG: Checking edge {i}, current_length={current_length:.2f}, target={target_length:.2f}, edge_length={edge_lengths[i]:.2f}\n")
                        if current_length <= target_length < current_length + edge_lengths[i]:
                            # This edge contains our proportion point
                            # Calculate local proportion on this edge
                            local_proportion = (target_length - current_length) / edge_lengths[i]
                            debug_file.write(f"DEBUG: Found target point on edge {i}, local_proportion={local_proportion:.2f}\n")
                            
                            # If this is a dart edge (not an interface edge), use the nearest dart tip
                            if any('dart_' in label for label in edge.semantic_labels):
                                debug_file.write(f"DEBUG: This is a dart edge with labels: {edge.semantic_labels}\n")
                                # Find which dart this edge belongs to
                                dart_num = None
                                for label in edge.semantic_labels:
                                    if label.startswith('dart_'):
                                        # Extract the dart number directly
                                        parts = label.split('_')
                                        if len(parts) >= 2 and parts[1].isdigit():
                                            dart_num = parts[1]
                                            debug_file.write(f"DEBUG: Found dart number: {dart_num}\n")
                                            break
                                
                                if dart_num:
                                    debug_file.write(f"DEBUG: Looking for dart tip for dart_{dart_num}\n")
                                    # Match the dart tip to the dart number
                                    # Since we're finding dart tips in order of dart numbers, we can use the index
                                    dart_index = None
                                    for i, num in enumerate(sorted(dart_numbers)):
                                        if num == dart_num:
                                            dart_index = i
                                            break
                                    
                                    if dart_index is not None and dart_index < len(dart_tips):
                                        # Use the dart tip that corresponds to this dart number
                                        split_point = dart_tips[dart_index]
                                        debug_file.write(f"DEBUG: Using dart tip for dart_{dart_num} (index {dart_index}): {split_point}\n")
                                    elif len(dart_tips) > 0:
                                        # Fall back to using any available dart tip
                                        split_point = dart_tips[0]
                                        debug_file.write(f"DEBUG: No specific dart tip for dart_{dart_num}, using first available: {split_point}\n")
                                    else:
                                        # Use the point on this edge if no dart tips available
                                        point = edge.point_at(local_proportion)
                                        debug_file.write(f"DEBUG: No dart tips available, using edge point: {point}\n")
                                        split_point = point
                                else:
                                    # If we couldn't determine the dart number, use the point on the edge
                                    point = edge.point_at(local_proportion)
                                    debug_file.write(f"DEBUG: Could not determine dart number, using edge point: {point}\n")
                                    split_point = point
                            else:
                                # This is a regular edge, use the point on this edge
                                point = edge.point_at(local_proportion)
                                debug_file.write(f"DEBUG: Regular edge point at {local_proportion:.2f}: {point}\n")
                                
                                # Check if this point is close to any dart tip
                                # Define "close" as within 10% of the total edge length
                                proximity_threshold = total_top_length * 0.1
                                debug_file.write(f"DEBUG: Proximity threshold: {proximity_threshold:.2f}\n")
                                
                                closest_dart_tip = None
                                min_distance = float('inf')
                                
                                for k, dart_tip in enumerate(dart_tips):
                                    # Calculate Euclidean distance to dart tip
                                    distance = np.sqrt((point[0] - dart_tip[0])**2 + (point[1] - dart_tip[1])**2)
                                    debug_file.write(f"DEBUG: Distance to dart tip #{k+1} {dart_tip}: {distance:.2f}\n")
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_dart_tip = dart_tip
                                
                                # If the point is close to a dart tip, use the dart tip instead
                                if closest_dart_tip is not None and min_distance < proximity_threshold:
                                    debug_file.write(f"DEBUG: Point is close to dart tip (distance: {min_distance:.2f}), using dart tip: {closest_dart_tip}\n")
                                    split_point = closest_dart_tip
                                else:
                                    debug_file.write(f"DEBUG: Using regular point: {point}\n")
                                    split_point = point
                            break
                        current_length += edge_lengths[i]
            
            # If no split point could be determined, fall back to default split
            with open(debug_log_path, "a") as debug_file:
                if split_point is None:
                    debug_file.write("DEBUG: No valid split point found, using default split.\n")
                    return super().split(proportion)
                
                # Get panel bounds to calculate equivalent proportion for bounding box
                x_min, y_min, x_max, y_max = self.get_bounds()
                panel_width = x_max - x_min
                
                # Calculate the equivalent proportion for the bounding box
                # This is the x-coordinate of the split point relative to the panel width
                equivalent_proportion = (split_point[0] - x_min) / panel_width
                debug_file.write(f"DEBUG: Panel bounds: x_min={x_min:.2f}, x_max={x_max:.2f}, width={panel_width:.2f}\n")
                debug_file.write(f"DEBUG: Split point: {split_point}, equivalent_proportion={equivalent_proportion:.2f}\n")
                
                # Call the superclass split method with the equivalent proportion
                return super().split(equivalent_proportion)
        else:
            # For front panels, just use the default split
            with open(os.path.expanduser("~/codestuff/GarmentCode/debug_split.log"), "a") as debug_file:
                debug_file.write(f"DEBUG: PantPanel.split called on front panel with proportion={proportion}\n")
            return super().split()

class PantsHalf(BaseBottoms):
    def __init__(self, tag, body, design, rise=None) -> None:
        super().__init__(body, design, tag, rise=rise)
        design = design['pants']
        self.rise = design['rise']['v'] if rise is None else rise
        waist, hips_depth, waist_back = self.eval_rise(self.rise)

        # NOTE: min value = full sum > leg curcumference
        # Max: pant leg falls flat from the back
        # Mostly from the back side
        # => This controls the foundation width of the pant
        min_ext = body['leg_circ'] - body['hips'] / 2  + 5  # 2 inch ease: from pattern making book 
        front_hip = (body['hips'] - body['hip_back_width']) / 2
        crotch_extention = min_ext * design['width']['v']  
        front_extention = front_hip / 4    # From pattern making book
        back_extention = crotch_extention - front_extention

        length, cuff_len = design['length']['v'], design['cuff']['cuff_len']['v']
        if design['cuff']['type']['v']: 
            if length - cuff_len < design['length']['range'][0]:   # Min length from paramss
                # Cannot be longer then a pant
                cuff_len = length - design['length']['range'][0]
            # Include the cuff into the overall length, 
            # unless the requested length is too short to fit the cuff 
            # (to avoid negative length)
            length -= cuff_len
        length *= body['_leg_length']
        cuff_len *= body['_leg_length']

        self.front = PantPanel(
            f'pant_f_{tag}', body, design,
            length=length,
            waist=(waist - waist_back) / 2,
            hips=(body['hips'] - body['hip_back_width']) / 2,
            hips_depth=hips_depth,
            dart_position = body['bust_points'] / 2,
            crotch_width=front_extention,
            match_top_int_to=(body['waist'] - body['waist_back_width']) / 2
            ).translate_by([0, body['_waist_level'] - 5, 25])
        self.back = PantPanel(
            f'pant_b_{tag}', body, design,
            length=length,
            waist=waist_back / 2,
            hips=body['hip_back_width'] / 2,
            hips_depth=hips_depth,
            hipline_ext=1.1,
            dart_position = body['bum_points'] / 2,
            crotch_width=back_extention,
            match_top_int_to=body['waist_back_width'] / 2,
            double_dart=True
            ).translate_by([0, body['_waist_level'] - 5, -20])

        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['outside'], self.back.interfaces['outside']),
            (self.front.interfaces['inside'], self.back.interfaces['inside'])
        )

        # add a cuff
        # TODOLOW This process is the same for sleeves -- make a function?
        if design['cuff']['type']['v']:
            
            pant_bottom = pyg.Interface.from_multiple(
                self.front.interfaces['bottom'],
                self.back.interfaces['bottom'])

            # Copy to avoid editing original design dict
            cdesign = deepcopy(design)
            cdesign['cuff']['b_width'] = {}
            cdesign['cuff']['b_width']['v'] = pant_bottom.edges.length() / design['cuff']['top_ruffle']['v']
            cdesign['cuff']['cuff_len']['v'] = cuff_len

            # Init
            cuff_class = getattr(bands, cdesign['cuff']['type']['v'])
            self.cuff = cuff_class(f'pant_{tag}', cdesign)

            # Position
            self.cuff.place_by_interface(
                self.cuff.interfaces['top'],
                pant_bottom,
                gap=5,
                alignment='left'
            )

            # Stitch
            self.stitching_rules.append((
                pant_bottom,
                self.cuff.interfaces['top'])
            )

        self.interfaces = {
            'crotch_f': self.front.interfaces['crotch'],
            'crotch_b': self.back.interfaces['crotch'],
            'top_f': self.front.interfaces['top'], 
            'top_b': self.back.interfaces['top'] 
        }

    def length(self):
        if self.design['pants']['cuff']['type']['v']:
            return self.front.length() + self.cuff.length()
        
        return self.front.length()

class Pants(BaseBottoms):
    def __init__(self, body, design, rise=None) -> None:
        super().__init__(body, design)

        self.right = PantsHalf('r', body, design, rise)
        self.left = PantsHalf('l', body, design, rise).mirror()

        self.stitching_rules = pyg.Stitches(
            (self.right.interfaces['crotch_f'], self.left.interfaces['crotch_f']),
            (self.right.interfaces['crotch_b'], self.left.interfaces['crotch_b']),
        )

        self.interfaces = {
            'top_f': pyg.Interface.from_multiple(
                self.right.interfaces['top_f'], self.left.interfaces['top_f']),
            'top_b': pyg.Interface.from_multiple(
                self.right.interfaces['top_b'], self.left.interfaces['top_b']),
            # Some are reversed for correct connection
            'top': pyg.Interface.from_multiple(   # around the body starting from front right
                self.right.interfaces['top_f'].flip_edges(),
                self.left.interfaces['top_f'].reverse(with_edge_dir_reverse=True),
                self.left.interfaces['top_b'].flip_edges(),
                self.right.interfaces['top_b'].reverse(with_edge_dir_reverse=True), # Flips the edges and restores the direction
            )
        }

    def get_rise(self):
        return self.right.get_rise()
    
    def length(self):
        return self.right.length()

