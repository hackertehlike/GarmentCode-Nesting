
from typing import Optional
from copy import deepcopy
import itertools
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
        
        for i, (off, dart) in enumerate(zip(offsets_mid, darts)):
            left_edge_len = top_edges[-1].length()
            original_edges_count = len(top_edges)
            
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
            
        # Add semantic labels to dart edges
        for start_idx, end_idx, dart_num in dart_indices:
            for j in range(start_idx, end_idx):
                # Skip interface edges (these are part of the dart perimeter)
                if top_edges[j] in int_edges:
                    continue
                else:
                    # These are the inner edges of the dart
                    top_edges[j].add_semantic_label(f'dart_{dart_num}')
                    # Add more specific labels if needed
                    if j == start_idx:
                        top_edges[j].add_semantic_label(f'dart_{dart_num}_left')
                    elif j == end_idx - 1:
                        top_edges[j].add_semantic_label(f'dart_{dart_num}_right')

        for e in int_edges:
            e.add_semantic_label('top')


        return top_edges, int_edges

    def split(self, proportion=0.5):
        """Split the panel while accounting for back darts."""

        if not self.name.startswith("pant_b_"):
            return super().split(proportion)

        def _collect_top_edges():
            return [
                e
                for e in self.edges
                if "top" in e.semantic_labels
                or any("dart" in lbl for lbl in e.semantic_labels)
            ]

        def _dart_number(label: str) -> Optional[str]:
            parts = label.split("_")
            return parts[1] if len(parts) >= 2 and parts[1].isdigit() else None

        def _find_dart_tips(edges):
            tips = {}
            for num in sorted(edges):
                dart_edges = [e for e in top_edges if any(lbl.startswith(f"dart_{num}") for lbl in e.semantic_labels)]
                pt = None
                for e1, e2 in itertools.combinations(dart_edges, 2):
                    for a in (e1.start, e1.end):
                        for b in (e2.start, e2.end):
                            if np.allclose(a, b, atol=1e-6):
                                pt = a
                                break
                        if pt is not None:
                            break
                    if pt is not None:
                        break
                if pt is None:
                    points = [p for de in dart_edges for p in (de.start, de.end)]
                    if points:
                        pt = min(points, key=lambda p: p[1])
                if pt is not None:
                    tips[num] = pt
            return tips

        def _split_point(prop):
            lengths = [e.length() for e in top_edges]
            total = sum(lengths)
            if not total:
                return None

            target = prop * total
            cur = 0
            for e, l in zip(top_edges, lengths):
                if cur <= target < cur + l:
                    local_prop = (target - cur) / l
                    if any("dart_" in lbl for lbl in e.semantic_labels):
                        dnum = next((
                            _dart_number(lbl)
                            for lbl in e.semantic_labels
                            if lbl.startswith("dart_")
                        ), None)
                        if dnum and dnum in dart_tips:
                            return dart_tips[dnum]
                        return next(iter(dart_tips.values()), e.point_at(local_prop))
                    point = e.point_at(local_prop)
                    prox = total * 0.02
                    if dart_tips:
                        nearest, dist = min(
                            (
                                (dt, np.linalg.norm(np.array(point) - np.array(dt)))
                                for dt in dart_tips.values()
                            ),
                            key=lambda t: t[1],
                        )
                        if dist < prox:
                            return nearest
                    return point
                cur += l
            return None

        top_edges = _collect_top_edges()
        dart_nums = {
            _dart_number(lbl)
            for e in top_edges
            for lbl in e.semantic_labels
            if lbl.startswith("dart_") and _dart_number(lbl) is not None
        }
        dart_tips = _find_dart_tips(dart_nums)
        split_pt = _split_point(proportion)

        if split_pt is None:
            return super().split(proportion)

        x_min, _, x_max, _ = self.get_bounds()
        prop_equiv = (split_pt[0] - x_min) / (x_max - x_min)
        return super().split(prop_equiv)

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

