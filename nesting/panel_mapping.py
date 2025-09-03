"""Mapping from design parameter keys to affected panel id patterns."""
from __future__ import annotations

from fnmatch import fnmatch
from typing import Iterable, Mapping, Sequence, Set, Dict, List, Optional, Tuple

from pygarment.garmentcode.panel import Panel
from nesting.layout import Piece

import nesting.config as config

# Mapping from parameter path (e.g. "waistband.width") to a set of
# fnmatch-style patterns identifying panels affected by that parameter.

PARAM_TO_PATTERNS: Mapping[str, Set[str]] = {
    # Waistband
    "waistband.width": {"*wb_front*", "*wb_back*"},
    "waistband.waist": {"*wb_front*", "*wb_back*"},

    # Shirt body (non-fitted shirt only)
    "shirt.length": {"*ftorso*", "*btorso*"},
    "shirt.width":  {"*ftorso*", "*btorso*"},
    "shirt.flare":  {"*ftorso*", "*btorso*"},
    "shirt.strapless": {"*ftorso*", "*btorso*"},

    # Collars
    "collar.width": {"*ftorso*", "*btorso*", "*lapel*", "*hood*"},
    "collar.f_collar": {"*ftorso*"},
    "collar.b_collar": {"*btorso*"},
    "collar.fc_depth": {"*ftorso*"},
    "collar.bc_depth": {"*btorso*"},
    "collar.fc_angle": {"*ftorso*"},
    "collar.bc_angle": {"*btorso*"},
    "collar.f_bezier_x": {"*ftorso*"},
    "collar.f_bezier_y": {"*ftorso*"},
    "collar.b_bezier_x": {"*btorso*"},
    "collar.b_bezier_y": {"*btorso*"},
    "collar.f_flip_curve": {"*ftorso*"},
    "collar.b_flip_curve": {"*btorso*"},
    "collar.component.style": {"*lapel*", "*hood*"},
    "collar.component.depth": {"*lapel*", "*hood*"},
    "collar.component.lapel_standing": {"*lapel*"},
    "collar.component.hood_depth": {"*hood*"},
    "collar.component.hood_length": {"*hood*"},

    # Sleeves
    "sleeve.sleeveless": {"*sleeve*"},
    "sleeve.length": {"*sleeve*"},
    "sleeve.armhole_shape": {"*sleeve*", "*torso"},
    "sleeve.connecting_width": {"*sleeve*", "*torso"},
    "sleeve.end_width": {"*sleeve*"},
    "sleeve.sleeve_angle": {"*sleeve*"},
    "sleeve.opening_dir_mix": {"*sleeve*"},
    "sleeve.standing_shoulder": {"*sleeve*"},
    "sleeve.standing_shoulder_len": {"*sleeve*"},
    "sleeve.connect_ruffle": {"*sleeve*"},
    "sleeve.smoothing_coeff": {"*sleeve*"},
    "sleeve.cuff.type": {"*cuff*"},
    "sleeve.cuff.top_ruffle": {"*cuff*"},
    "sleeve.cuff.cuff_len": {"*cuff*"},
    "sleeve.cuff.skirt_fraction": {"*cuff*"},
    "sleeve.cuff.skirt_flare": {"*cuff*"},
    "sleeve.cuff.skirt_ruffle": {"*cuff*"},

    # Left-side overrides
    "left.shirt.width": {"*left_ftorso*", "*left_btorso*"},
    "left.shirt.flare": {"*left_ftorso*", "*left_btorso*"},
    "left.shirt.strapless": {"*left_ftorso*", "*left_btorso*"},

    "left.collar.width": {"*left_ftorso*", "*left_btorso*", "*lapel*", "*hood*"},
    "left.collar.f_collar": {"*left_ftorso*"},
    "left.collar.b_collar": {"*left_btorso*"},
    "left.collar.fc_angle": {"*left_ftorso*"},
    "left.collar.bc_angle": {"*left_btorso*"},
    "left.collar.f_bezier_x": {"*left_ftorso*"},
    "left.collar.f_bezier_y": {"*left_ftorso*"},
    "left.collar.b_bezier_x": {"*left_btorso*"},
    "left.collar.b_bezier_y": {"*left_btorso*"},
    "left.collar.f_flip_curve": {"*left_ftorso*"},
    "left.collar.b_flip_curve": {"*left_btorso*"},

    "left.sleeve.sleeveless": {"*left_sleeve_*"},
    "left.sleeve.armhole_shape": {"*left_sleeve_*", "*left_ftorso*", "*left_btorso*"},
    "left.sleeve.length": {"*left_sleeve_*"},
    "left.sleeve.connecting_width": {"*left_sleeve_*", "*left_ftorso*", "*left_btorso*"},
    "left.sleeve.end_width": {"*left_sleeve_*"},
    "left.sleeve.sleeve_angle": {"*left_sleeve_*"},
    "left.sleeve.opening_dir_mix": {"*left_sleeve_*"},
    "left.sleeve.standing_shoulder": {"*left_sleeve_*"},
    "left.sleeve.standing_shoulder_len": {"*left_sleeve_*"},
    "left.sleeve.connect_ruffle": {"*left_sleeve_*"},
    "left.sleeve.smoothing_coeff": {"*left_sleeve_*"},
    "left.sleeve.cuff.type": {"*left_cuff*"},
    "left.sleeve.cuff.top_ruffle": {"*left_cuff*"},
    "left.sleeve.cuff.cuff_len": {"*left_cuff*"},
    "left.sleeve.cuff.skirt_fraction": {"*left_cuff*"},
    "left.sleeve.cuff.skirt_flare": {"*left_cuff*"},
    "left.sleeve.cuff.skirt_ruffle": {"*left_cuff*"},

    # Skirts (includes various styles)
    "skirt.length": {"*skirt_*"},
    "skirt.rise": {"*skirt_*"},
    "skirt.ruffle": {"*skirt_*"},
    "skirt.bottom_cut": {"*skirt_*"},
    "skirt.flare": {"*skirt_*"},

    "flare-skirt.length": {"*skirt_*"},
    "flare-skirt.rise": {"*skirt_*"},
    "flare-skirt.suns": {"*skirt_*"},
    "flare-skirt.skirt-many-panels.n_panels": {"*ins_skirt_*"},
    "flare-skirt.skirt-many-panels.panel_curve": {"*ins_skirt_*"},
    "flare-skirt.asymm.front_length": {"*skirt_*"},
    "flare-skirt.cut.add": {"*skirt_*"},
    "flare-skirt.cut.depth": {"*skirt_*"},
    "flare-skirt.cut.width": {"*skirt_*"},
    "flare-skirt.cut.place": {"skirt_*"},

    "godet-skirt.base": {"skirt_*"},
    "godet-skirt.insert_w": {"ins_*"},
    "godet-skirt.insert_depth": {"*ins_*"},
    "godet-skirt.num_inserts": {"*ins_*"},
    "godet-skirt.cuts_distance": {"*ins_*"},

    "pencil-skirt.length": {"*skirt_*"},
    "pencil-skirt.rise": {"*skirt_*"},
    "pencil-skirt.flare": {"*skirt_*"},
    "pencil-skirt.low_angle": {"*skirt_*"},
    "pencil-skirt.front_slit": {"*skirt_*"},
    "pencil-skirt.back_slit": {"*skirt_*"},
    "pencil-skirt.left_slit": {"*skirt_*"},
    "pencil-skirt.right_slit": {"*skirt_*"},
    "pencil-skirt.style_side_cut": {"*skirt_*"},

    "levels-skirt.base": {"*skirt_*"},
    "levels-skirt.level": {"*skirt_*"},
    "levels-skirt.num_levels": {"*skirt_*"},
    "levels-skirt.level_ruffle": {"*skirt_*"},
    "levels-skirt.length": {"*skirt_*"},
    "levels-skirt.rise": {"*skirt_*"},
    "levels-skirt.base_length_frac": {"skirt_*"},

    # Pants
    "pants.length": {"*pant_*"},
    "pants.width": {"*pant_*"},
    "pants.flare": {"*pant_*"},
    "pants.rise": {"*pant_*"},
    "pants.cuff.type": {"*cuff*"},
    "pants.cuff.top_ruffle": {"*cuff*"},
    "pants.cuff.cuff_len": {"*cuff*"},
    "pants.cuff.skirt_fraction": {"*cuff*"},
    "pants.cuff.skirt_flare": {"*cuff*"},
    "pants.cuff.skirt_ruffle": {"*cuff*"},
}

# Define hierarchical parameter relationships
# Map of parent parameters to their children that should be hidden when the parent has a specific value
PARAMETER_HIERARCHY: Dict[str, List[Tuple[str, Optional[bool]]]] = {
    # When sleeve.sleeveless is True, hide all other sleeve parameters
    "sleeve.sleeveless": [
        ("sleeve.length", True),
        ("sleeve.armhole_shape", True),
        ("sleeve.connecting_width", True),
        ("sleeve.end_width", True),
        ("sleeve.sleeve_angle", True),
        ("sleeve.opening_dir_mix", True),
        ("sleeve.standing_shoulder", True),
        ("sleeve.standing_shoulder_len", True),
        ("sleeve.connect_ruffle", True),
        ("sleeve.smoothing_coeff", True),
        ("sleeve.cuff.type", True),
        ("sleeve.cuff.top_ruffle", True),
        ("sleeve.cuff.cuff_len", True),
        ("sleeve.cuff.skirt_fraction", True),
        ("sleeve.cuff.skirt_flare", True),
        ("sleeve.cuff.skirt_ruffle", True),
    ],
    
    # When left.sleeve.sleeveless is True, hide all other left sleeve parameters
    "left.sleeve.sleeveless": [
        ("left.sleeve.length", True),
        ("left.sleeve.armhole_shape", True),
        ("left.sleeve.connecting_width", True),
        ("left.sleeve.end_width", True),
        ("left.sleeve.sleeve_angle", True),
        ("left.sleeve.opening_dir_mix", True),
        ("left.sleeve.standing_shoulder", True),
        ("left.sleeve.standing_shoulder_len", True),
        ("left.sleeve.connect_ruffle", True),
        ("left.sleeve.smoothing_coeff", True),
        ("left.sleeve.cuff.type", True),
        ("left.sleeve.cuff.top_ruffle", True),
        ("left.sleeve.cuff.cuff_len", True),
        ("left.sleeve.cuff.skirt_fraction", True),
        ("left.sleeve.cuff.skirt_flare", True),
        ("left.sleeve.cuff.skirt_ruffle", True),
    ],
    
    # If collar.component.style is None, hide other collar.component parameters
    "collar.component.style": [
        ("collar.component.depth", None),
        ("collar.component.lapel_standing", None),
        ("collar.component.hood_depth", None),
        ("collar.component.hood_length", None),
    ],
    
    # If sleeve.cuff.type is None, hide other sleeve.cuff parameters
    "sleeve.cuff.type": [
        ("sleeve.cuff.top_ruffle", None),
        ("sleeve.cuff.cuff_len", None),
        ("sleeve.cuff.skirt_fraction", None),
        ("sleeve.cuff.skirt_flare", None),
        ("sleeve.cuff.skirt_ruffle", None),
    ],
    
    # If pants.cuff.type is None, hide other pants.cuff parameters
    "pants.cuff.type": [
        ("pants.cuff.top_ruffle", None),
        ("pants.cuff.cuff_len", None),
        ("pants.cuff.skirt_fraction", None),
        ("pants.cuff.skirt_flare", None),
        ("pants.cuff.skirt_ruffle", None),
    ],

    # If left.enable_asym is False, hide other left parameters
    "left.enable_asym": [
    ("left.shirt.strapless", False),
    ("left.shirt.width", False),
    ("left.shirt.flare", False),

    ("left.collar.f_collar", False),
    ("left.collar.b_collar", False),
    ("left.collar.width", False),
    ("left.collar.fc_angle", False),
    ("left.collar.bc_angle", False),
    ("left.collar.f_bezier_x", False),
    ("left.collar.f_bezier_y", False),
    ("left.collar.b_bezier_x", False),
    ("left.collar.b_bezier_y", False),
    ("left.collar.f_flip_curve", False),
    ("left.collar.b_flip_curve", False),

    ("left.sleeve.sleeveless", False),
    ("left.sleeve.armhole_shape", False),
    ("left.sleeve.length", False),
    ("left.sleeve.connecting_width", False),
    ("left.sleeve.end_width", False),
    ("left.sleeve.sleeve_angle", False),
    ("left.sleeve.opening_dir_mix", False),
    ("left.sleeve.standing_shoulder", False),
    ("left.sleeve.standing_shoulder_len", False),
    ("left.sleeve.connect_ruffle", False),
    ("left.sleeve.smoothing_coeff", False),

    ("left.sleeve.cuff.type", False),
    ("left.sleeve.cuff.top_ruffle", False),
    ("left.sleeve.cuff.cuff_len", False),
    ("left.sleeve.cuff.skirt_fraction", False),
    ("left.sleeve.cuff.skirt_flare", False),
    ("left.sleeve.cuff.skirt_ruffle", False),
    ],

    "flare-skirt.cut.add": [
        ("flare-skirt.cut.depth", False),
        ("flare-skirt.cut.width", False),
        ("flare-skirt.cut.place", False),
    ],
}


def affected_panels(params: Sequence[str], design_params: Optional[Dict] = None) -> Set[str]:
    """
    Return the set of panel patterns affected by the given param paths.
    
    Args:
        params: Parameter paths to check
        design_params: Optional design parameters to filter by garment type
    """
    patterns: Set[str] = set()
    
    # Check if we have meta parameters to filter by garment type
    active_bottom_type = None
    allowed_bottom_prefixes = []
    
    # Check for active collar types
    f_collar_type = None
    b_collar_type = None
    
    if design_params:
        # Get active bottom type
        if 'meta' in design_params and 'bottom' in design_params['meta']:
            meta_bottom = design_params['meta']['bottom']
            if isinstance(meta_bottom, dict) and 'v' in meta_bottom:
                active_bottom_type = meta_bottom['v']
                
                # Mapping from meta bottom types to corresponding parameter sections
                bottom_type_mapping = {
                    'SkirtCircle': ['skirt'],
                    'AsymmSkirtCircle': ['flare-skirt'],
                    'GodetSkirt': ['godet-skirt'],
                    'PencilSkirt': ['pencil-skirt'],
                    'Skirt2': ['skirt'],
                    'SkirtManyPanels': ['skirt-many-panels'],
                    'SkirtLevels': ['levels-skirt'],
                    'Pants': ['pants'],
                }
                
                if active_bottom_type in bottom_type_mapping:
                    allowed_bottom_prefixes = bottom_type_mapping[active_bottom_type]
                    # print(f" Active bottom type: {active_bottom_type}, allowed prefixes: {allowed_bottom_prefixes}")
        
        # Get collar types
        if 'collar' in design_params:
            if 'f_collar' in design_params['collar'] and 'v' in design_params['collar']['f_collar']:
                f_collar_type = design_params['collar']['f_collar']['v']
                
            if 'b_collar' in design_params['collar'] and 'v' in design_params['collar']['b_collar']:
                b_collar_type = design_params['collar']['b_collar']['v']
    
    for p in params:
        # Check if this is a skirt-related parameter
        param_prefix = p.split('.')[0]
        parts = p.split('.')
        
        # Filter skirt parameters (there are many skirt types)
        is_skirt_param = param_prefix.endswith('-skirt') or param_prefix == 'skirt'
        if is_skirt_param and active_bottom_type and allowed_bottom_prefixes:
            if not any(param_prefix == prefix for prefix in allowed_bottom_prefixes):
                # print(f" Parameter '{p}' ignored - wrong skirt type (expected {allowed_bottom_prefixes})")
                continue
        
        # Filter bezier collar parameters
        # is_bezier_param = False
        if len(parts) > 1 and parts[0] == 'collar':
            # Front collar bezier parameters
            if parts[1] in ['f_bezier_x', 'f_bezier_y', 'f_flip_curve'] and f_collar_type != 'Bezier2NeckHalf':
                # print(f" Parameter '{p}' ignored - not applicable for front collar type {f_collar_type}")
                continue
                
            # Back collar bezier parameters
            if parts[1] in ['b_bezier_x', 'b_bezier_y', 'b_flip_curve'] and b_collar_type != 'Bezier2NeckHalf':
                # print(f" Parameter '{p}' ignored - not applicable for back collar type {b_collar_type}")
                continue
        
        pats = PARAM_TO_PATTERNS.get(p)
        if pats:
            patterns.update(pats)
            # print(f"Parameter '{p}' affects panel patterns: {pats}")
        # else:
        #     print(f"Parameter '{p}' has no registered panel patterns")
            
    return patterns


def select_genes(genes: Iterable[str], patterns: Iterable[str]) -> Set[str]:
    """Return the subset of *genes* whose id matches any of the given *patterns*."""
    pats = list(patterns)
    # print(f"Selecting genes from {list(genes)} with patterns {pats}")
    selected = {g for g in genes if any(fnmatch(g, pat) for pat in pats)}
    # print(f"Selected genes: {selected}")
    return selected


def filter_parameters(design_params: Dict, panel_ids: Optional[Set[str]] = None) -> Dict:
    """
    Filter design parameters based on:
    1. The panel IDs present in the design (if provided)
    2. Meta parameters (to determine active skirt/bottom type)
    3. Hierarchical parameter relationships
    
    Args:
        design_params: The full design parameter dictionary
        panel_ids: Optional set of panel IDs to filter parameters by
        
    Returns:
        Filtered design parameter dictionary
    """
    # Make a deep copy to avoid modifying the original
    import copy
    dp = copy.deepcopy(design_params)
    
    # Get active bottom garment type from meta parameters
    active_bottom_type = None
    if 'meta' in dp and 'bottom' in dp['meta'] and 'v' in dp['meta']['bottom']:
        active_bottom_type = dp['meta']['bottom']['v']
        # print(f"[DEBUG] Active bottom type from meta: {active_bottom_type}")
    
    # Get active collar types
    f_collar_type = None
    b_collar_type = None
    
    if 'collar' in dp:
        if 'f_collar' in dp['collar'] and 'v' in dp['collar']['f_collar']:
            f_collar_type = dp['collar']['f_collar']['v']
            # print(f"[DEBUG] Front collar type: {f_collar_type}")
            
        if 'b_collar' in dp['collar'] and 'v' in dp['collar']['b_collar']:
            b_collar_type = dp['collar']['b_collar']['v']
            # print(f"[DEBUG] Back collar type: {b_collar_type}")
    
    # Mapping from meta bottom types to corresponding parameter sections
    bottom_type_mapping = {
        'SkirtCircle': ['skirt'],
        'AsymmSkirtCircle': ['flare-skirt'],
        'GodetSkirt': ['godet-skirt'],
        'PencilSkirt': ['pencil-skirt'],
        'Skirt2': ['skirt'],
        'SkirtManyPanels': ['skirt-many-panels'],
        'SkirtLevels': ['levels-skirt'],
        'Pants': ['pants'],
    }
    
    # Get allowed skirt parameter prefixes based on active bottom type
    allowed_prefixes = []
    if active_bottom_type and active_bottom_type in bottom_type_mapping:
        allowed_prefixes = bottom_type_mapping[active_bottom_type]
        # print(f"Allowed parameter prefixes from meta: {allowed_prefixes}")

    # If panel IDs are provided, filter by relevant parameters for those panels
    # Collect all parameters that would affect any of our loaded panels
    top_level_params = set()
    skirt_params_to_remove = set()
    
    for param_path, panel_patterns in PARAM_TO_PATTERNS.items():
        # First part of the parameter path (e.g., 'skirt', 'flare-skirt', etc.)
        param_prefix = param_path.split('.')[0]
        parts = param_path.split('.')
        
        # Check if this is a skirt-related parameter
        is_skirt_param = param_prefix.endswith('-skirt') or param_prefix == 'skirt'
        
        # If it's a skirt parameter and we have active_bottom_type, check if it's allowed
        if is_skirt_param and active_bottom_type:
            if not any(param_prefix == prefix for prefix in allowed_prefixes):
                skirt_params_to_remove.add(param_path)
                # print(f" Removing irrelevant skirt parameter: {param_path}")
                continue
        
        # Filter bezier collar parameters
        if len(parts) > 1 and parts[0] == 'collar':
            # Front collar bezier parameters
            if parts[1] in ['f_bezier_x', 'f_bezier_y', 'f_flip_curve'] and f_collar_type and f_collar_type != 'Bezier2NeckHalf':
                # print(f" Removing irrelevant front collar parameter: {param_path} for collar type {f_collar_type}")
                continue
                
            # Back collar bezier parameters
            if parts[1] in ['b_bezier_x', 'b_bezier_y', 'b_flip_curve'] and b_collar_type and b_collar_type != 'Bezier2NeckHalf':
                # print(f" Removing irrelevant back collar parameter: {param_path} for collar type {b_collar_type}")
                continue
        
        # Check if any of our panels match the patterns for this parameter
        if panel_ids:
            matching_panels = select_genes(panel_ids, panel_patterns)
            if matching_panels:
                # This parameter affects at least one of our loaded panels
                top_level_params.add(param_prefix)  # Add the top-level component
    
    # Always include meta
    top_level_params.add('meta')

    # Filter the design parameters to only include relevant sections
    dp = {k: v for k, v in dp.items() if k in top_level_params}
    
    # 2. Apply hierarchical parameter filtering
    _apply_parameter_hierarchy(dp)
    
    return dp


def _apply_parameter_hierarchy(dp: Dict) -> None:
    """
    Apply hierarchical parameter filtering to the design parameters.
    Modifies the dp dictionary in place.
    """
    # Helper function to get a nested parameter value
    def _get_nested(params: Dict, path: str) -> Optional[Dict]:
        parts = path.split('.')
        current = params
        for part in parts:
            if part not in current:
                return None
            current = current[part]
        return current
    
    # Helper function to set a nested parameter value
    def _set_nested(params: Dict, path: str, value: Optional[Dict]) -> None:
        parts = path.split('.')
        current = params
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                return
            current = current[part]
        if parts[-1] in current:
            if value is None:
                current.pop(parts[-1])
            else:
                current[parts[-1]] = value
    
    # Process each parent parameter
    for parent_path, children in PARAMETER_HIERARCHY.items():
        # Get the parent parameter
        parent = _get_nested(dp, parent_path)
        
        # If parent exists and has a value that triggers hiding children
        if parent is not None:
            parent_value = parent.get('v')
            
            # Process each child parameter
            for child_path, trigger_value in children:
                # If parent value matches the trigger value (or trigger is None), hide the child
                if trigger_value is None or parent_value == trigger_value:
                    _set_nested(dp, child_path, None)

# --- Panel Splitting Dispatcher ---

# # 1. Mapping from panel name prefixes to panel type identifiers
# PANEL_TYPE_MAPPING = {
#     'skirt_front': 'circle_skirt',
#     'skirt_back': 'circle_skirt',
#     'circle_skirt': 'circle_skirt',  # Direct match
#     'circle_panel': 'circle_skirt',  # For any generic circle panel
#     'halfcircle': 'circle_skirt',    # For half circle skirts
#     'test_panel': 'circle_skirt',    # For tests
#     # Add other panel types here in the future
#     # e.g., 'tshirt_front': 'tshirt',
# }

# def get_panel_type(panel_name):
#     """Determines the panel type from its name using the mapping."""
#     #print(f"[DEBUG] get_panel_type called for panel '{panel_name}'")
#     for prefix, panel_type in PANEL_TYPE_MAPPING.items():
#         if panel_name.startswith(prefix):
#             #print(f"[DEBUG] Matched prefix '{prefix}' -> type '{panel_type}'")
#             return panel_type
#     #print(f"[DEBUG] No type mapping found for panel '{panel_name}'")
#     return None


# def dispatch_split(piece, design_params=None, body_params=None):
#     """Calls the correct split() method on a piece based on its type or panel.
    
#     Args:
#         piece: The piece object to be split.
#         design_params: Optional design parameters to regenerate the pattern.
#         body_params: Optional body parameters to regenerate the pattern.
        
#     Returns:
#         A tuple of (left_piece, right_piece) if split is successful, or None.
#     """
#     import tempfile
#     from pathlib import Path
#     import copy
#     import nesting.config as config
    
#     #print(f"[DEBUG] dispatch_split called for piece '{piece.id}'")
    
#     # If we have design params and body params, regenerate the MetaGarment pattern
#     if design_params and body_params:
#         #print(f"[DEBUG] Regenerating MetaGarment pattern for piece '{piece.id}'")

#         from assets.garment_programs.meta_garment import MetaGarment
#         from nesting.path_extractor import PatternPathExtractor

#         # Determine panel type from piece ID
#         panel_type = get_panel_type(piece.id)
#         if not panel_type:
#             #print(f"[DEBUG] Cannot determine panel type for piece '{piece.id}'")
#             return None

#         #print(f"[DEBUG] Panel type identified as: '{panel_type}'")

#         # For now, only handle circle_skirt panels
#         #if panel_type != 'circle_skirt':
#             #print(f"[DEBUG] Only circle_skirt panels are currently supported for regeneration")
#         #    return None

#         # Regenerate the MetaGarment
#         mg = MetaGarment("split_regenerate", body_params, design_params)

#         # ------------------------------------------------------------------
#         # Find the panel on the MetaGarment BEFORE assembly and split it
#         # ------------------------------------------------------------------
#         panel = None

#         is_front = "front" in piece.id.lower()
#         is_back = "back" in piece.id.lower()

#         skirt_component = None
#         for sub in mg.subs:
#             if hasattr(sub, 'front') and hasattr(sub, 'back'):
#                 skirt_component = sub
#                 break

#         if skirt_component is None:
#             #print(f"[DEBUG] Could not find skirt component in the garment")
#             return None

#         if is_front and hasattr(skirt_component, 'front'):
#             panel = skirt_component.front
#         elif is_back and hasattr(skirt_component, 'back'):
#             panel = skirt_component.back

#         if panel is None:
#             print(f"[DEBUG] Could not find panel '{piece.id}' in regenerated garment")
#             return None

#         #print(f"[DEBUG] Found panel '{panel.name}' in regenerated garment")

#         if not hasattr(panel, 'split') or not callable(panel.split):
#             print(f"[DEBUG] Panel '{panel.name}' does not have a split method")
#             return None

#         print(f"[DEBUG] Using specialized panel.split() method")
#         left_panel, right_panel = panel.split()
#         print(f"[DEBUG] Panel split successful: {left_panel.name}, {right_panel.name}")

#         # Assemble pattern AFTER the split
#         pattern = mg.assembly()
        
#         # Export to temporary directory and create pieces
#         with tempfile.TemporaryDirectory() as td:
#             spec_file = Path(td) / f"{pattern.name}_specification.json"
#             pattern.serialize(Path(td), to_subfolder=False, with_3d=False,
#                                 with_text=False, view_ids=False)
#             extractor = PatternPathExtractor(spec_file)
            
#             # Get pieces for the split panels
#             all_pieces = extractor.get_all_panel_pieces(
#                 samples_per_edge=config.SAMPLES_PER_EDGE
#             )
            
#             if left_panel.name not in all_pieces or right_panel.name not in all_pieces:
#                 #print(f"[DEBUG] Failed to extract split pieces from pattern")
#                 return None
                
#             left_piece = all_pieces[left_panel.name]
#             right_piece = all_pieces[right_panel.name]
            
#             # Copy necessary attributes from the original piece
#             for new_piece in [left_piece, right_piece]:
#                 new_piece.parent_id = piece.id
#                 new_piece.root_id = getattr(piece, "root_id", piece.id)
#                 new_piece.rotation = piece.rotation
#                 # Apply seam allowance
#                 new_piece.add_seam_allowance()
#                 new_piece.update_bbox()
            
#             #print(f"[DEBUG] Created split pieces: '{left_piece.id}' and '{right_piece.id}'")
#             return left_piece, right_piece
    
#     # Fall back to the piece's own split method
#     if hasattr(piece, 'split') and callable(piece.split):
#         #print(f"[DEBUG] Falling back to generic piece.split() method")
#         return piece.split()
    
#     # If we got here, no splitting method was available
#     #print(f"[DEBUG] No split method available for piece '{piece.id}'")
#     return None