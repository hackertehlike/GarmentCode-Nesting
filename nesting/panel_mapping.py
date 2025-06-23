"""Mapping from design parameter keys to affected panel id patterns."""
from __future__ import annotations

from fnmatch import fnmatch
from typing import Iterable, Mapping, Sequence, Set, Dict, List, Optional, Tuple

# Mapping from parameter path (e.g. "waistband.width") to a set of
# fnmatch-style patterns identifying panels affected by that parameter.
# The patterns are intentionally coarse; they can be refined as needed.
PARAM_TO_PATTERNS: Mapping[str, Set[str]] = {
    # Waistband
    "waistband.width": {"wb_front", "wb_back"},
    "waistband.waist": {"wb_front", "wb_back"},

    # Shirt body (non-fitted shirt only)
    "shirt.length": {"*ftorso", "*btorso"},
    "shirt.width":  {"*ftorso", "*btorso"},
    "shirt.flare":  {"*ftorso", "*btorso"},
    "shirt.strapless": {"*ftorso", "*btorso"},

    # Collars
    "collar.width": {"*ftorso", "*btorso", "*lapel*", "*hood*"},
    "collar.f_collar": {"*ftorso"},
    "collar.b_collar": {"*btorso"},
    "collar.fc_depth": {"*ftorso"},
    "collar.bc_depth": {"*btorso"},
    "collar.fc_angle": {"*ftorso"},
    "collar.bc_angle": {"*btorso"},
    "collar.f_bezier_x": {"*ftorso"},
    "collar.f_bezier_y": {"*ftorso"},
    "collar.b_bezier_x": {"*btorso"},
    "collar.b_bezier_y": {"*btorso"},
    "collar.f_flip_curve": {"*ftorso"},
    "collar.b_flip_curve": {"*btorso"},
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
    "left.shirt.width": {"left_ftorso", "left_btorso"},
    "left.shirt.flare": {"left_ftorso", "left_btorso"},
    "left.shirt.strapless": {"left_ftorso", "left_btorso"},
    "left.collar.f_collar": {"left_ftorso"},
    "left.collar.b_collar": {"left_btorso"},
    "left.sleeve.sleeveless": {"left_sleeve_*"},

    # Skirts (includes various styles)
    "skirt.length": {"skirt_*"},
    "skirt.rise": {"skirt_*"},
    "skirt.ruffle": {"skirt_*"},
    "skirt.bottom_cut": {"skirt_*"},
    "skirt.flare": {"skirt_*"},

    "flare-skirt.length": {"skirt_*"},
    "flare-skirt.rise": {"skirt_*"},
    "flare-skirt.suns": {"skirt_*"},
    "flare-skirt.skirt-many-panels.n_panels": {"ins_skirt_*"},
    "flare-skirt.skirt-many-panels.panel_curve": {"ins_skirt_*"},
    "flare-skirt.asymm.front_length": {"skirt_*"},
    "flare-skirt.cut.add": {"skirt_*"},
    "flare-skirt.cut.depth": {"skirt_*"},
    "flare-skirt.cut.width": {"skirt_*"},
    "flare-skirt.cut.place": {"skirt_*"},

    "godet-skirt.base": {"skirt_*"},
    "godet-skirt.insert_w": {"ins_*"},
    "godet-skirt.insert_depth": {"ins_*"},
    "godet-skirt.num_inserts": {"ins_*"},
    "godet-skirt.cuts_distance": {"ins_*"},

    "pencil-skirt.length": {"skirt_*"},
    "pencil-skirt.rise": {"skirt_*"},
    "pencil-skirt.flare": {"skirt_*"},
    "pencil-skirt.low_angle": {"skirt_*"},
    "pencil-skirt.front_slit": {"skirt_*"},
    "pencil-skirt.back_slit": {"skirt_*"},
    "pencil-skirt.left_slit": {"skirt_*"},
    "pencil-skirt.right_slit": {"skirt_*"},
    "pencil-skirt.style_side_cut": {"skirt_*"},

    "levels-skirt.base": {"skirt_*"},
    "levels-skirt.level": {"skirt_*"},
    "levels-skirt.num_levels": {"skirt_*"},
    "levels-skirt.level_ruffle": {"skirt_*"},
    "levels-skirt.length": {"skirt_*"},
    "levels-skirt.rise": {"skirt_*"},
    "levels-skirt.base_length_frac": {"skirt_*"},

    # Pants
    "pants.length": {"pant_*"},
    "pants.width": {"pant_*"},
    "pants.flare": {"pant_*"},
    "pants.rise": {"pant_*"},
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
}


def affected_panels(params: Sequence[str]) -> Set[str]:
    """Return the set of panel patterns affected by the given param paths."""
    patterns: Set[str] = set()
    for p in params:
        pats = PARAM_TO_PATTERNS.get(p)
        if pats:
            patterns.update(pats)
    return patterns


def select_genes(genes: Iterable[str], patterns: Iterable[str]) -> Set[str]:
    """Return the subset of *genes* whose id matches any of the fnmatch patterns."""
    pats = list(patterns)
    return {g for g in genes if any(fnmatch(g, pat) for pat in pats)}


def filter_parameters(design_params: Dict, panel_ids: Optional[Set[str]] = None) -> Dict:
    """
    Filter design parameters based on:
    1. The panel IDs present in the design (if provided)
    2. Hierarchical parameter relationships
    
    Args:
        design_params: The full design parameter dictionary
        panel_ids: Optional set of panel IDs to filter parameters by
        
    Returns:
        Filtered design parameter dictionary
    """
    # Make a deep copy to avoid modifying the original
    import copy
    dp = copy.deepcopy(design_params)
    
    # If panel IDs are provided, filter by relevant parameters for those panels
    # Collect all parameters that would affect any of our loaded panels
    top_level_params = set()
    for param_path, panel_patterns in PARAM_TO_PATTERNS.items():
        # Check if any of our panels match the patterns for this parameter
        matching_panels = select_genes(panel_ids, panel_patterns)
        if matching_panels:
            # This parameter affects at least one of our loaded panels
            top_level_params.add(param_path.split('.')[0])  # Add the top-level component
    
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
    x
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