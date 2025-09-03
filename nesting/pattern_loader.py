# from __future__ import annotations
# from pathlib import Path
# import copy
# import yaml
# from typing import Dict, Tuple, Iterable
# import nesting.config as config
# from nesting.path_extractor import PatternPathExtractor
# from assets.bodies.body_params import BodyParameters
# from .layout import Piece

# __all__ = [
#     "find_pattern_triples",
#     "load_pattern_bundle",
# ]

# def find_pattern_triples(root: Path) -> Iterable[tuple[Path, Path, Path]]:
#     """Yield (spec_path, design_params_path, body_measurements_path) triples under *root*.
#     A triple is emitted only if all three files exist.
#     """
#     for spec in sorted(root.rglob("*_specification.json")):
#         stem = spec.stem.replace("_specification", "")
#         design_yaml = spec.parent / f"{stem}_design_params.yaml"
#         body_yaml = spec.parent / f"{stem}_body_measurements.yaml"
#         if design_yaml.exists() and body_yaml.exists():
#             yield spec, design_yaml, body_yaml


# def load_pattern_bundle(spec_path: Path) -> tuple[Dict[str, Piece], dict | None, BodyParameters | None, str]:
#     """Load a single pattern specification plus its design & body params (if present).

#     This encapsulates the logic used by run_experiments and (optionally) the GUI so
#     tests and tools share a single implementation.

#     Returns:
#         pieces: dict[id -> Piece] (deep‑copied, seam allowance added, translation reset)
#         design_params: nested dict under key 'design' if YAML present, else None
#         body_params: BodyParameters instance if measurements YAML present, else None
#         pattern_name: stem without the trailing _specification

#     Raises:
#         FileNotFoundError if *spec_path* does not exist.
#         RuntimeError for malformed design params yaml.
#     """
#     if not spec_path.exists():
#         raise FileNotFoundError(f"Specification file not found: {spec_path}")

#     pattern_stem = spec_path.stem
#     pattern_name = pattern_stem.replace("_specification", "") if pattern_stem.endswith("_specification") else pattern_stem

#     # --- geometry -----------------------------------------------------
#     extractor = PatternPathExtractor(spec_path)
#     panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
#     if not panel_pieces:
#         raise RuntimeError(f"No pieces found in specification: {spec_path}")
#     pieces: Dict[str, Piece] = {p.id: copy.deepcopy(p) for p in panel_pieces.values()}
#     for piece in pieces.values():
#         piece.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
#         piece.translation = (0, 0)

#     # --- design params ------------------------------------------------
#     design_yaml = spec_path.parent / f"{pattern_name}_design_params.yaml"
#     design_params = None
#     if design_yaml.exists():
#         with open(design_yaml, "r", encoding="utf-8") as f:
#             raw = yaml.safe_load(f)
#         if raw and "design" in raw:
#             design_params = raw["design"]
#         else:
#             raise RuntimeError(f"Design params yaml missing 'design' key: {design_yaml}")

#     # --- body params --------------------------------------------------
#     body_yaml = spec_path.parent / f"{pattern_name}_body_measurements.yaml"
#     body_params = BodyParameters(body_yaml) if body_yaml.exists() else None

#     return pieces, design_params, body_params, pattern_name
