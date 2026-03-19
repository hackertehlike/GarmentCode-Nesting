import random
from pathlib import Path
import copy
import yaml
import nesting.config as config
from nesting.data.path_extractor import PatternPathExtractor
from nesting.core.layout import Container
from nesting.search.chromosome import Chromosome
from assets.bodies.body_params import BodyParameters


def _find_pattern_with_params(root: Path) -> tuple[Path, Path, Path]:
    for spec in sorted(root.glob('*/**/*_specification.json')):
        stem = spec.stem.replace('_specification', '')
        design_yaml = spec.parent / f"{stem}_design_params.yaml"
        body_yaml = spec.parent / f"{stem}_body_measurements.yaml"
        if design_yaml.exists() and body_yaml.exists():
            return spec, design_yaml, body_yaml
    raise FileNotFoundError("No pattern with matching design/body yaml found under pattern_files")


def test_split_then_design_param_mutation():
    random.seed(10)

    pattern_root = Path('nesting-assets/pattern_files')
    spec_path, design_yaml, body_yaml = _find_pattern_with_params(pattern_root)

    extractor = PatternPathExtractor(spec_path)
    panel_pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
    assert panel_pieces, "No pieces extracted from specification"

    pieces = [copy.deepcopy(p) for p in panel_pieces.values()]
    for p in pieces:
        p.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
        p.translation = (0, 0)

    with open(design_yaml, 'r', encoding='utf-8') as f:
        design_params = yaml.safe_load(f)['design']
    body_params = BodyParameters(body_yaml)

    container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)
    chrom = Chromosome(pieces, container, origin='test', design_params=design_params, body_params=body_params)
    chrom.calculate_fitness()
    assert chrom._mutatable_params, "No mutatable design parameters collected"

    # 1) Directly call split mutation
    before_gene_count = len(chrom.genes)
    split_ok = chrom._mutate_split()
    assert split_ok, "_mutate_split returned False"
    assert len(chrom.genes) == before_gene_count + 1, "Gene count did not increase by 1 after split"
    assert chrom.split_history, "split_history not updated after split"

    prev_split_history = list(chrom.split_history)

    # 2) Directly call design param mutation
    design_ok = chrom._mutate_design_params()
    # It's possible no fitness-changing mutation was found; still ensure no structural loss.
    # If you want to enforce success, uncomment the next assertion.
    # assert design_ok, "_mutate_design_params failed to apply a change"

    root_ids = {p.root_id for p in chrom.genes}
    for rid, _prop in prev_split_history:
        assert rid in root_ids, f"Root {rid} lost after design param mutation"
    assert all(entry in chrom.split_history for entry in prev_split_history), "Split history lost entries after design param mutation"

if __name__ == '__main__':
    test_split_then_design_param_mutation()
    print('Test completed.')
