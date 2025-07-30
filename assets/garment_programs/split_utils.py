from typing import Optional
import itertools
import numpy as np


from typing import Iterable

def collect_edges_by_label(panel, labels: Iterable[str] = (), prefixes: Iterable[str] = ("")):
    """
    Collect edges from a panel whose semantic labels match any of the given labels or start with any of the given prefixes.

    Args:
        panel: The panel object containing edges with semantic labels.
        labels: An iterable of exact label strings to match.
        prefixes: An iterable of string prefixes; any label starting with one of these will match.

    Returns:
        List of edges matching the criteria.
    """
    labels = set(labels)
    prefixes = tuple(prefixes)
    return [
        e for e in panel.edges
        if any(label in e.semantic_labels for label in labels)
        or any(any(lbl.startswith(prefix) for prefix in prefixes) for lbl in e.semantic_labels)
    ]


def dart_number(label: str) -> Optional[str]:
    """
    Extract the dart number from a label string of the form 'dart_X'.

    Args:
        label: The label string (e.g., 'dart_1', 'dart_2_left').

    Returns:
        The dart number as a string if present, otherwise None.
    """
    parts = label.split("_")
    return parts[1] if len(parts) >= 2 and parts[1].isdigit() else None


def find_dart_tips(top_edges, dart_nums):
    """
    Find the tip points for each dart number among the given edges.

    Args:
        top_edges: List of edges to search for dart tips.
        dart_nums: Iterable of dart numbers to look for.

    Returns:
        Dictionary mapping dart number to its tip point (coordinate tuple).
    """
    tips = {}
    for num in sorted(dart_nums):
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


def split_point(top_edges, dart_tips, proportion, dart_number_fn):
    """
    Find the split point along a set of edges, accounting for darts if present.
    This function calculates the split point based on the total length of the edges and the specified proportion.

    Args:
        top_edges: List of edges to split along.
        dart_tips: Dictionary of dart number to tip point.
        proportion: Proportion (0-1) along the total edge length to split.
        dart_number_fn: Function to extract dart number from a label.

    Returns:
        The coordinate tuple of the split point, or None if not found.
    """
    lengths = [e.length() for e in top_edges]
    total = sum(lengths)
    if not total:
        return None

    target = proportion * total
    cur = 0
    for e, l in zip(top_edges, lengths):
        if cur <= target < cur + l:
            local_prop = (target - cur) / l
            if any("dart_" in lbl for lbl in e.semantic_labels):
                dnum = next((
                    dart_number_fn(lbl)
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
