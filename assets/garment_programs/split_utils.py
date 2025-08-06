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


def cut_number(label: str) -> Optional[str]:
    """
    Extract the cut number from a label string of the form 'cut_X'.

    Args:
        label: The label string (e.g., 'cut_1', 'cut_2_left').

    Returns:
        The cut number as a string if present, otherwise None.
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


def find_cut_tips(bottom_edges, cut_nums):
    """
    Find the tip points for each cut number among the given edges.
    Cuts are similar to darts but typically found on bottom edges.

    Args:
        bottom_edges: List of edges to search for cut tips.
        cut_nums: Iterable of cut numbers to look for.

    Returns:
        Dictionary mapping cut number to its tip point (coordinate tuple).
    """
    tips = {}
    for num in sorted(cut_nums):
        cut_edges = [e for e in bottom_edges if any(lbl.startswith(f"cut_{num}") for lbl in e.semantic_labels)]
        pt = None
        for e1, e2 in itertools.combinations(cut_edges, 2):
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
            points = [p for ce in cut_edges for p in (ce.start, ce.end)]
            if points:
                pt = max(points, key=lambda p: p[1])  # Cut tips are typically at the highest point (closest to waist)
        if pt is not None:
            tips[num] = pt
    return tips


def split_point(top_edges, proportion, dart_tips = None, bottom_edges=None, cut_tips=None):
    """
    Find the split point along a set of edges, accounting for darts if present.
    This function calculates the split point based on the total length of the edges and the specified proportion.

    Args:
        top_edges: List of edges to split along.
        proportion: Proportion (0-1) along the total edge length to split.
        dart_tips: Dictionary of dart number to tip point.
        bottom_edges: List of bottom edges to check for cuts (optional).
        cut_tips: Dictionary of cut number to tip point (optional).

    Returns:
        The coordinate tuple of the split point, or None if not found.
    """
    lengths = [e.length() for e in top_edges]
    total = sum(lengths)
    if not total:
        return None, None

    target = proportion * total
    cur = 0
    for e, l in zip(top_edges, lengths):
        if cur <= target < cur + l:
            local_prop = (target - cur) / l
            if any("dart_" in lbl for lbl in e.semantic_labels):
                dnum = next((
                    dart_number(lbl)
                    for lbl in e.semantic_labels
                    if lbl.startswith("dart_")
                ), None)
                if dnum and dart_tips and dnum in dart_tips:
                    return dart_tips[dnum], e
                return next(iter(dart_tips.values()), e.point_at(local_prop)) if dart_tips else e.point_at(local_prop), e
            point = e.point_at(local_prop)
            
            # Check if we should snap to a cut tip instead
            if cut_tips and bottom_edges:
                prox = total * 0.05  # 5% proximity threshold
                for cut_pt in cut_tips.values():
                    if abs(point[0] - cut_pt[0]) < prox:  # Check x-coordinate proximity
                        return cut_pt, e
                        
            return point, e
        cur += l
    return None, None
