#!/usr/bin/env python3
import argparse, json, os, re, sys
from typing import List, Tuple, Dict

TEXT_PIECE_RE = re.compile(r"^\s*PIECE\s+(\d+)\s*$", re.IGNORECASE)
TEXT_VERTS_HEADER_RE = re.compile(r"^\s*VERTICES\s*\(X\s*,\s*Y\)\s*$", re.IGNORECASE)

def make_edges(nv: int):
    if nv < 2: return []
    edges = [{"endpoints": [i, i+1]} for i in range(nv-1)]
    edges.append({"endpoints": [nv-1, 0]})
    return edges

def parse_txt(lines: List[str]) -> Dict[int, Dict]:
    i, n = 0, len(lines)
    def skip_blanks(k): 
        while k < n and not lines[k].strip(): k += 1
        return k
    pieces = {}

    while True:
        i = skip_blanks(i)
        if i >= n: break
        m = TEXT_PIECE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        pid = int(m.group(1)); i += 1
        i = skip_blanks(i); assert lines[i].strip().upper() == "QUANTITY"; i += 1
        i = skip_blanks(i); qty = int(lines[i].strip()); i += 1
        i = skip_blanks(i); assert lines[i].strip().upper() == "NUMBER OF VERTICES"; i += 1
        i = skip_blanks(i); numv = int(lines[i].strip()); i += 1
        i = skip_blanks(i); assert TEXT_VERTS_HEADER_RE.match(lines[i]); i += 1

        verts = []
        for _ in range(numv):
            i = skip_blanks(i)
            x, y = map(float, lines[i].split())
            verts.append([x, y])
            i += 1

        pieces[pid] = {"quantity": qty, "vertices": verts}
    return pieces

def build_panels(pieces: Dict[int, Dict]) -> Dict[str, Dict]:
    panels = {}
    for pid, data in pieces.items():
        for c in range(1, data["quantity"] + 1):
            key = f"piece{pid}_{c}"
            panels[key] = {
                "translation": [0,0,0],
                "rotation": [0,0,0],
                "vertices": data["vertices"],
                "edges": make_edges(len(data["vertices"]))
            }
    return panels

def convert_file(path: str, outdir: str=None) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    pieces = parse_txt(lines)
    panels = build_panels(pieces)
    output = {
        "parameters": {},
        "parameter_order": [],
        "properties": {
            "curvature_coords": "relative",
            "normalize_panel_translation": False,
            "normalized_edge_loops": True,
            "units_in_meter": 1
        },
        "pattern": {
            "panels": panels
        }
    }
    base = os.path.splitext(os.path.basename(path))[0]
    outdir = outdir or os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{base}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Convert PIECE/QUANTITY TXT format to panel JSON.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--outdir", default=None)
    args = ap.parse_args()

    for file in args.inputs:
        try:
            out = convert_file(file, args.outdir)
            print(f"Wrote: {out}")
        except Exception as e:
            print(f"Failed {file}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
