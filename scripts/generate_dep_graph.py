#!/usr/bin/env python3
"""
Generate a proof dependency graph from the SOS Lean 4 formalization.

Parses all .lean files to extract declarations (theorems, definitions, lemmas)
and their dependencies, producing a JSON graph suitable for Sigma.js visualization
in the DEM platform's graph explorer.

Output: dep_graph.json with nodes and edges.
"""

import json
import re
import sys
from pathlib import Path

SOS_DIR = Path.home() / "sos-formalisation" / "SosFormalization"


def extract_declarations(filepath: Path) -> list[dict]:
    """Extract all named declarations from a Lean 4 file."""
    declarations = []
    content = filepath.read_text()
    module = filepath.stem

    # Match: theorem|def|lemma|noncomputable def|axiom|structure|instance NAME
    pattern = re.compile(
        r'^(noncomputable\s+)?(theorem|def|lemma|axiom|structure|instance|abbrev)\s+'
        r'(\S+)',
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        noncomputable = match.group(1) is not None
        kind = match.group(2)
        name = match.group(3)
        line = content[:match.start()].count('\n') + 1

        # Skip if it's inside a comment
        prefix = content[:match.start()]
        if prefix.count('/-') > prefix.count('-/'):
            continue

        declarations.append({
            "name": name,
            "kind": kind,
            "module": module,
            "line": line,
            "noncomputable": noncomputable,
            "has_export": f"@[export" in content[max(0, match.start()-200):match.start()],
        })

    return declarations


def extract_dependencies(filepath: Path, all_names: set[str]) -> list[tuple[str, str]]:
    """Extract dependency edges: which declarations reference which others."""
    edges = []
    content = filepath.read_text()

    # For each declaration, find references to other known declarations
    decl_pattern = re.compile(
        r'^(noncomputable\s+)?(theorem|def|lemma|axiom|structure|instance|abbrev)\s+'
        r'(\S+)',
        re.MULTILINE
    )

    matches = list(decl_pattern.finditer(content))

    for i, match in enumerate(matches):
        source_name = match.group(3)
        # Get the body of this declaration (up to next declaration or EOF)
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end]

        # Find references to other declarations
        for name in all_names:
            if name == source_name:
                continue
            # Look for the name as a word boundary
            if re.search(r'\b' + re.escape(name) + r'\b', body):
                edges.append((source_name, name))

    return edges


def main():
    if not SOS_DIR.exists():
        print(f"SOS directory not found: {SOS_DIR}")
        sys.exit(1)

    lean_files = sorted(SOS_DIR.glob("*.lean"))
    print(f"Parsing {len(lean_files)} Lean files from {SOS_DIR}")

    # Phase 1: Extract all declarations
    all_declarations = []
    for filepath in lean_files:
        decls = extract_declarations(filepath)
        all_declarations.extend(decls)
        print(f"  {filepath.stem}: {len(decls)} declarations")

    all_names = {d["name"] for d in all_declarations}
    print(f"\nTotal declarations: {len(all_declarations)}")

    # Phase 2: Extract dependency edges
    all_edges = []
    for filepath in lean_files:
        edges = extract_dependencies(filepath, all_names)
        all_edges.extend(edges)
        print(f"  {filepath.stem}: {len(edges)} edges")

    # Deduplicate edges
    all_edges = list(set(all_edges))
    print(f"\nTotal edges: {len(all_edges)}")

    # Phase 3: Build graph JSON
    # Categorize by module for coloring
    module_colors = {
        "Basic": "#4CAF50",         # Green — foundations
        "PowerLaw": "#2196F3",      # Blue — rates
        "StochasticSOS": "#9C27B0", # Purple — stochastic
        "StochasticRates": "#E91E63",# Pink — stoch rates
        "AReaL": "#FF9800",         # Orange — applications
        "ProofForge": "#F44336",    # Red — ProofForge
    }

    kind_shapes = {
        "theorem": "circle",
        "def": "square",
        "noncomputable def": "square",
        "lemma": "circle",
        "axiom": "diamond",
        "structure": "triangle",
        "instance": "pentagon",
        "abbrev": "hexagon",
    }

    nodes = []
    for d in all_declarations:
        nodes.append({
            "id": d["name"],
            "label": d["name"],
            "module": d["module"],
            "kind": d["kind"],
            "line": d["line"],
            "color": module_colors.get(d["module"], "#607D8B"),
            "shape": kind_shapes.get(d["kind"], "circle"),
            "noncomputable": d["noncomputable"],
            "has_export": d["has_export"],
            "size": 8 if d["kind"] in ("theorem", "axiom") else 5,
        })

    edges = []
    for source, target in all_edges:
        edges.append({
            "source": source,
            "target": target,
        })

    graph = {
        "description": "SOS Formalization Proof Dependency Graph",
        "generated": "2026-03-18",
        "stats": {
            "total_declarations": len(all_declarations),
            "total_edges": len(all_edges),
            "modules": {m: len([d for d in all_declarations if d["module"] == m])
                       for m in set(d["module"] for d in all_declarations)},
            "by_kind": {},
        },
        "nodes": nodes,
        "edges": edges,
    }

    # Count by kind
    for d in all_declarations:
        kind = d["kind"]
        graph["stats"]["by_kind"][kind] = graph["stats"]["by_kind"].get(kind, 0) + 1

    # Write output
    output_path = Path.home() / "projects" / "proofforge" / "data" / "sos_dep_graph.json"
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"\nGraph written to {output_path}")
    print(f"\nStats:")
    print(f"  Declarations: {len(all_declarations)}")
    print(f"  Edges: {len(all_edges)}")
    print(f"  Modules: {json.dumps(graph['stats']['modules'], indent=4)}")
    print(f"  By kind: {json.dumps(graph['stats']['by_kind'], indent=4)}")

    # Also output a summary for quick reference
    print(f"\n=== Key declarations ===")
    for d in all_declarations:
        if d["kind"] in ("axiom", "structure") or d["has_export"]:
            tag = "[EXPORT]" if d["has_export"] else f"[{d['kind'].upper()}]"
            print(f"  {tag} {d['name']} ({d['module']}:{d['line']})")


if __name__ == "__main__":
    main()
