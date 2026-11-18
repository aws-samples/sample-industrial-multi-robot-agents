#!/usr/bin/env python3
"""
Generate per-namespace URDF files with correct world-base_link origins.

Reads the cell's robots.yaml for base positions and
copies/patches URDF from cells/<cell_id>/urdf/template/ or a source URDF.

Usage:
    python3 scripts/generate_cell_urdfs.py --cell welding_cell_2_small
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inject_base_origin(urdf_str: str, x: float, y: float, z: float, yaw: float) -> str:
    """
    Inject the world-base_link joint origin with correct xyz/rpy.
    Replaces the origin in the root joint connecting world to base_link.
    """
    xyz_str = f"{x:.6f} {y:.6f} {z:.6f}"
    rpy_str = f"0 0 {yaw:.6f}"

    # Pattern: origin in a joint that connects world to base_link
    pattern = r'(<joint[^>]*name="[^"]*(?:world[^"]*base|root)[^"]*"[^>]*>.*?<origin\s+xyz=")[^"]*("\s+rpy=")[^"]*(")'
    replacement = rf'\g<1>{xyz_str}\g<2>{rpy_str}\g<3>'
    result = re.sub(pattern, replacement, urdf_str, flags=re.DOTALL | re.IGNORECASE)

    if result == urdf_str:
        # Fallback: try to find any world-frame joint and patch its origin
        pattern2 = r'(<origin\s+xyz=")[^"]*("\s+rpy=")[^"]*(")'
        result = urdf_str.replace(
            next(re.finditer(pattern2, urdf_str), type('', (), {'group': lambda _: ''})()).group(0),
            f'<origin xyz="{xyz_str}" rpy="{rpy_str}"',
            1,
        ) if re.search(pattern2, urdf_str) else urdf_str

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--cell-base", default=None)
    args   = parser.parse_args()

    cell_id   = args.cell
    cell_base = args.cell_base or os.environ.get("CELL_BASE_PATH", "./cells")
    base_path = Path(cell_base) / cell_id

    robots_path = base_path / "config" / "robots.yaml"
    with open(robots_path) as f:
        robots_cfg = yaml.safe_load(f)

    for ns, ns_block in robots_cfg.get("robots", {}).items():
        if not isinstance(ns_block, dict):
            continue
        for _, rcfg in ns_block.items():
            if not isinstance(rcfg, dict):
                continue

            bx, by, bz = rcfg["base_position"]
            yaw        = float(rcfg.get("base_orientation_yaw", 0.0))

            # Source URDF: look in urdf/<ns>/ or urdf/template/
            src_dir = base_path / "urdf" / ns
            if not src_dir.exists():
                src_dir = base_path / "urdf" / "template"

            urdfs = list(src_dir.glob("*.urdf")) if src_dir.exists() else []
            if not urdfs:
                print(f"[generate_urdfs] No URDF found for {ns}, skipping")
                continue

            src_urdf = urdfs[0]
            dst_dir  = base_path / "urdf" / ns
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_urdf = dst_dir / src_urdf.name

            urdf_str = src_urdf.read_text()
            patched  = inject_base_origin(urdf_str, bx, by, bz, yaw)

            dst_urdf.write_text(patched)
            print(f"[generate_urdfs] {ns}: {dst_urdf} (base=[{bx},{by},{bz}] yaw={yaw:.3f})")

            # Copy meshes if present
            mesh_src = src_dir / "meshes"
            mesh_dst = dst_dir / "meshes"
            if mesh_src.exists() and not mesh_dst.exists():
                shutil.copytree(mesh_src, mesh_dst)
                print(f"[generate_urdfs] {ns}: copied meshes")
            break


if __name__ == "__main__":
    main()
