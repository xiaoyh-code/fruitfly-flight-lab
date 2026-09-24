#!/usr/bin/env python3
"""Export the seven MIT-licensed Crazyflie visual meshes for a Three.js viewer.

Positions remain in the Crazyflie body frame, in metres with +Z up. The MJCF
body's initial world position is deliberately not baked into the visual asset.
No collision meshes, simplification, coordinate changes, or downloads are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/bitcraze_crazyflie_2/cf2.xml"
OUTPUT = ROOT / "dashboard/assets/crazyflie.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounds(positions: list[float]) -> dict:
    lower = [min(positions[axis::3]) for axis in range(3)]
    upper = [max(positions[axis::3]) for axis in range(3)]
    return {"min": lower, "max": upper,
            "size": [upper[axis] - lower[axis] for axis in range(3)]}


def read_obj(path: Path) -> dict:
    vertices, source_normals, faces = [], [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] in {"v", "vn"}:
            values = tuple(float(value) for value in fields[1:])
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Invalid {fields[0]} in {path.name}")
            (vertices if fields[0] == "v" else source_normals).append(values)
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError(f"Expected triangulated source: {path.name}")
            corners = []
            for field in fields[1:]:
                indices = field.split("/")
                if len(indices) != 3 or not indices[2]:
                    raise ValueError(f"Source normal missing: {path.name}")
                v, n = int(indices[0]), int(indices[2])
                v = v - 1 if v > 0 else len(vertices) + v
                n = n - 1 if n > 0 else len(source_normals) + n
                if not 0 <= v < len(vertices) or not 0 <= n < len(source_normals):
                    raise ValueError(f"Source index out of bounds: {path.name}")
                corners.append((v, n))
            faces.append(corners)
    if not vertices or not source_normals or not faces:
        raise ValueError(f"Source geometry is empty: {path.name}")
    positions, normals, indices, corner_map = [], [], [], {}
    for face in faces:
        for corner in face:
            if corner not in corner_map:
                corner_map[corner] = len(corner_map)
                positions.extend(vertices[corner[0]])
                normals.extend(source_normals[corner[1]])
            indices.append(corner_map[corner])
    # Verify every exported triangle corner, not only overall extents/counts.
    for i, corner in enumerate(corner for face in faces for corner in face):
        start = indices[i] * 3
        if tuple(positions[start:start + 3]) != vertices[corner[0]]:
            raise ValueError(f"Position changed during export: {path.name}")
        if tuple(normals[start:start + 3]) != source_normals[corner[1]]:
            raise ValueError(f"Normal changed during export: {path.name}")
    normal_lengths = [math.sqrt(sum(value * value for value in n)) for n in source_normals]
    if min(normal_lengths) < .99 or max(normal_lengths) > 1.01:
        raise ValueError(f"Source normals are not approximately unit length: {path.name}")
    return {"positions": positions, "normals": normals, "indices": indices,
            "vertex_count": len(positions) // 3, "triangle_count": len(indices) // 3,
            "bounds": bounds(positions), "source_mesh": path.name,
            "source_sha256": sha256(path)}


def export(source: Path, output: Path) -> dict:
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    mesh_dir = source.parent / compiler.get("meshdir", "")
    materials = {item.get("name"): [float(v) for v in item.get("rgba").split()]
                 for item in root.findall("asset/material")}
    meshes = {item.get("name", Path(item.get("file")).stem): item
              for item in root.findall("asset/mesh")}
    parts = []
    for geom in root.findall("worldbody/body/geom"):
        if geom.get("class") != "visual":
            continue
        if any(geom.get(key) for key in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("Visual geom transforms need explicit export support")
        name, material = geom.get("mesh"), geom.get("material")
        mesh = meshes[name]
        if any(mesh.get(key) for key in ("scale", "refpos", "refquat")):
            raise ValueError("Mesh transforms need explicit export support")
        part = read_obj(mesh_dir / mesh.get("file"))
        part.update(name=name, material=material, rgba=materials[material])
        parts.append(part)
    if len(parts) != 7 or {part["name"] for part in parts} != {f"cf2_{i}" for i in range(7)}:
        raise ValueError("Expected the seven Crazyflie visual parts")
    all_positions = [v for part in parts for v in part["positions"]]
    asset = {
        "version": 1, "name": "Bitcraze Crazyflie 2", "units": "meters", "up": [0, 0, 1],
        "coordinate_frame": "Crazyflie body frame; MJCF initial world translation excluded",
        "source": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2",
        "source_description": "Menagerie MJCF visual meshes, derived from whoenig/crazyflie_ros",
        "source_xml_sha256": sha256(source),
        "license": "MIT", "copyright": "Copyright (c) 2024 whoenig",
        "license_text": (source.parent / "LICENSE").read_text(encoding='utf-8'),
        "bounds": bounds(all_positions),
        "vertex_count": sum(part["vertex_count"] for part in parts),
        "triangle_count": sum(part["triangle_count"] for part in parts),
        "parts": parts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asset, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding='utf-8')
    # JSON round-trip must preserve all source coordinates, normals and indices.
    if json.loads(output.read_text(encoding='utf-8')) != asset:
        raise ValueError("Export did not survive JSON round-trip")
    return asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    asset = export(args.source, args.output)
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "parts": len(asset["parts"]), "vertices": asset["vertex_count"],
                      "triangles": asset["triangle_count"], "bounds": asset["bounds"],
                      "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
