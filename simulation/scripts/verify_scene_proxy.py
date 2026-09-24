#!/usr/bin/env python3
"""Project the approximate cylinder onto saved 3DGS inspection RGB images.

Diagnostic only: this script never changes the saved camera frames and never
sends an overlay to the policy. Wireframe geometry is the proposed collision
proxy, not an independently measured object surface.
"""
from pathlib import Path
import math

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/realistic"
POSITIONS = np.array([[13.55, -3.15, 1.2], [15.15, -4.45, 1.2],
                      [16.75, -3.15, 1.2], [15.15, -1.45, 1.2]])
TARGET = np.array([15.15, -2.95, 1.0])
CENTER = np.array([15.15, -2.95])
RADIUS = 0.45
BOTTOM, TOP = 0.0, 1.55
WIDTH, HEIGHT, FOVY = 320, 240, 85.0


def project(points, position):
    forward = TARGET - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points) - position
    depth = delta @ forward
    focal = HEIGHT / (2 * math.tan(math.radians(FOVY / 2)))
    uv = np.column_stack((WIDTH/2 + focal * (delta @ right) / depth,
                          HEIGHT/2 - focal * (delta @ up) / depth))
    return uv, depth > 0.03


def main():
    angles = np.linspace(0, 2 * math.pi, 129)
    xy = CENTER + RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
    rings = [np.column_stack((xy, np.full(len(xy), z))) for z in (BOTTOM, TOP)]
    segments = rings + [np.array([[*xy[i], BOTTOM], [*xy[i], TOP]]) for i in range(0, 128, 16)]
    panels = []
    for index, position in enumerate(POSITIONS):
        original = cv2.imread(str(OUT / f"obstacle-view-{index}.png"))
        if original is None or original.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Missing or unexpected inspection frame {index}")
        panel = cv2.resize(original, (640, 480), interpolation=cv2.INTER_CUBIC)
        for segment in segments:
            pixels, visible = project(segment, position)
            points = np.rint(pixels * 2).astype(np.int32)
            for i in range(len(points) - 1):
                if visible[i] and visible[i+1]:
                    cv2.line(panel, tuple(points[i]), tuple(points[i+1]), (255, 35, 255), 2, cv2.LINE_AA)
        cv2.rectangle(panel, (0, 0), (640, 34), (25, 25, 25), -1)
        cv2.putText(panel, f"View {index}: camera {position.tolist()}", (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(panel)
    board = np.concatenate((np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)), axis=0)
    header = np.full((85, board.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(header, "Approximate cylinder proxy overlay - diagnostic only", (22, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 90, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "r=0.45 m; z=0..1.55 m; centre=(15.15,-2.95). Not measured depth or an image supplied to Flyvis.",
                (22, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (230, 230, 230), 1, cv2.LINE_AA)
    path = OUT / "proxy-overlay.png"
    cv2.imwrite(str(path), np.concatenate((header, board), axis=0))
    print(path)


if __name__ == "__main__":
    main()
