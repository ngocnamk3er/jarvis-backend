"""Finding ruled tables in an image, from their lines.

Reading a table as lines of text mixes up its columns: one OCR "line" runs across the whole
row and glues unrelated cells together ("1 Ứng dụng phần mềm ... Gói 1 1.320.000 1.320.000"),
which a model then pairs up wrongly. So tables are found first, from the drawn grid, and the
words are put into cells (see ocr.py); only what is outside the tables is read as lines.

The method is the usual one for ruled tables: keep only the long horizontal and vertical
strokes (morphological opening), treat the white regions they enclose as cells, and group cells
that touch into a table. It needs the lines to be drawn; a table without rules (just aligned
columns of text) is not found and is read as ordinary lines.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

Box = tuple[int, int, int, int]

MIN_CELLS = 4            # a table has at least this many cells ...
MIN_ROWS = MIN_COLS = 2  # ... in at least this many rows and columns
FULL_WIDTH = 0.85        # a region this wide is a section of the page, not a table cell
TOL = 6                  # px: starts within this of each other are the same row or column


@dataclass
class Grid:
    box: Box
    cells: list[Box]


def _touch(a: Box, b: Box, gap: int) -> bool:
    """Whether two cells belong to the same table: they share an edge (allowing for the thickness of
    the rule between them), and for cells stacked one above the other they are about as wide, so a
    wide strip under a table (a totals line, say) is not taken for another row of its columns."""
    h_overlap = min(a[3], b[3]) - max(a[1], b[1])
    v_overlap = min(a[2], b[2]) - max(a[0], b[0])
    side = 0 <= b[0] - a[2] <= gap or 0 <= a[0] - b[2] <= gap
    stack = 0 <= b[1] - a[3] <= gap or 0 <= a[1] - b[3] <= gap
    widest = max(a[2] - a[0], b[2] - b[0])
    return (side and h_overlap >= 0.5 * min(a[3] - a[1], b[3] - b[1])) or (stack and v_overlap >= 0.6 * widest)


def starts(values: list[int], tol: int = TOL) -> list[int]:
    """The distinct positions in `values`, merging ones within `tol` of each other."""
    out: list[int] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def find_grids(img: Image.Image) -> list[Grid]:
    """The ruled tables in the picture, top to bottom, each with the boxes of its cells."""
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    if h < 40 or w < 40:
        return []
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40))))
    rules = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(255 - rules, connectivity=4)

    cells: list[Box] = []
    for i in range(1, count):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:       # the page background around everything
            continue
        if cw >= FULL_WIDTH * w or ch >= FULL_WIDTH * h or cw < 12 or ch < 8:
            continue
        if area < 0.8 * cw * ch:                                   # not a rectangle: not a cell
            continue
        cells.append((x, y, x + cw, y + ch))

    parent = list(range(len(cells)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gap = max(8, round(0.015 * max(w, h)))
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if _touch(cells[i], cells[j], gap):
                parent[find(i)] = find(j)
    groups: dict[int, list[Box]] = {}
    for i, c in enumerate(cells):
        groups.setdefault(find(i), []).append(c)

    grids = []
    for group in groups.values():
        box = (min(c[0] for c in group), min(c[1] for c in group), max(c[2] for c in group), max(c[3] for c in group))
        if len(group) < MIN_CELLS or len(starts([c[1] for c in group])) < MIN_ROWS or len(starts([c[0] for c in group])) < MIN_COLS:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > 0.9 * w * h:
            continue
        grids.append(Grid(box, sorted(group, key=lambda c: (c[1], c[0]))))
    grids.sort(key=lambda g: (g.box[1], g.box[0]))
    return grids


def anchors(cells: list[Box]) -> tuple[list[int], list[int]]:
    """(column x positions, row y positions) of a table's grid.

    Taken from the row with the most cells and the column with the most cells, which are the
    ones not broken up by merged cells; every other cell is placed against these."""
    by_y: dict[int, list[Box]] = {}
    for y in starts([c[1] for c in cells]):
        by_y[y] = [c for c in cells if abs(c[1] - y) <= TOL]
    by_x: dict[int, list[Box]] = {}
    for x in starts([c[0] for c in cells]):
        by_x[x] = [c for c in cells if abs(c[0] - x) <= TOL]
    widest_row = max(by_y.values(), key=len)
    tallest_col = max(by_x.values(), key=len)
    return starts([c[0] for c in widest_row]), starts([c[1] for c in tallest_col])
