"""Draw the QA-DARTS supernet as an IEEE-width publication figure.

The script writes a vector PDF for LaTeX and a 300 dpi PNG preview.  It uses
ReportLab and Pillow rather than a neural-network-specific drawing package so
that every label, colour, and physical dimension is deterministic.

Run from any directory with:

    python scripts/draw_supernet.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


# IEEEtran's double-column text width is approximately 7.16 in on US Letter.
WIDTH_IN = 7.16
HEIGHT_IN = 3.35
DPI = 300
PT_PER_IN = 72.0
WIDTH_PT = WIDTH_IN * PT_PER_IN
HEIGHT_PT = HEIGHT_IN * PT_PER_IN

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
PDF_PATH = FIGURE_DIR / "qa_darts_supernet.pdf"
PNG_PATH = FIGURE_DIR / "qa_darts_supernet.png"

INK = "#25313C"
MUTED = "#5B6570"
CONNECTOR = "#78838E"
FIXED = "#D9EAD3"
SEARCH = "#FFE7B3"
ATTENTION = "#C9DAF8"
DOWNSAMPLE = "#F4CCCC"
INPUT = "#E9EEF5"


@dataclass(frozen=True)
class Stage:
    name: str
    shape: str
    resolution: int
    channels: int
    colour: str


STAGES = (
    Stage("Input", "32x32x3", 32, 3, INPUT),
    Stage("Stem", "32x32x32", 32, 32, FIXED),
    Stage("L1", "32x32x32", 32, 32, SEARCH),
    Stage("L2", "32x32x64", 32, 64, SEARCH),
    Stage("D1", "16x16x64", 16, 64, DOWNSAMPLE),
    Stage("L3", "16x16x64", 16, 64, SEARCH),
    Stage("L4", "16x16x128", 16, 128, SEARCH),
    Stage("D2", "8x8x128", 8, 128, DOWNSAMPLE),
    Stage("L5", "8x8x128", 8, 128, ATTENTION),
    Stage("L6", "8x8x256", 8, 256, ATTENTION),
    Stage("L7", "8x8x200", 8, 200, ATTENTION),
    Stage("D3", "4x4x200", 4, 200, DOWNSAMPLE),
    Stage("GAP", "1x1x200", 1, 200, FIXED),
    Stage("Classifier", "200 -> C", 1, 1, FIXED),
)


def _mix(hex_colour: str, white_fraction: float) -> str:
    rgb = tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(v + (255 - v) * white_fraction) for v in rgb)
    return "#%02X%02X%02X" % mixed


def _darken(hex_colour: str, factor: float = 0.78) -> str:
    rgb = tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    dark = tuple(round(v * factor) for v in rgb)
    return "#%02X%02X%02X" % dark


def _stage_height(resolution: int, name: str) -> float:
    if name == "Classifier":
        return 22.0
    return {32: 54.0, 16: 43.0, 8: 34.0, 4: 26.0, 1: 18.0}[resolution]


def _stage_width(resolution: int, name: str) -> float:
    if name == "Classifier":
        return 9.0
    return {32: 30.0, 16: 24.0, 8: 18.0, 4: 13.0, 1: 9.0}[resolution]


def _stage_depth(channels: int, name: str) -> float:
    if name == "Classifier":
        return 4.0
    return 2.5 + 1.5 * math.log2(max(channels, 2))


def _font_path(bold: bool) -> str | None:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        windows / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists() or not candidate.is_absolute():
            return str(candidate)
    return None


class Scene:
    def __init__(self) -> None:
        self.elements: list[tuple] = []

    def polygon(
        self,
        points: Iterable[tuple[float, float]],
        fill: str,
        stroke: str = INK,
        width: float = 0.55,
    ) -> None:
        self.elements.append(("polygon", tuple(points), fill, stroke, width))

    def line(
        self,
        points: Iterable[tuple[float, float]],
        colour: str = CONNECTOR,
        width: float = 0.75,
    ) -> None:
        self.elements.append(("line", tuple(points), colour, width))

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        *,
        bold: bool = False,
        colour: str = INK,
    ) -> None:
        self.elements.append(("text", x, y, value, size, bold, colour))


def _add_arrow(scene: Scene, start: tuple[float, float], end: tuple[float, float]) -> None:
    x1, y1 = start
    x2, y2 = end
    scene.line(((x1, y1), (x2 - 2.2, y2)), width=0.8)
    scene.polygon(
        ((x2, y2), (x2 - 4.0, y2 + 2.0), (x2 - 4.0, y2 - 2.0)),
        CONNECTOR,
        CONNECTOR,
        0.0,
    )


def _add_poly_arrow(scene: Scene, points: tuple[tuple[float, float], ...]) -> None:
    """Draw a polyline whose final segment carries an arrowhead."""
    scene.line(points, width=0.8)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base_x, base_y = x2 - 4.0 * ux, y2 - 4.0 * uy
    scene.polygon(
        (
            (x2, y2),
            (base_x + 2.0 * px, base_y + 2.0 * py),
            (base_x - 2.0 * px, base_y - 2.0 * py),
        ),
        CONNECTOR,
        CONNECTOR,
        0.0,
    )


def _add_block(
    scene: Scene,
    centre_x: float,
    bottom: float,
    width: float,
    height: float,
    depth: float,
    colour: str,
) -> None:
    dx, dy = depth, depth * 0.62
    front_centre = centre_x - dx / 2.0
    left = front_centre - width / 2.0
    right = front_centre + width / 2.0
    scene.polygon(
        ((left, bottom + height), (right, bottom + height),
         (right + dx, bottom + height + dy), (left + dx, bottom + height + dy)),
        _mix(colour, 0.35),
    )
    scene.polygon(
        ((right, bottom), (right + dx, bottom + dy),
         (right + dx, bottom + height + dy), (right, bottom + height)),
        _darken(colour, 0.88),
    )
    scene.polygon(
        ((left, bottom), (right, bottom), (right, bottom + height), (left, bottom + height)),
        colour,
    )


def build_scene() -> Scene:
    scene = Scene()
    stage_name_font = 10.5
    shape_font = 8.7
    row1 = STAGES[:7]
    row2 = STAGES[7:]
    row1_centres = [45.0 + i * (425.0 / (len(row1) - 1)) for i in range(len(row1))]
    row2_centres = list(row1_centres)
    row1_bottom = 145.0
    row2_bottom = 48.0
    geometry = [
        (
            _stage_width(stage.resolution, stage.name),
            _stage_height(stage.resolution, stage.name),
            _stage_depth(stage.channels, stage.name),
        )
        for stage in STAGES
    ]

    # Legend and residual-path annotation.
    legend = (
        (FIXED, "fixed"),
        (SEARCH, "searchable L-stage"),
        (ATTENTION, "attention-enabled L-stage"),
        (DOWNSAMPLE, "searchable downsampling"),
    )
    legend_font = 7.8
    swatch = 7.0
    swatch_gap = 5.0
    item_gap = 18.0
    legend_widths = [stringWidth(label, "Helvetica", legend_font) for _, label in legend]
    total_legend_width = sum(swatch + swatch_gap + width for width in legend_widths)
    total_legend_width += item_gap * (len(legend) - 1)
    x = (WIDTH_PT - total_legend_width) / 2.0
    for (colour, label), label_width in zip(legend, legend_widths):
        scene.polygon(((x, 228.5), (x + swatch, 228.5),
                       (x + swatch, 235.5), (x, 235.5)), colour, INK, 0.35)
        scene.text(x + swatch + swatch_gap + label_width / 2.0,
                   232, label, legend_font, colour=MUTED)
        x += swatch + swatch_gap + label_width + item_gap
    scene.text(
        WIDTH_PT / 2.0,
        218.0,
        "Face size encodes spatial resolution; stack depth encodes channel count (log-scaled)",
        7.8,
        bold=True,
        colour=MUTED,
    )

    # Draw the upper row and its connectors.
    for i in range(len(row1) - 1):
        w1, h1, d1 = geometry[i]
        w2, h2, d2 = geometry[i + 1]
        y = row1_bottom + 0.52 * min(h1, h2)
        _add_arrow(
            scene,
            (row1_centres[i] + (w1 + d1) / 2.0 + 2.0, y),
            (row1_centres[i + 1] - (w2 + d2) / 2.0 - 2.0, y),
        )

    # The wraparound arrow preserves the left-to-right reading order on row two.
    split = len(row1)
    w_last, h_last, d_last = geometry[split - 1]
    w_first, h_first, d_first = geometry[split]
    upper_y = row1_bottom + 0.52 * h_last
    lower_y = row2_bottom + 0.52 * h_first
    _add_poly_arrow(
        scene,
        (
            (row1_centres[-1] + (w_last + d_last) / 2.0 + 2.0, upper_y),
            (WIDTH_PT - 7.0, upper_y),
            (WIDTH_PT - 7.0, 104.0),
            (7.0, 104.0),
            (7.0, lower_y),
            (row2_centres[0] - (w_first + d_first) / 2.0 - 2.0, lower_y),
        ),
    )

    # Draw the lower row and its connectors.
    for local_i in range(len(row2) - 1):
        i = local_i + len(row1)
        w1, h1, d1 = geometry[i]
        w2, h2, d2 = geometry[i + 1]
        y = row2_bottom + 0.52 * min(h1, h2)
        _add_arrow(
            scene,
            (row2_centres[local_i] + (w1 + d1) / 2.0 + 2.0, y),
            (row2_centres[local_i + 1] - (w2 + d2) / 2.0 - 2.0, y),
        )

    for centre, stage, (width, height, depth) in zip(row1_centres, row1, geometry[:split]):
        _add_block(scene, centre, row1_bottom, width, height, depth, stage.colour)
        scene.text(centre, 130.0, stage.name, stage_name_font, bold=True)
        scene.text(centre, 116.0, stage.shape, shape_font, colour=MUTED)

    for centre, stage, (width, height, depth) in zip(row2_centres, row2, geometry[split:]):
        _add_block(scene, centre, row2_bottom, width, height, depth, stage.colour)
        scene.text(centre, 34.0, stage.name, stage_name_font, bold=True)
        scene.text(centre, 20.0, stage.shape, shape_font, colour=MUTED)
    return scene


def render_pdf(scene: Scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(WIDTH_PT, HEIGHT_PT), pageCompression=1)
    pdf.setTitle("QA-DARTS supernet topology")
    pdf.setAuthor("Filippo Lucchesi")
    for element in scene.elements:
        kind = element[0]
        if kind == "polygon":
            _, points, fill, stroke, line_width = element
            path = pdf.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            path.close()
            pdf.setFillColor(HexColor(fill))
            pdf.setStrokeColor(HexColor(stroke))
            pdf.setLineWidth(line_width)
            pdf.drawPath(path, fill=1, stroke=1 if line_width > 0 else 0)
        elif kind == "line":
            _, points, colour, line_width = element
            pdf.setStrokeColor(HexColor(colour))
            pdf.setLineWidth(line_width)
            path = pdf.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            pdf.drawPath(path, fill=0, stroke=1)
        elif kind == "text":
            _, x, y, value, size, bold, colour = element
            pdf.setFillColor(HexColor(colour))
            pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            width = pdf.stringWidth(value, "Helvetica-Bold" if bold else "Helvetica", size)
            pdf.drawString(x - width / 2.0, y - size * 0.32, value)
    pdf.showPage()
    pdf.save()


def render_png(scene: Scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = DPI / PT_PER_IN
    image = Image.new("RGB", (round(WIDTH_PT * scale), round(HEIGHT_PT * scale)), "white")
    draw = ImageDraw.Draw(image)
    fonts: dict[tuple[float, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def point(point: tuple[float, float]) -> tuple[int, int]:
        return round(point[0] * scale), round((HEIGHT_PT - point[1]) * scale)

    def font(size: float, bold: bool):
        key = (size, bold)
        if key not in fonts:
            path = _font_path(bold)
            try:
                fonts[key] = ImageFont.truetype(path, round(size * scale)) if path else ImageFont.load_default()
            except OSError:
                fonts[key] = ImageFont.load_default()
        return fonts[key]

    for element in scene.elements:
        kind = element[0]
        if kind == "polygon":
            _, points, fill, stroke, line_width = element
            pixels = [point(p) for p in points]
            draw.polygon(pixels, fill=fill)
            if line_width > 0:
                draw.line(pixels + [pixels[0]], fill=stroke, width=max(1, round(line_width * scale)), joint="curve")
        elif kind == "line":
            _, points, colour, line_width = element
            draw.line([point(p) for p in points], fill=colour, width=max(1, round(line_width * scale)))
        elif kind == "text":
            _, x, y, value, size, bold, colour = element
            selected_font = font(size, bold)
            centre = point((x, y))
            bbox = draw.textbbox((0, 0), value, font=selected_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (centre[0] - text_width / 2, centre[1] - text_height / 2 - bbox[1]),
                value,
                font=selected_font,
                fill=colour,
            )
    image.save(output, dpi=(DPI, DPI), optimize=True)


def main() -> None:
    scene = build_scene()
    render_pdf(scene, PDF_PATH)
    render_png(scene, PNG_PATH)
    print(f"Wrote {PDF_PATH} ({WIDTH_IN:.2f} x {HEIGHT_IN:.2f} in, vector PDF)")
    print(f"Wrote {PNG_PATH} ({round(WIDTH_IN * DPI)} x {round(HEIGHT_IN * DPI)} px, {DPI} dpi)")


if __name__ == "__main__":
    main()
