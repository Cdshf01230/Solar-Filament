"""README workflow diagrams as plain SVG, one light and one dark file each.

    python scripts/make_diagrams.py --out docs/diagrams

Writes baseline_{light,dark}.svg and final_pipeline_{light,dark}.svg. No dependencies
beyond the standard library; text uses the viewer's system sans-serif font, so every
label is checked against a conservative width estimate before writing.
"""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

THEMES = {
    'light': dict(
        text='#1f2328',
        muted='#59636e',
        arrow='#59636e',
        bracket='#8c959f',
        input=('#f6f8fa', '#8c959f', '#59636e'),
        neural=('#e9f7ef', '#1a7f37', '#1a7f37'),
        decision=('#ffebe9', '#cf222e', '#cf222e'),
        post=('#fff1e5', '#bc4c00', '#bc4c00'),
        output=('#ddf4ff', '#0969da', '#0969da'),
        review=('#fbefff', '#8250df', '#8250df'),
        note=('#fff8f8', '#cf222e', '#cf222e'),
    ),
    'dark': dict(
        text='#e6edf3',
        muted='#9198a1',
        arrow='#9198a1',
        bracket='#6e7681',
        input=('#161b22', '#6e7681', '#9198a1'),
        neural=('#0f2d1c', '#3fb950', '#56d364'),
        decision=('#3a1418', '#f85149', '#ff7b72'),
        post=('#3a2210', '#db6d28', '#f0883e'),
        output=('#0c2d4f', '#388bfd', '#79c0ff'),
        review=('#2b1a47', '#a371f7', '#d2a8ff'),
        note=('#2a1215', '#f85149', '#ff7b72'),
    ),
}


def width(s, size, bold=False):
    """Conservative width estimate for common sans-serif fonts."""
    wide = sum(1 for c in s if c in 'MWmw@%')
    narrow = sum(1 for c in s if c in "il.,:;'|!·()/ ")
    rest = len(s) - wide - narrow
    return size * ((0.88 if bold else 0.82) * wide + 0.32 * narrow + (0.62 if bold else 0.58) * rest)


class SVG:
    def __init__(self, w, h, theme, title, desc):
        self.w, self.h, self.t = w, h, THEMES[theme]
        self.body = []
        self.right = 0
        self.title, self.desc = title, desc

    def text(self, x, y, s, size, color, bold=False, anchor='middle', max_w=None, italic=False):
        if max_w is not None:
            assert width(s, size, bold) <= max_w, f'label too wide: {s!r} ({width(s, size, bold):.0f} > {max_w})'
        style = (' font-weight="600"' if bold else '') + (' font-style="italic"' if italic else '')
        self.body.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}"{style}>{escape(s)}</text>'
        )

    def node(self, x, y, w, h, kind, title, lines=(), tag=None, dashed=False):
        fill, stroke, accent = self.t[kind]
        self.right = max(self.right, x + w)
        dash = ' stroke-dasharray="6 4"' if dashed else ''
        self.body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.6"{dash}/>'
        )
        n = len(lines) + (tag is not None)
        top = y + h / 2 - (22 + 17 * n) / 2 + 15  # vertically centre the text block
        cx, inner = x + w / 2, w - 18
        self.text(cx, top, title, 15, self.t['text'], bold=True, max_w=inner)
        for i, s in enumerate(lines):
            self.text(cx, top + 22 + 17 * i, s, 12.5, self.t['muted'], max_w=inner)
        if tag is not None:
            self.text(cx, top + 22 + 17 * len(lines), tag, 12.5, accent, bold=True, max_w=inner)

    def arrow(self, x1, y1, x2, y2, color=None, dashed=False, marker='a'):
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        self.body.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or self.t["arrow"]}" '
            f'stroke-width="1.8"{dash} marker-end="url(#{marker})"/>'
        )

    def bracket(self, x1, x2, y, label):
        c = self.t['bracket']
        lw = width(label, 12.5, True) + 16
        mid = (x1 + x2) / 2
        self.body.append(
            f'<path d="M{x1} {y+7} V{y} H{mid-lw/2} M{mid+lw/2} {y} H{x2} V{y+7}" '
            f'fill="none" stroke="{c}" stroke-width="1.2"/>'
        )
        self.text(mid, y + 4.5, label, 12.5, self.t['muted'], bold=True, max_w=x2 - x1)

    def write(self, path, markers):
        assert self.right <= self.w - 3, f'content reaches the right edge ({self.right} > {self.w - 3})'
        defs = ''.join(
            f'<marker id="{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{v}"/></marker>'
            for k, v in markers.items()
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" role="img" aria-labelledby="t d" font-family="{FONT}">'
            f'<title id="t">{escape(self.title)}</title><desc id="d">{escape(self.desc)}</desc>'
            f'<defs>{defs}</defs>' + '\n'.join(self.body) + '</svg>\n'
        )
        Path(path).write_text(svg, encoding='utf-8')


def row(x0, widths, gap):
    xs, x = [], x0
    for w in widths:
        xs.append(x)
        x += w + gap
    return xs


def baseline(theme, out):
    s = SVG(
        1000,
        268,
        theme,
        'Baseline: semantic head and connected components',
        'H-alpha image, frozen ConvNeXt-B semantic head on 768 px tiles, threshold chosen on quick74, '
        '8-connected components of at least 8 px, one instance per component. Failure modes: a faint '
        'section splits one filament; a bridge of false pixels merges two.',
    )
    t = s.t
    widths, gap, y, h = [118, 178, 172, 160, 160], 51, 40, 104
    xs = row(4, widths, gap)
    s.bracket(xs[1], xs[2] + widths[2], 14, 'Pixel classification')
    s.bracket(xs[3], xs[4] + widths[4], 14, 'Pixels to instances')
    s.node(xs[0], y, widths[0], h, 'input', 'H-alpha', ['full disk', '2048 × 2048'])
    s.node(
        xs[1],
        y,
        widths[1],
        h,
        'neural',
        'Semantic head',
        ['frozen ConvNeXt-B', 'FPN · U-Net · DDP-mini', '9 tiles, Hann blend'],
        '1.6–2.1 M trainable',
    )
    s.node(xs[2], y, widths[2], h, 'decision', 'Threshold', ['probability map', 'to binary mask'], 'chosen on quick74')
    s.node(xs[3], y, widths[3], h, 'post', 'Components', ['8-connected', 'area ≥ 8 px'])
    s.node(xs[4], y, widths[4], h, 'output', 'Instances', ['one per component'])
    for i in range(4):
        s.arrow(xs[i] + widths[i] + 2, y + h / 2, xs[i + 1] - 3, y + h / 2)
    # failure note under the components step
    nx, ny, nw, nh = xs[2], 182, xs[4] + widths[4] - xs[2], 70
    fill, stroke, accent = t['note']
    s.body.append(
        f'<path d="M{xs[3] + widths[3]/2} {y+h+2} V{ny-1}" stroke="{stroke}" stroke-width="1.4" '
        f'stroke-dasharray="3 3"/>'
    )
    s.body.append(
        f'<rect x="{nx}" y="{ny}" width="{nw}" height="{nh}" rx="10" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.2"/>'
    )
    s.text(nx + 16, ny + 22, 'Where it fails', 13, accent, bold=True, anchor='start', max_w=nw - 32)
    s.text(
        nx + 16,
        ny + 41,
        'A faint, thin section splits one filament into pieces.',
        12.5,
        t['muted'],
        anchor='start',
        max_w=nw - 32,
    )
    s.text(
        nx + 16,
        ny + 58,
        'A bridge of false pixels merges two filaments into one.',
        12.5,
        t['muted'],
        anchor='start',
        max_w=nw - 32,
    )
    s.write(Path(out) / f'baseline_{theme}.svg', {'a': t['arrow']})


def final_pipeline(theme, out):
    s = SVG(
        1000,
        292,
        theme,
        'Final pipeline: proposal, filter, refiner',
        'H-alpha image to 1 Proposal (DINOv3 ViT-S+/16 + 0.43 M head), 2 Filter (LightGBM on 32 features, '
        'keep score at least 0.5), 3 Refiner (DINOv3 ViT-B/16, 3 rounds, one filament per prompt), '
        'post-processing and pixel-disjoint instances. Each learned stage also yields an inspectable '
        'output: probability map, candidate crops (balanced or recall policy), instance masks with presence.',
    )
    t = s.t
    widths, gap, y, h = [112, 156, 156, 156, 150, 138], 24.8, 40, 108
    xs = row(4, widths, gap)
    s.bracket(xs[1], xs[3] + widths[3], 14, 'Three learned stages')
    s.node(xs[0], y, widths[0], h, 'input', 'H-alpha', ['full disk', '2048 × 2048'])
    s.node(xs[1], y, widths[1], h, 'neural', '1 · Proposal', ['DINOv3 ViT-S+/16', '+ 0.43 M head'], 'high recall')
    s.node(xs[2], y, widths[2], h, 'decision', '2 · Filter', ['LightGBM', '32 features'], 'keep if score ≥ 0.5')
    s.node(
        xs[3],
        y,
        widths[3],
        h,
        'neural',
        '3 · Refiner',
        ['DINOv3 ViT-B/16', '3 recurrent rounds'],
        '1 filament / prompt',
    )
    s.node(xs[4], y, widths[4], h, 'post', 'Post-process', ['native resampling', 'de-duplication'])
    s.node(xs[5], y, widths[5], h, 'output', 'Instances', ['pixel-disjoint', 'submission.csv'])
    for i in range(5):
        s.arrow(xs[i] + widths[i] + 2, y + h / 2, xs[i + 1] - 3, y + h / 2)
    # inspectable outputs of the three learned stages
    ry, rh = 206, 70
    review = [
        ('Probability map', 'components, p ≥ 0.05'),
        ('Candidate crops', 'balanced or recall'),
        ('Instance masks', 'mask + presence'),
    ]
    for i, (title, line) in enumerate(review, start=1):
        s.node(xs[i], ry, widths[i], rh, 'review', title, [line], dashed=True)
        cx = xs[i] + widths[i] / 2
        s.arrow(cx, y + h + 3, cx, ry - 4, color=t['review'][1], dashed=True, marker='r')
    s.text(
        xs[0] + widths[0] - 4, ry + 31, 'Inspectable', 12.5, t['review'][2], bold=True, anchor='end', max_w=widths[0]
    )
    s.text(xs[0] + widths[0] - 4, ry + 48, 'outputs', 12.5, t['review'][2], bold=True, anchor='end', max_w=widths[0])
    nx = xs[4]
    nw = xs[5] + widths[5] - nx
    s.text(nx + nw / 2, ry + 22, 'Test time: every stage averages', 12.5, t['muted'], max_w=nw)
    s.text(nx + nw / 2, ry + 39, 'its five fold models; the refiner', 12.5, t['muted'], max_w=nw)
    s.text(nx + nw / 2, ry + 56, 'also averages 8 D4 transforms.', 12.5, t['muted'], max_w=nw)
    s.write(Path(out) / f'final_pipeline_{theme}.svg', {'a': t['arrow'], 'r': t['review'][1]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='docs/diagrams')
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        baseline(theme, a.out)
        final_pipeline(theme, a.out)
    print('wrote', sorted(p.name for p in Path(a.out).glob('*.svg')))


if __name__ == '__main__':
    main()
