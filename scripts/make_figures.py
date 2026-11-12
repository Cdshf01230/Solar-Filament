"""Figure for the report: the four stages on one test image (report/fig_pipeline.pdf).

python scripts/make_figures.py --images DIR --weights weights --out report/fig_pipeline.pdf [--name FILE]
"""

import argparse, sys
from pathlib import Path
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_pq import pipeline, postprocess  # noqa: E402
from filament_pq.refiner.infer import prep  # noqa: E402

WIN = 768


def best_window(label, step=128):
    """Top-left corner of the WIN x WIN window that contains the most instance pixels."""
    ii = cv2.integral((label > 0).astype(np.uint8))
    h, w = label.shape
    best = (-1, 0, 0)
    for y in range(0, h - WIN + 1, step):
        for x in range(0, w - WIN + 1, step):
            s = ii[y + WIN, x + WIN] - ii[y, x + WIN] - ii[y + WIN, x] + ii[y, x]
            if s > best[0]:
                best = (s, y, x)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    names = sorted(p.name for p in Path(a.images).glob('*.jpeg'))
    name = a.name or names[len(names) // 2]
    image = pipeline.read_image(Path(a.images) / name)
    out = pipeline.Predictor(a.weights, device=a.device)(image, name)
    inst = out['instances']
    label = np.zeros(image.shape, np.int32)
    for j, x in enumerate(inst, 1):
        label[postprocess.to_full(x)] = j
    y, x = best_window(label)
    win = (slice(y, y + WIN), slice(x, x + WIN))
    n, lab, stats, centers, ids, boxes = out['components']
    kept = {int(c.crop_id.rsplit('_', 1)[1]) for c in out['crops']}
    k = np.isin(lab, list(kept))
    d = (lab > 0) & ~k

    # the crop whose component overlaps the window most
    ov = [0] * len(out['crops'])
    for i, c in enumerate(out['crops']):
        x0, y0, x1, y1 = c.box
        ov[i] = max(0, min(x1, x + WIN) - max(x0, x)) * max(0, min(y1, y + WIN) - max(y0, y))
    ci = int(np.argmax(ov))
    c, r = out['crops'][ci], out['refined'][ci]
    arr, _, (nw, nh) = prep(c)

    plt.rcParams.update({'font.size': 7})
    fig, axs = plt.subplots(1, 6, figsize=(7.16, 1.45))
    axs[0].imshow(image, cmap='gray')
    axs[0].add_patch(plt.Rectangle((x, y), WIN, WIN, fill=False, ec='yellow', lw=0.8))
    axs[0].set_title('input (2048$^2$)')
    axs[1].imshow(out['probability'][win], cmap='magma', vmin=0, vmax=1)
    axs[1].set_title('(1) proposal map')
    rgb = np.dstack([image / 255.0] * 3)
    rgb[d] = [0.2, 0.45, 1]
    rgb[k] = [1, 0.8, 0]
    axs[2].imshow(rgb[win])
    axs[2].set_title('(2) kept / dropped')
    axs[3].imshow(arr['image'][:nh, :nw], cmap='gray')
    cs, _ = cv2.findContours(arr['anchor'][:nh, :nw].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for q in cs:
        axs[3].plot(q[:, 0, 0], q[:, 0, 1], c='cyan', lw=0.6)
    axs[3].set_title('(3) crop + prompt')
    axs[4].imshow(r['r3'][:nh, :nw] / 255.0, cmap='viridis', vmin=0, vmax=1)
    axs[4].set_title(f'refined, pres. {r["presence"][1]:.2f}')
    m = max(1, len(inst))
    hues = [0, 1, 2, 3, 4, 6, 8, 9]  # tab10 without grey/brown, visible on the grey disk
    palette = plt.cm.tab10(np.array(hues)[np.arange(m) % len(hues)])
    colors = np.concatenate([[[0, 0, 0, 0]], np.c_[palette[:, :3], np.full(m, 0.9)]])
    axs[5].imshow(image[win], cmap='gray')
    axs[5].imshow(colors[label[win]])
    axs[5].set_title('(4) instances')
    for ax in axs:
        ax.axis('off')
    plt.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.04)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(str(Path(a.out).with_suffix('.png')), dpi=200)
    print(
        'FIGURE',
        name,
        'window',
        (y, x),
        'crop',
        c.crop_id,
        'instances',
        len(inst),
        'crops',
        len(out['crops']),
        'components',
        len(ids),
    )


if __name__ == '__main__':
    main()
