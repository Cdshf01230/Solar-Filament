"""Fast CPU checks of the pure parts of the pipeline (no weights, no data, no GPU).

python -m pytest tests/test_units.py      or      python tests/test_units.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from filament_pq import metrics, postprocess as PP, submission as S  # noqa: E402
from filament_pq.config import IMAGE_SIZE, PipelineConfig  # noqa: E402


def rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))


def square(y, x, side, shape=(64, 64)):
    m = np.zeros(shape, bool)
    m[y : y + side, x : x + side] = True
    return m


# ---------------------------------------------------------------- metrics


def test_duplicates_count_twice_under_all_pairs_but_once_under_strict():
    gt, pred = [rle(square(0, 0, 10))], [rle(square(0, 0, 10)), rle(square(0, 0, 10))]
    host, strict = metrics.match(metrics.iou_matrix(gt, pred))
    assert host[:3] == [2, 0, 0] and strict[:3] == [1, 1, 0]


def test_iou_must_exceed_one_half():
    # prediction covers exactly half of the GT and nothing else: IoU = .5 is not a match
    half = square(0, 0, 10)
    half[:, 5:] = False
    host, strict = metrics.match(metrics.iou_matrix([rle(square(0, 0, 10))], [rle(half)]))
    assert host[:3] == strict[:3] == [0, 1, 1]


def test_empty_sides():
    assert metrics.match(metrics.iou_matrix([], [rle(square(0, 0, 4))]))[1][:3] == [0, 1, 0]
    assert metrics.match(metrics.iou_matrix([rle(square(0, 0, 4))], []))[1][:3] == [0, 0, 1]


def test_pq_formula():
    r = metrics.pq([2, 1, 1, 1.6])
    assert r['pq'] == pytest.approx(1.6 / 3) and r['sq'] == pytest.approx(0.8) and r['rq'] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- post-processing


def test_anchor_component_prefers_anchor_overlap_then_largest():
    m = square(0, 0, 20) | square(40, 40, 12)
    assert PP.select_anchor_component(m, square(42, 42, 4), 1).sum() == 144
    assert PP.select_anchor_component(m, np.zeros_like(m), 1).sum() == 400  # fallback: largest component
    assert PP.select_anchor_component(m, square(42, 42, 4), 200) is None  # below min_area


def test_merge_nms_drops_duplicates_and_keeps_instances_disjoint():
    cand = lambda y, x, side, score, cid: dict(y0=y, x0=x, m=np.ones((side, side), bool), score=score, crop_id=cid)
    cfg = PipelineConfig().postprocess
    out = PP.merge_nms([cand(0, 0, 20, 0.9, 'a'), cand(1, 1, 20, 0.8, 'dup'), cand(10, 10, 20, 0.7, 'overlap')], cfg)
    assert [o[3] for o in out] == [0.9, 0.7]  # the near-duplicate (IoU > .5) is dropped
    full = [PP.to_full(o) for o in out]
    assert not (full[0] & full[1]).any() and full[1].sum() == 400 - 100  # taken pixels removed


# ---------------------------------------------------------------- submission format


def test_rle_round_trip_and_validation():
    m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool)
    m[100:140, 200:260] = True
    counts = S.rle_counts(m)
    assert np.array_equal(S.decode(counts).astype(bool), m)
    assert S.validate([('img_1', counts)], ['img.jpeg'])['rows'] == 1
    with pytest.raises(ValueError):
        S.validate([('img_1', counts), ('img_1', counts)], ['img.jpeg'])
    with pytest.raises(ValueError):
        S.validate([('other_1', counts)], ['img.jpeg'])
    with pytest.raises(ValueError):
        S.validate([('img_1', S.rle_counts(np.zeros_like(m)))], ['img.jpeg'])


# ---------------------------------------------------------------- splits and configuration


def test_holdout_splits_are_date_disjoint_and_complete():
    sys.path.insert(0, str(REPO / 'scripts'))
    from make_splits import holdout_split

    assignment = json.loads((REPO / 'splits' / 'fold_assignment.json').read_text(encoding='utf-8'))
    for k in range(5):
        s = holdout_split(assignment, k)
        stored = json.loads((REPO / 'splits' / f'holdout{k}.json').read_text(encoding='utf-8'))
        assert s['train_sources'] == stored['train_sources'] and s['val_sources'] == stored['val_sources']


def test_documented_configuration_matches_code():
    doc, cfg = json.loads((REPO / 'configs' / 'pipeline.json').read_text(encoding='utf-8')), PipelineConfig()
    assert doc['proposal']['threshold'] == cfg.components.threshold
    assert doc['proposal']['min_component_area'] == cfg.components.min_area
    assert doc['ml']['threshold'] == cfg.components.ml_threshold
    inf = doc['inference']
    assert (inf['tta'], inf['rounds']) == (cfg.refiner.tta, cfg.refiner.rounds)
    assert (
        inf['threshold'] == cfg.postprocess.mask_threshold
        and inf['presence_threshold'] == cfg.postprocess.presence_threshold
    )
    assert inf['min_area'] == cfg.postprocess.min_area and inf['pad_each_side'] == cfg.components.crop_pad
    assert tuple(inf['fold_order']) == cfg.folds


# ---------------------------------------------------------------- inference cache


def test_cache_is_reused_only_with_identical_inputs():
    from filament_pq import pipeline

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'weights' / 'proposal').mkdir(parents=True)
        (d / 'weights' / 'proposal' / 'head.pt').write_bytes(b'v1')
        work = d / 'work'
        work.mkdir()
        key = pipeline.cache_key(['b.jpeg', 'a.jpeg'], d / 'weights')
        assert pipeline.check_cache(work, key) is False  # no finished upstream yet
        (work / 'cache.json').write_text(json.dumps(key))
        (work / 'manifest.csv').write_text('crop_id,source,ml\n')
        assert pipeline.check_cache(work, pipeline.cache_key(['a.jpeg', 'b.jpeg'], d / 'weights')) is True
        (d / 'weights' / 'proposal' / 'head.pt').write_bytes(b'v2')
        with pytest.raises(RuntimeError):
            pipeline.check_cache(work, pipeline.cache_key(['a.jpeg', 'b.jpeg'], d / 'weights'))
        with pytest.raises(RuntimeError):
            pipeline.check_cache(work, pipeline.cache_key(['a.jpeg'], d / 'weights'))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
