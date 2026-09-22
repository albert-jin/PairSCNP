"""Data preparation, cache alignment and published manifest integrity."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pairscnp.paths import ROOT, sha256
from scripts import prepare_data


def test_manifests_and_reference_ids():
    ids = json.loads((ROOT / 'results/reported_evaluation_ids.json').read_text())
    for dataset, counts in [('topomortar', (50, 20, 350)), ('crack500', (243, 49, 200))]:
        manifest = json.loads((ROOT / 'data' / dataset / 'manifest.json').read_text())
        assert tuple(len(manifest['splits'][s]) for s in ('train', 'val', 'test')) == counts
        split_hashes = []
        for rows in manifest['splits'].values():
            assert len({r['id'] for r in rows}) == len(rows)
            split_hashes.append({r['image_sha256'] for r in rows})
            for row in rows:
                for field in ('image', 'mask'):
                    assert not Path(row[field]).is_absolute()
                    assert '..' not in Path(row[field]).parts
                    assert len(bytes.fromhex(row[field + '_sha256'])) == 32
        assert not split_hashes[0] & split_hashes[1]
        assert not split_hashes[0] & split_hashes[2]
        assert not split_hashes[1] & split_hashes[2]
        assert set(ids[dataset]) <= {r['id'] for r in manifest['splits']['test']}


@pytest.mark.parametrize('rotate', [False, True])
def test_preparation_checks_hashes_and_honors_recorded_orientation(tmp_path, monkeypatch, rotate):
    raw = tmp_path / 'raw'
    raw.mkdir()
    aligned = np.zeros((16, 24, 3), dtype=np.uint8)
    aligned[4:12, 10:14] = 255
    image = Image.fromarray(aligned)
    if rotate:
        image = image.transpose(Image.Transpose.ROTATE_270)
    image.save(raw / 'image.png')
    Image.fromarray(aligned[..., 0]).save(raw / 'mask.png')
    row = {'id': 'sample', 'image': 'image.png', 'mask': 'mask.png',
        'image_sha256': sha256(raw / 'image.png'), 'mask_sha256': sha256(raw / 'mask.png'),
        'mask_threshold': 127, 'cache': 'data/crack500/cache/train_sample.npz'}
    if rotate:
        row['image_rotation_ccw'] = 90
    manifest_path = tmp_path / 'data/crack500/manifest.json'
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({'dataset': 'crack500', 'splits': {'train': [row], 'val': [], 'test': []}}))
    monkeypatch.setattr(prepare_data, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['prepare_data.py', '--dataset', 'crack500', '--raw-root', str(raw)])
    prepare_data.main()
    with np.load(tmp_path / row['cache']) as cache:
        np.testing.assert_array_equal(cache['image'], aligned)
        np.testing.assert_array_equal(cache['mask'], aligned[..., 0] > 127)
        assert cache['geometry'].dtype == np.float16
    runtime = json.loads((tmp_path / 'data/crack500/prepared_manifest.json').read_text())
    assert runtime['splits']['train'][0]['cache_sha256'] == sha256(tmp_path / row['cache'])
    (raw / 'image.png').write_bytes(b'changed file')
    with pytest.raises(FileNotFoundError, match='hash-matching image'):
        prepare_data.main()


def test_training_crop_keeps_target_and_geometry_aligned(tmp_path, monkeypatch):
    from pairscnp import data
    mask = np.zeros((520, 540), dtype=np.uint8)
    mask[90:400, 200:220] = 1
    image = np.repeat((mask * 255)[..., None], 3, axis=2)
    np.savez(tmp_path / 'sample.npz', image=image, mask=mask, geometry=mask.astype(np.float16))
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    random.seed(1)
    x, y, geometry, name = data.TrainingData([{'cache': 'sample.npz', 'id': 'sample'}])[0]
    assert x.shape == (3, 512, 512) and y.shape == (512, 512)
    assert name == 'sample'
    assert torch.equal(y, geometry[0].long())
    assert torch.equal(y.bool(), x[0] > 0)
