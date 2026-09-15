"""Offline geometry and full setup/loader integration checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from data import build_dataset
from data.realestate10k import (
    RealEstate10KDataset,
    RealEstate10KEpipolarConstraint,
    fundamental_matrix,
    parse_camera_file,
    safe_path,
)
from scripts.setup_realestate10k import prepare, select_indices


def test_geometry():
    K1 = np.array([[500., 0, 260], [0, 490, 150], [0, 0, 1]])
    K2 = np.array([[450., 0, 245], [0, 480, 140], [0, 0, 1]])
    angle = .17
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    P1 = np.column_stack([R, [.1, .2, .3]])
    P2 = np.column_stack([np.eye(3), [.4, -.1, .2]])
    F, valid = fundamental_matrix(K1, P1, K2, P2)
    X = np.array([[.1, .3, 3, 1], [-.5, .2, 4, 1], [.4, -.3, 5, 1]]).T
    p1, p2 = K1 @ P1 @ X, K2 @ P2 @ X
    p1, p2 = p1 / p1[2], p2 / p2[2]
    assert valid
    np.testing.assert_allclose(np.sum(p2 * (F @ p1), axis=0), 0, atol=1e-10)
    A = np.array([[.5, 0, 20], [0, .6, 10], [0, 0, 1]])
    transformed, _ = fundamental_matrix(A @ K1, P1, K2, P2)
    np.testing.assert_allclose(np.sum(p2 * (transformed @ A @ p1), axis=0), 0, atol=1e-10)
    zero, valid = fundamental_matrix(K1, P1, K1, P1)
    assert not valid and not zero.any()


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        select_indices(np.array([0, 1000000]), 8, .01)
    with pytest.raises(ValueError):
        safe_path(tmp_path, '../escape')
    bad = tmp_path / 'bad.txt'
    bad.write_text('https://example.com\n0 1 2\n')
    with pytest.raises(ValueError):
        parse_camera_file(bad)


def make_fixture(tmp_path, count=10):
    av = pytest.importorskip('av')
    poses, videos = tmp_path / 'poses/train', tmp_path / 'videos'
    poses.mkdir(parents=True)
    videos.mkdir()
    template = videos / 'template.mp4'
    with av.open(str(template), 'w') as out:
        stream = out.add_stream('mpeg4', rate=20)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(12):
            rgb = np.full((48, 64, 3), i * 15, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    for i in range(count):
        (videos / f'clip{i}.mp4').write_bytes(template.read_bytes())
        rows = [f'https://example.com/video{i}']
        for j in range(10):
            P = np.column_stack([np.eye(3), [j*.01, 0, 0]])
            rows.append(' '.join(map(str, [j*50000, .8, .9, .5, .5, 0, 0, *P.ravel()])))
        (poses / f'clip{i}.txt').write_text('\n'.join(rows))
    return argparse.Namespace(root=tmp_path/'prepared', poses_dir=poses.parent, videos_dir=videos,
        split='train', n_clips=count, n_frames=3, interval=.1, width=32, height=32,
        seed=42, max_attempts=20, tolerance_us=20000)


def test_setup_ten_clips_and_resume(tmp_path):
    args = make_fixture(tmp_path)
    prepare(args)
    prepare(args)
    manifest = json.loads((args.root/'manifests/train.json').read_text())
    assert len(manifest['clips']) == 10 and not manifest['failures']
    assert not list(args.root.glob('prepare-*'))
    cfg = OmegaConf.create({'data': {'name': 'realestate10k', 'cache_dir': str(args.root), 'n_eval': 10}})
    bundle = build_dataset(cfg)
    assert len(bundle.eval) == 10 and bundle.train is None
    sample = bundle.eval[0]
    assert sample['video'].shape == (3, 3, 32, 32)
    assert sample['wan_video'].shape == (3, 5, 32, 32)
    assert sample['wan_frame_mask'].tolist() == [True, True, True, False, False]
    torch.testing.assert_close(sample['wan_video'][:, -1], sample['video'][:, -1])
    assert sample['video'].dtype == torch.float32
    assert sample['prompt'] == 'a realistic real estate interior'
    assert sample['noise_seed'].item() == 42
    assert sample['fundamental_valid'].all()
    np.testing.assert_allclose(sample['K'][0], [[25.6, 0, 16], [0, 28.8, 16], [0, 0, 1]])
    np.testing.assert_array_equal(sample['actual_timestamps_us'], [0, 100000, 200000])
    np.testing.assert_array_equal(sample['timestamps_us'], sample['actual_timestamps_us'])
    args.width += 16
    with pytest.raises(ValueError, match='settings differ'):
        prepare(args)


def test_epipolar_constraint_numpy_and_torch():
    constraint = RealEstate10KEpipolarConstraint(tolerance_px=0.5)
    # F p1 gives the horizontal line y = reference y.
    F = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    refs = np.array([[[2., 3.], [7., 5.]], [[1., 4.], [3., 8.]]])
    targets = refs + np.array([[[0., 2.], [0., -1.]], [[0., 4.], [0., 2.]]])
    Fs = np.broadcast_to(F, (2, 3, 3)).copy()
    valid = np.array([True, False])
    violation = constraint.h(refs, targets, Fs, valid)['epipolar_px']
    np.testing.assert_allclose(violation[0], [1.5, 0.5])
    assert np.isinf(violation[1]).all()
    projected, mask = constraint.project_feasible(refs, targets, Fs, valid)
    assert mask.tolist() == [[True, True], [False, False]]
    np.testing.assert_allclose(projected[0, :, 1], refs[0, :, 1])
    np.testing.assert_allclose(projected[1], targets[1])

    torch_targets = torch.tensor(targets, requires_grad=True)
    torch_violation = constraint.h(torch.tensor(refs), torch_targets, torch.tensor(Fs), valid)['epipolar_px']
    torch_violation[0].sum().backward()
    assert torch_targets.grad is not None and torch.isfinite(torch_targets.grad).all()


def test_shortfall_recorded(tmp_path):
    args = make_fixture(tmp_path, 1)
    (args.videos_dir/'clip0.mp4').unlink()
    with pytest.raises(RuntimeError, match='Only 0/1'):
        prepare(args)
    manifest = json.loads((args.root/'manifests/train.json').read_text())
    assert len(manifest['failures']) == 1 and manifest['clips'] == []
    with pytest.raises(ValueError, match='No prepared clips'):
        RealEstate10KDataset(args.root)
