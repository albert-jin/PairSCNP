"""Numerical and alignment checks that do not require Detectron2 or a GPU."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from pairscnp.amodal_completion import amodal_view
from pairscnp.chroma_consistency import paired_view, structure_js
from pairscnp.geometry import structure_prior
from pairscnp.metrics import betti, metrics


def test_js_symmetry_identity_and_gradients():
    torch.manual_seed(2)
    a = torch.randn(2, 2, 12, 12, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    labels = torch.randint(2, (2, 1, 12, 12))
    geometry = torch.rand(2, 1, 12, 12, requires_grad=True)
    loss = structure_js(a, b, labels, geometry)
    assert loss.item() > 0
    torch.testing.assert_close(loss, structure_js(b, a, labels, geometry))
    assert structure_js(a, a, labels, geometry).item() == 0
    loss.backward()
    for logits in (a, b):
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0
    assert geometry.grad is None


def test_js_balances_classes_instead_of_pixel_counts():
    # One disagreeing foreground pixel receives half the total class weight,
    # independent of the number of agreeing background pixels.
    def example(width):
        a = torch.zeros(1, 2, 1, width)
        b = a.clone()
        a[0, :, 0, 0] = torch.tensor([3., -3.])
        b[0, :, 0, 0] = torch.tensor([-3., 3.])
        y = torch.zeros(1, 1, 1, width)
        y[..., 0] = 1
        return structure_js(a, b, y, torch.zeros_like(y))
    torch.testing.assert_close(example(2), example(100))
    # Only one present class: do not divide by two.
    a = torch.tensor([[[[3.]], [[-3.]]]])
    b = -a
    single = structure_js(a, b, torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
    torch.testing.assert_close(example(2), single / 2)


def test_views_are_seeded_and_occlusions_preserve_alignment():
    torch.manual_seed(3)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(2, (2, 1, 64, 64))
    g = torch.rand(2, 1, 64, 64)
    original_y = y.clone()
    v1 = paired_view(x, torch.Generator().manual_seed(4))
    v2 = paired_view(x, torch.Generator().manual_seed(4))
    assert torch.equal(v1, v2)
    assert not torch.equal(v1, x)
    result, mask = amodal_view(v1, y, g, torch.Generator().manual_seed(5), 600, probability=1)
    assert mask.any()
    outside = (~mask.bool()).expand_as(result)
    assert torch.equal(result[outside], v1[outside])
    assert torch.equal(y, original_y)
    result0, mask0 = amodal_view(x, y, g, torch.Generator().manual_seed(5), 600, probability=0)
    assert not mask0.any()
    assert torch.equal(result0, x)


def test_geometry_is_finite_on_thin_structures_and_constant_masks():
    thin = np.zeros((21, 21), dtype=bool)
    thin[3:18, 10] = True
    g = structure_prior(thin)
    assert g[10, 10] > 0
    assert g[~thin].max() > 0
    for mask in (thin, np.zeros_like(thin), np.ones_like(thin)):
        g = structure_prior(mask)
        assert np.isfinite(g).all() and g.min() >= 0 and g.max() <= 1
        assert not g[[0, -1], :].any()
        assert not g[:, [0, -1]].any()


def test_topology_and_empty_mask_conventions():
    empty = np.zeros((9, 9), dtype=bool)
    ring = empty.copy()
    ring[2:7, 2:7] = True
    ring[3:6, 3:6] = False
    assert betti(empty) == (0, 0)
    assert betti(ring) == (1, 1)
    assert betti(np.eye(5, dtype=bool)) == (1, 0)  # eight-connected foreground
    for mask in (empty, ring):
        result = metrics(mask, mask)
        assert result['dice'] == result['foreground_iou'] == result['cldice'] == 1
        assert result['betti0_error'] == result['betti1_error'] == 0
    result = metrics(empty, ring)
    assert result['dice'] == result['cldice'] == 0
    assert result['betti0_error'] == result['betti1_error'] == 1


def test_upstream_scnp_loss_has_finite_gradients_and_prefers_correct_logits():
    path = Path(__file__).resolve().parents[1] / 'pairscnp/author_deeplab/loss.py'
    spec = importlib.util.spec_from_file_location('upstream_scnp_loss', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    criterion = module.SCNPCEDiceLoss()
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 5:11, 5:11] = 1
    correct = torch.nn.functional.one_hot(labels, 2).permute(0, 3, 1, 2).float() * 8 - 4
    correct.requires_grad_()
    loss = criterion(correct, labels)
    assert loss < criterion(-correct, labels)
    loss.backward()
    assert torch.isfinite(correct.grad).all()
    assert correct.grad.abs().sum() > 0


def test_training_and_evaluation_share_numerical_settings():
    from pairscnp.runtime import configure_runtime
    configure_runtime(19)
    first = torch.rand(4)
    configure_runtime(19)
    assert torch.equal(first, torch.rand(4))
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32
