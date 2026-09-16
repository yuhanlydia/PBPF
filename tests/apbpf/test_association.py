from __future__ import annotations

import numpy as np
import pytest

from pbpf.apbpf.association import association_gap, association_margin_loss


def test_margin_loss_is_exactly_zero_when_every_example_is_ineligible():
    loss = association_margin_loss(
        np.array([0.1, 0.4]), np.array([0.2, 0.5]), np.array([False, False]), margin=0.03
    )

    assert loss == 0.0


def test_margin_loss_penalizes_missing_counterfactual_advantage_on_eligible_examples():
    loss = association_margin_loss(
        np.array([0.5, 0.4]), np.array([0.6, 0.35]), np.array([True, True]), margin=0.1
    )

    assert loss == pytest.approx(0.075)


def test_clustered_association_gap_has_positive_sign_when_shuffled_nll_is_larger():
    report = association_gap(
        aligned_nll=np.array([0.20, 0.30, 0.25, 0.40]),
        shuffled_nll=np.array([0.35, 0.45, 0.40, 0.55]),
        clusters=np.array(["source-a", "source-a", "source-b", "source-b"]),
        seed=17,
        replicates=200,
    )

    assert report.estimate == pytest.approx(0.15)
    assert report.lower > 0
    assert report.upper > 0
    assert report.replicates == 200


def test_torch_all_ineligible_loss_is_differentiable_zero():
    torch = pytest.importorskip("torch")
    aligned = torch.tensor([0.1, 0.4], requires_grad=True)
    loss = association_margin_loss(
        aligned, torch.tensor([0.3, 0.8]), torch.tensor([False, False]), margin=0.03
    )

    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(aligned.grad, torch.zeros_like(aligned))
