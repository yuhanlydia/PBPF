import pytest

torch = pytest.importorskip("torch")

from pbpf.conditioning.soft_prompt import SoftPrefixProjector
from pbpf.conditioning.kv_delta import LowRankKVProjector


def test_soft_prefix_prepends_eight_trainable_embeddings_and_matching_masks_labels():
    torch.manual_seed(9)
    projector = SoftPrefixProjector(latent_dim=3, d_model=5, hidden_dim=7)
    z = torch.randn(2, 3, requires_grad=True)
    embeds = torch.randn(2, 4, 5)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    labels = torch.tensor([[4, 3, 2, -100], [1, 7, -100, -100]])
    combined = projector.prepend(z, inputs_embeds=embeds, attention_mask=mask, labels=labels)
    assert combined["inputs_embeds"].shape == (2, 12, 5)
    assert torch.equal(combined["inputs_embeds"][:, 8:], embeds)
    assert torch.equal(combined["attention_mask"][:, :8], torch.ones(2, 8, dtype=mask.dtype))
    assert torch.equal(combined["attention_mask"][:, 8:], mask)
    assert torch.equal(combined["labels"][:, :8], torch.full((2, 8), -100))
    assert torch.equal(combined["labels"][:, 8:], labels)
    combined["inputs_embeds"].square().sum().backward()
    assert z.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in projector.parameters())


def test_prefix_without_labels_and_bad_shapes():
    projector = SoftPrefixProjector(3, 5)
    result = projector.prepend(torch.zeros(1, 3), inputs_embeds=torch.zeros(1, 2, 5),
                               attention_mask=torch.ones(1, 2))
    assert "labels" not in result
    with pytest.raises(ValueError, match="shape"):
        projector.prepend(torch.zeros(1, 3), inputs_embeds=torch.zeros(1, 2, 4), attention_mask=torch.ones(1, 2))
    with pytest.raises(ValueError, match="shape"):
        projector(torch.zeros(1, 4))


@pytest.mark.parametrize("mode", ["k_only", "v_only", "kv"])
def test_low_rank_ablation_changes_only_selected_cache_channels(mode):
    torch.manual_seed(5)
    projector = LowRankKVProjector(latent_dim=3, layers=2, heads=2, head_dim=4, rank=2, mode=mode)
    z = torch.randn(2, 3, requires_grad=True)
    delta = projector(z)
    assert delta.key.shape == delta.value.shape == (2, 2, 2, 4, 4)
    assert torch.linalg.matrix_rank(delta.key).max() <= 2
    assert torch.linalg.matrix_rank(delta.value).max() <= 2
    cache = tuple((torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)) for _ in range(2))
    changed = delta.apply_to_cache(cache)
    total = 0.
    for old, new in zip(cache, changed):
        assert torch.equal(old[0], new[0]) == (mode == "v_only")
        assert torch.equal(old[1], new[1]) == (mode == "k_only")
        total = total + new[0].square().mean() + new[1].square().mean()
    total.backward()
    assert z.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters())
    with pytest.raises(ValueError, match="shape"):
        delta.apply_to_cache(((torch.zeros(2, 4, 3, 4), torch.zeros(2, 4, 3, 4)),) * 2)


def test_primary_factory_does_not_load_kv_ablation():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "from pbpf.conditioning import SoftPrefixProjector; import sys; SoftPrefixProjector(3,5); assert 'pbpf.conditioning.kv_delta' not in sys.modules"], check=True)
