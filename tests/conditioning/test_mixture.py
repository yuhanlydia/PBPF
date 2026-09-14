import copy
import math

import pytest

torch = pytest.importorskip("torch")

from pbpf.conditioning.mixture import (
    whole_sequence_mixture_loss, sample_components_once, serial_mixture_backward,
)


def test_aa_bb_whole_sequence_mixture_excludes_chimeras():
    # Rows enumerate AA, AB, BA, BB; each component is deterministic A or B.
    component_tokens = torch.tensor([
        [[1., 1.], [0., 0.]], [[1., 0.], [0., 1.]],
        [[0., 1.], [1., 0.]], [[0., 0.], [1., 1.]],
    ]).log()
    weights = torch.full((4, 2), -math.log(2.))
    loss = whole_sequence_mixture_loss(component_tokens, weights, reduction="none")
    torch.testing.assert_close((-loss).exp(), torch.tensor([.5, 0., 0., .5]))
    remix = (component_tokens.exp() * weights.exp()[..., None]).sum(1).prod(-1)
    torch.testing.assert_close(remix, torch.full((4,), .25))


def test_sampled_component_is_reused_across_every_token_of_a_repair():
    components = sample_components_once(torch.tensor([[.5, .5]]).log(), num_sequences=4,
                                        generator=torch.Generator().manual_seed(8))
    assert components.shape == (1, 4)
    assert components.tolist() == [[0, 0, 1, 1]]
    patches = ["".join("AB"[component] for _ in range(5)) for component in components[0].tolist()]
    assert patches == ["AAAAA", "AAAAA", "BBBBB", "BBBBB"]


def test_masked_negative_infinity_tokens_do_not_poison_mixture():
    tokens = torch.tensor([[[-math.log(2.), -torch.inf], [-math.log(4.), -torch.inf]]], requires_grad=True)
    weights = torch.tensor([[.75, .25]], requires_grad=True)
    loss = whole_sequence_mixture_loss(tokens, weights.log(), token_mask=torch.tensor([[1, 0]]))
    assert loss.item() == pytest.approx(-math.log(.4375))
    loss.backward()
    assert weights.grad is None
    assert torch.isfinite(tokens.grad).all()
    assert torch.equal(tokens.grad[:, :, 1], torch.zeros(1, 2))


@pytest.mark.parametrize("dropout", [0., .4])
def test_serial_parameter_gradients_match_materialized_mixture(dropout):
    torch.manual_seed(18)
    actor = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Tanh(),
                                torch.nn.Dropout(dropout), torch.nn.Linear(6, 2)).double()
    serial_actor = copy.deepcopy(actor)
    z = torch.randn(2, 3, 3, dtype=torch.double, requires_grad=True)
    weights = torch.tensor([[.6, .3, .1], [.1, .2, .7]], dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[1, 1], [1, 0]])
    rng = torch.get_rng_state()
    token_values = torch.stack([actor(z[:, m].detach()).log_softmax(-1) for m in range(3)], 1)
    materialized = whole_sequence_mixture_loss(token_values, weights.log(), token_mask=mask)
    materialized.backward()
    torch.set_rng_state(rng)
    value = serial_mixture_backward(lambda latent, index: serial_actor(latent).log_softmax(-1),
                                     z, weights.log(), parameters=serial_actor.parameters(), token_mask=mask)
    torch.testing.assert_close(value, materialized.detach(), rtol=1e-12, atol=1e-12)
    for expected, actual in zip(actor.parameters(), serial_actor.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-12)
    assert z.grad is None and weights.grad is None


def test_serial_reports_impossible_targets_without_nan_gradients():
    with pytest.raises(ValueError, match="finite mixture"):
        serial_mixture_backward(lambda z, i: z.new_full((1, 2), -torch.inf),
                                 torch.zeros(1, 2, 3), torch.zeros(1, 2),
                                 parameters=[torch.nn.Parameter(torch.zeros(()))])


@pytest.mark.parametrize("weights", [torch.tensor([[torch.nan, 0.]]),
                                      torch.tensor([[-torch.inf, -torch.inf]])])
def test_invalid_belief_weights_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        sample_components_once(weights)


def test_sampling_never_allocates_zero_mass_at_random_zero_boundary():
    # CPU half RNG seed 2616 draws exactly 0; a left-closed search selects mass 0.
    weights = torch.tensor([[-torch.inf, 0.]], dtype=torch.float16)
    ids = sample_components_once(weights, generator=torch.Generator().manual_seed(2616))
    assert ids.tolist() == [[1]]


def test_many_half_precision_calls_do_not_round_into_zero_mass_tail():
    weights = torch.tensor([[0., -torch.inf]], dtype=torch.float16)
    ids = sample_components_once(weights, num_sequences=4096, generator=torch.Generator().manual_seed(1))
    assert torch.equal(ids, torch.zeros(1, 4096, dtype=torch.long))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("serial", [False, True])
def test_long_low_precision_sequences_match_promoted_loss_and_gradients(dtype, serial):
    values = torch.full((1, 2, 8192), -10., dtype=dtype, requires_grad=True)
    weights = torch.tensor([[-.1, -2.3]], dtype=dtype)
    reference = values.detach().float().requires_grad_()
    normalized = weights.float().log_softmax(-1)
    expected = -torch.logsumexp(normalized + reference.sum(-1), -1).mean()
    expected.backward()
    if serial:
        actual = serial_mixture_backward(lambda z, index: values[:, index],
                                         torch.zeros(1, 2, 1), weights, parameters=[values])
    else:
        actual = whole_sequence_mixture_loss(values, weights)
        actual.backward()
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual)
    assert actual.item() == pytest.approx(81920.)
    torch.testing.assert_close(actual, expected.detach())
    assert values.grad is not None and torch.isfinite(values.grad).all()
    torch.testing.assert_close(values.grad, reference.grad.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("existing_grad", [False, True])
def test_late_serial_replay_failure_preserves_existing_gradients(existing_grad):
    parameter = torch.nn.Parameter(torch.tensor([[-1., -2.], [-3., -4.]]))
    if existing_grad:
        parameter.grad = torch.full_like(parameter, 7.)
    before = parameter.grad
    calls = [0, 0]

    def changed_replay(z, index):
        calls[index] += 1
        # First component replay succeeds; second changes only on its replay.
        return parameter[index:index+1] - (1. if index == 1 and calls[index] == 2 else 0.)

    with pytest.raises(ValueError, match="replay changed"):
        serial_mixture_backward(changed_replay, torch.zeros(1, 2, 1), torch.zeros(1, 2), parameters=[parameter])
    assert calls == [2, 2]
    assert parameter.grad is before
    if existing_grad:
        assert torch.equal(parameter.grad, torch.full_like(parameter, 7.))


def test_serial_success_atomically_adds_gradients_and_preserves_unused_parameters():
    parameter = torch.nn.Parameter(torch.tensor([[[-1., -2.], [-3., -4.]]], dtype=torch.double))
    unused = torch.nn.Parameter(torch.zeros(1, dtype=torch.double))
    parameter.grad = torch.full_like(parameter, 7.)
    unused.grad = torch.ones_like(unused)
    before_unused = unused.grad
    reference = parameter.detach().clone().requires_grad_()
    expected_loss = whole_sequence_mixture_loss(reference, torch.tensor([[.3, .7]], dtype=torch.double).log())
    expected_loss.backward()
    actual = serial_mixture_backward(lambda z, index: parameter[:, index],
                                     torch.zeros(1, 2, 1), torch.tensor([[.3, .7]], dtype=torch.double).log(),
                                     parameters=[parameter, unused])
    torch.testing.assert_close(actual, expected_loss.detach())
    torch.testing.assert_close(parameter.grad, 7. + reference.grad, rtol=1e-12, atol=1e-12)
    assert unused.grad is before_unused
    assert unused.grad.item() == 1.


@pytest.mark.parametrize("invalid", ["duplicate", "nonleaf", "frozen", "empty"])
def test_serial_rejects_invalid_parameter_lists_without_modifying_gradients(invalid):
    parameter = torch.nn.Parameter(torch.tensor([[-1., -2.]]))
    entries = {"duplicate": [parameter, parameter], "nonleaf": [parameter + 1.],
               "frozen": [parameter.detach()], "empty": []}[invalid]
    with pytest.raises(ValueError, match="parameters"):
        serial_mixture_backward(lambda z, index: parameter, torch.zeros(1, 1, 1),
                                 torch.zeros(1, 1), parameters=entries)
    assert parameter.grad is None


def test_serial_shared_parameter_accumulates_broadcast_gradients():
    parameter = torch.nn.Parameter(torch.tensor([[-1., -2., -3.]], dtype=torch.double))
    loss = serial_mixture_backward(lambda z, index: parameter, torch.zeros(1, 2, 1),
                                   torch.tensor([[.25, .75]], dtype=torch.double).log(), parameters=[parameter])
    assert loss.item() == pytest.approx(6.)
    torch.testing.assert_close(parameter.grad, -torch.ones_like(parameter), rtol=1e-12, atol=1e-12)
