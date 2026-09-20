import torch
from torch import nn
from pbpf.belief.diagnostic import BoundHistoryISBeliefModel
from pbpf.belief.features import BeliefBatch


def test_linear_binding_retains_correspondence_and_ignores_presentation():
    model = BoundHistoryISBeliefModel(2, 4, 8, difficulty_dim=1)
    model.pair_encoder = nn.Linear(10, 1, bias=False)
    with torch.no_grad():
        model.pair_encoder.weight.zero_()
        model.pair_encoder.weight[0, 0] = 1
    tests = torch.tensor([[[1., 0.], [-1., 0.]]])
    labels = torch.nn.functional.one_hot(torch.tensor([[0, 1]]), 5).float()
    aligned = model.encode_pairs(tests, labels)
    shuffled = model.encode_pairs(tests, labels.flip(1))
    torch.testing.assert_close(aligned, torch.tensor([[.5]]))
    torch.testing.assert_close(shuffled, torch.tensor([[-.5]]))
    torch.testing.assert_close(aligned, model.encode_pairs(tests.flip(1), labels.flip(1)))


def test_bound_proposal_is_causal_and_difficulty_is_histogram_only():
    torch.manual_seed(17)
    model = BoundHistoryISBeliefModel(8, 6, 12, difficulty_dim=2)
    tests = torch.randn(3, 6, 8)
    labels = torch.tensor([[0, 1, 0, 1, 2, 3]]).expand(3, -1).clone()
    batch = BeliefBatch(torch.randn(3,8), torch.randn(3,8), tests, labels)
    altered = labels.clone(); altered[:, :4] = 1 - altered[:, :4]
    shuffled = BeliefBatch(batch.task,batch.candidate,tests,altered)
    a,s = model.history_proposal(batch,4),model.history_proposal(shuffled,4)
    torch.testing.assert_close(a.mean[:,:2],s.mean[:,:2])
    assert (a.mean[:,2:]-s.mean[:,2:]).abs().max() > 1e-4
    altered = labels.clone(); altered[:,4:] = 4
    changed_tests = tests.clone(); changed_tests[:,4:] = 100
    future_changed = BeliefBatch(batch.task,batch.candidate,changed_tests,altered)
    torch.testing.assert_close(a.mean,model.history_proposal(future_changed,4).mean)


def test_interaction_only_diagnosis_requires_both_test_and_g():
    from pbpf.belief.diagnostic import InteractionOnlyHistoryISBeliefModel
    model = InteractionOnlyHistoryISBeliefModel(2,4,8,difficulty_dim=1)
    task,candidate,test = torch.randn(1,2),torch.randn(1,2),torch.tensor([[1.,0.]])
    z=torch.tensor([[[.5,1.,0.,0.]]])
    with torch.no_grad():
        model.test_projection.weight.zero_();model.test_projection.weight[0,0]=1
        model.interaction_head.weight.zero_();model.interaction_head.weight[0,0]=1
    d,v=model.likelihood_components(z,task,candidate,test)
    assert v[0,0,0] == 1
    zero_g=z.clone();zero_g[...,1:]=0
    torch.testing.assert_close(model.likelihood_components(zero_g,task,candidate,test)[1],torch.zeros_like(v))
    torch.testing.assert_close(model.likelihood_components(z,task,candidate,torch.zeros_like(test))[1],torch.zeros_like(v))
    reversed_d,reversed_v=model.likelihood_components(z,task,candidate,-test)
    torch.testing.assert_close(d,reversed_d)
    torch.testing.assert_close(v,-reversed_v)


def test_high_gain_changes_only_test_diagnosis_logit_scale_at_initialization():
    from pbpf.belief.diagnostic import InteractionOnlyHistoryISBeliefModel, HighGainHistoryISBeliefModel
    torch.manual_seed(19)
    base = InteractionOnlyHistoryISBeliefModel(8,6,12,difficulty_dim=2)
    torch.manual_seed(19)
    high = HighGainHistoryISBeliefModel(8,6,12,difficulty_dim=2)
    base_parameters = dict(base.named_parameters())
    for name, parameter in high.named_parameters():
        if name != 'interaction_head.weight':
            torch.testing.assert_close(parameter, base_parameters[name], rtol=0, atol=0)
    task,candidate,test = torch.randn(3,8),torch.randn(3,8),torch.randn(3,8)
    z=torch.randn(3,4,6)
    d0,v0=base.likelihood_components(z,task,candidate,test)
    d1,v1=high.likelihood_components(z,task,candidate,test)
    torch.testing.assert_close(d0,d1)
    torch.testing.assert_close(v1,10*v0)
    torch.testing.assert_close(high.likelihood(z,task,candidate,test).exp().sum(-1),torch.ones(3,4))
