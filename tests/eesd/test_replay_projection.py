import copy
import hashlib
import pytest
from pbpf.eesd import replay_materialization as module


def normalized(s):
    return '\n'.join(x.rstrip() for x in s.replace('\r\n', '\n').strip().splitlines())


def pairsha(a,b):
    return hashlib.sha256((normalized(a)+'\0'+normalized(b)).encode()).hexdigest()


def fixture():
    selected={};private=[]
    for domain in module.DOMAINS:
        tests=[]
        for i in range(10):
            a,b=f'  {i}\r\n',f'{i} \n'
            tests.append(dict(id=str(i),input=a,output=b,pair_sha256=pairsha(a,b),
                original_pools=['apps_input_output'] if domain=='apps_replay' else ['private_tests','generated_tests'],
                known_exposed_input=i==0))
        member=dict(domain=domain,task_id=domain+':task',source_id=domain+':source',
            statement='  Complete statement.\n',visible_tests=[{'input':t['input'],'output':t['output']} for t in tests[:4]],
            reference_code='DO NOT DISCLOSE',ordered_tests=tests)
        evaluator=dict(domain=domain,task_id=member['task_id'],source_id=member['source_id'],ordered_tests=tests,
            known_exposed_input_sha256=[hashlib.sha256(normalized(tests[0]['input']).encode()).hexdigest()],
            locator='original:7',prior_provenance='receipt/source.json')
        selected[domain]=[('development',member)];private.append(evaluator)
    return selected,private


def test_projection_allowlist_raw_io_provenance_and_no_mutation():
    selected,private=fixture();before=copy.deepcopy((selected,private))
    public,evaluator=module.project_views(selected,private)
    assert (selected,private)==before
    assert len(public)==len(evaluator)==2
    for pub,ev,original in zip(public,evaluator,private):
        assert set(pub)=={'task_id','source_component_id','split','domain','statement','visible_tests'}
        assert pub['statement']=='  Complete statement.\n'
        assert all(set(t)=={'input','output'} for t in pub['visible_tests'])
        assert ev['tests']==original['ordered_tests']
        assert ev['known_exposed_input_sha256']==original['known_exposed_input_sha256']
        assert ev['locator']==original['locator'] and ev['prior_provenance']==original['prior_provenance']
        assert pub['visible_tests'][0]['input']=='  0\r\n'
    public[0]['visible_tests'][0]['input']='mutated'
    evaluator[0]['tests'][0]['original_pools'].append('mutated')
    assert (selected,private)==before


@pytest.mark.parametrize('change',[
    lambda s,p:p.pop(),
    lambda s,p:p.append(copy.deepcopy(p[0])),
    lambda s,p:p[0].update(source_id='wrong'),
    lambda s,p:p[0].update(domain='codecontests_replay'),
    lambda s,p:p[0]['ordered_tests'].pop(),
    lambda s,p:p[0]['ordered_tests'].reverse(),
    lambda s,p:p[0]['ordered_tests'][0].update(id=False),
    lambda s,p:p[0]['ordered_tests'][4].update(known_exposed_input=0),
    lambda s,p:p[0]['ordered_tests'][4].update(known_exposed_input=True),
    lambda s,p:p[0]['ordered_tests'][4].update(pair_sha256='a'*64),
    lambda s,p:p[0]['ordered_tests'][4].update(original_pools=[]),
    lambda s,p:p[0]['ordered_tests'][4].update(original_pools=['unknown']),
    lambda s,p:p[0]['ordered_tests'][4].update(original_pools=['apps_input_output']*2),
    lambda s,p:p[0]['ordered_tests'][4].update(input=' '*8193),
    lambda s,p:p[0]['ordered_tests'][4].update(output=' '*4097),
    lambda s,p:s['apps_replay'][0][1].update(statement='x'*6001),
    lambda s,p:s['apps_replay'][0][1].update(statement=' \n'),
    lambda s,p:s['apps_replay'][0][1]['visible_tests'][0].update(input='normalized-but-not-exact'),
    lambda s,p:p[0].update(known_exposed_input_sha256=['not-a-sha']),
    lambda s,p:p[0].update(known_exposed_input_sha256='a'*64),
    lambda s,p:s['apps_replay'].__setitem__(0,('train',s['apps_replay'][0][1])),
    lambda s,p:s['apps_replay'].append(s['apps_replay'][0]),
])
def test_projection_rejects_invalid_binding_or_test_contract(change):
    selected,private=fixture();change(selected,private)
    with pytest.raises(ValueError):module.project_views(selected,private)


def test_normalized_input_collision_rejected_even_with_valid_pair_hash():
    selected,private=fixture();test=private[0]['ordered_tests'][4]
    test['input']='0\n';test['pair_sha256']=pairsha(test['input'],test['output'])
    with pytest.raises(ValueError,match='input'):module.project_views(selected,private)


def test_hidden_input_cannot_appear_in_known_exposure_set():
    selected,private=fixture()
    private[0]['known_exposed_input_sha256'].append(hashlib.sha256(normalized(private[0]['ordered_tests'][4]['input']).encode()).hexdigest())
    with pytest.raises(ValueError,match='expos'):module.project_views(selected,private)


def test_cross_domain_component_reuse_rejected():
    selected,private=fixture()
    selected['codecontests_replay'][0][1]['source_id']=private[0]['source_id']
    private[1]['source_id']=private[0]['source_id']
    with pytest.raises(ValueError,match='source'):module.project_views(selected,private)
