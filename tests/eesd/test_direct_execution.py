"""Trusted tiny programs only; these tests deliberately use direct execution."""
import json
import os
import pytest
from pbpf.apbpf.direct_execution import execute_stdin,execute_call

pytestmark=pytest.mark.skipif(os.geteuid()!=0,reason='direct profile requires root parent to drop uid')

@pytest.mark.parametrize('code,expected,outcome',[
    ('print(input())','hello\n','PASS'),
    ('print("wrong")','hello\n','WRONG_OUTPUT'),
    ('not python !','hello\n','COMPILE_ERROR'),
    ('raise ValueError("test")','hello\n','RUNTIME_EXCEPTION'),
    ('while True: pass','hello\n','TIMEOUT'),
])
def test_stdin_categories(code,expected,outcome):
    result=execute_stdin(code,{'input':'hello','expected':expected},timeout=.3)
    assert result['outcome']==outcome
    assert set(result)>={'stdout','stderr','returncode','timed_out','stdout_truncated'}


def test_native_stdin_comparison_and_newline():
    r=execute_stdin('import sys; print(repr(sys.stdin.read()))',{'input':'','expected':"'\\n'\n"})
    assert r['outcome']=='PASS' and r['terminal_newline_added'] is True
    assert execute_stdin('print(1.00001)',{'input':'','expected':'1.0'})['outcome']=='PASS'


def test_uid_groups_env_root_access_and_resources(monkeypatch):
    monkeypatch.setenv('DIRECT_SECRET_TEST','must-not-be-inherited')
    code='''import os,json,resource
try:
 os.listdir('/root')
 root_access=True
except PermissionError:
 root_access=False
print(json.dumps(dict(uid=os.getuid(),gid=os.getgid(),groups=os.getgroups(),env=dict(os.environ),root_access=root_access,cwd=os.getcwd(),limits={k:resource.getrlimit(getattr(resource,k)) for k in ['RLIMIT_AS','RLIMIT_FSIZE','RLIMIT_NOFILE','RLIMIT_NPROC']})))
'''
    r=execute_stdin(code,{'input':'','expected':''});v=json.loads(r['stdout'])
    assert v['uid']==v['gid']==65534 and v['groups']==[] and not v['root_access']
    assert 'DIRECT_SECRET_TEST' not in v['env']
    assert v['cwd'].startswith('/tmp/pbpf-direct-')
    assert v['limits']=={'RLIMIT_AS':[1024**3]*2,'RLIMIT_FSIZE':[8*1024**2]*2,'RLIMIT_NOFILE':[64]*2,'RLIMIT_NPROC':[64]*2}


@pytest.mark.parametrize('code,invocation,expected,error,outcome',[
    ('def f(): return 3','print(f())','3',False,'PASS'),
    ('def f(): return 3','print(f())','4',False,'WRONG_OUTPUT'),
    ('def f(): raise ValueError("expected")','f()','expected',True,'PASS'),
    ('bad syntax !','','',False,'COMPILE_ERROR'),
    ('raise ValueError("bad")','','',False,'RUNTIME_EXCEPTION'),
    ('while True: pass','','',False,'TIMEOUT'),
])
def test_call_semantics(code,invocation,expected,error,outcome):
    result=execute_call(code,{'input':invocation,'expected':expected,'expected_error':error},timeout=.3)
    assert result['outcome']==outcome


def test_call_dictionary_order_diagnostic():
    r=execute_call("print({'b':2,'a':1})",{'input':'','expected':"{'a': 1, 'b': 2}",'expected_error':False})
    assert r['outcome']=='WRONG_OUTPUT' and r['dictionary_order_only_mismatch'] is True


def test_explicit_direct_command_credentials_and_process_group_cleanup(monkeypatch):
    from pbpf.apbpf import direct_execution as module
    real_popen=module.subprocess.Popen;real_killpg=module.os.killpg;calls=[];killed=[]
    def trace(command,**kwargs):
        calls.append((command,kwargs))
        return real_popen(command,**kwargs)
    def killpg(pid,sig):
        killed.append((pid,sig));return real_killpg(pid,sig)
    monkeypatch.setattr(module.subprocess,'Popen',trace);monkeypatch.setattr(module.os,'killpg',killpg)
    result=execute_stdin('print(17)',{'input':'','expected':'17'})
    assert result['outcome']=='PASS' and len(calls)==1 and len(killed)==1
    command,kwargs=calls[0]
    assert command[:2]==['/usr/bin/python3','-I'] and len(command)==3
    assert kwargs['user']==kwargs['group']==65534 and kwargs['extra_groups']==[]
    assert kwargs['shell'] is False and kwargs['start_new_session'] is True and kwargs['close_fds'] is True
    assert set(kwargs['env'])=={'PATH','LANG','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'}
