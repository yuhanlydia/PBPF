import hashlib
import json
import pytest
from pbpf.eesd.replay_prompt import messages_for, render_and_verify


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages=messages
        return 'rendered prompt'
    def __call__(self, text, **kwargs):
        assert kwargs==dict(add_special_tokens=False,return_token_type_ids=False,truncation=False)
        return {'input_ids':[1,2,3]}


def sha(data):return hashlib.sha256(data).hexdigest()
def compact(value):return json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()
@pytest.fixture
def case():
    row=dict(statement='完整 statement\n',visible_tests=[{'input':' a\r\n','output':' b \n'} for _ in range(4)])
    audit=dict(message_sha256=sha(compact(messages_for(row['statement'],row['visible_tests']))),models={'qwen25_7b':dict(input_tokens=3,rendered_prompt_sha256=sha(b'rendered prompt'),token_ids_sha256=sha(b'[1,2,3]'))})
    return row,audit


def test_raw_observations_retained_in_messages():
    message=messages_for('full statement\n',[{'input':' a\r\n','output':' b \n'}]*4)[0]['content']
    assert 'full statement\n' in message
    assert 'Input: " a\\r\\n"' in message and 'Output: " b \\n"' in message


def test_public_only_fields_and_frozen_token_identity(case):
    row,audit=case
    row.update(targets=['secret'],reference_solution='secret')
    tokenizer=Tokenizer()
    rendered,metadata=render_and_verify(tokenizer,row,audit,'qwen25_7b')
    assert rendered=='rendered prompt' and metadata['input_tokens']==3
    assert 'secret' not in str(tokenizer.messages)


@pytest.mark.parametrize('field',['input_tokens','rendered_prompt_sha256','token_ids_sha256'])
def test_changed_tokenization_rejected(case,field):
    row,audit=case
    audit['models']['qwen25_7b'][field]=100 if field=='input_tokens' else 'f'*64
    with pytest.raises(ValueError,match='token'):
        render_and_verify(Tokenizer(),row,audit,'qwen25_7b')


def test_changed_statement_rejected(case):
    row,audit=case;row['statement']='different task'
    with pytest.raises(ValueError,match='message'):
        render_and_verify(Tokenizer(),row,audit,'qwen25_7b')


def test_all_four_visible_required():
    with pytest.raises(ValueError):messages_for('task',[{'input':'a','output':'b'}]*3)
