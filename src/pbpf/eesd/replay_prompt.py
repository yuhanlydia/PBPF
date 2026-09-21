"""Exact full-public Replay prompt, checked against the sealed admission audit."""
import hashlib
import json

POLICY='eesd-replay-synthesis-full-public-v1'

TEMPLATE='''Write a complete Python3 program that reads standard input and writes standard output.
Solve the task below. Return only the complete program, without explanation.
The four designated observations are JSON strings preserving the exact input/output text.

TASK STATEMENT
{statement}

FOUR DESIGNATED VISIBLE OBSERVATIONS
{observations}
'''

def messages_for(statement,tests):
    if not isinstance(statement,str) or len(tests)!=4: raise ValueError('Require full statement and exactly four observations')
    observations=[]
    for i,test in enumerate(tests,1):
        if not isinstance(test['input'],str) or not isinstance(test['output'],str): raise ValueError('IO must be exact strings')
        observations.append(f'Observation {i}\nInput: '+json.dumps(test['input'],ensure_ascii=False)+'\nOutput: '+json.dumps(test['output'],ensure_ascii=False))
    return [{'role':'user','content':TEMPLATE.format(statement=statement,observations='\n\n'.join(observations))}]

def render_and_verify(tokenizer, row, audit, family):
    """Fail before generation if message, rendered text or input IDs changed."""
    messages = messages_for(row['statement'], row['visible_tests'])
    message_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if message_sha != audit['message_sha256']:
        raise ValueError('message differs from admitted full-public prompt')
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False, return_token_type_ids=False, truncation=False)['input_ids']
    actual = dict(input_tokens=len(ids),
                  rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                  token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',', ':')).encode()).hexdigest())
    expected = audit['models'][family]
    if not 1 <= len(ids) <= 4096 or any(actual[key] != expected[key] for key in actual):
        raise ValueError('tokenization differs from admitted input or exceeds token limit')
    return rendered, {**actual, 'message_sha256': message_sha, 'prompt_policy': POLICY,
                      'input_clipped': False, 'visible_observations': 4}
