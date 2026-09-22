#!/usr/bin/env python3
"""Run the direct correction protocol with Gemma 3's user-only chat format."""
import hashlib
import json
import builtins
from pathlib import Path
from types import FunctionType, SimpleNamespace

import transformers

from pbpf.apbpf import direct_execution as direct
from pbpf.eesd import correction_prompt
from pbpf.eesd import direct_corrections as adapter


GEMMA_POLICY = 'eesd-correction-gemma-user-only-v1'


def gemma_prompt(tokenizer, row, domain, max_input_tokens=4096, model_id=''):
    if model_id != 'google/gemma-3-4b-it' or max_input_tokens < 1:
        raise ValueError('Gemma 3 4B pinned correction prompt required')
    if len(row['tests']) != 4 or len({test['id'] for test in row['tests']}) != 4:
        raise ValueError('four unique public correction slots required')
    public_size = len(correction_prompt._text(row['task_text'])) + sum(
        len(correction_prompt._text(test[key])) for test in row['tests']
        for key in correction_prompt.FIELDS)
    caps = [None] if public_size <= 4 * max_input_tokens else []
    cap = 2048
    while cap:
        caps.append(cap)
        cap = cap * 3 // 4
    caps.append(0)
    for cap in caps:
        view, clipped = correction_prompt._view(row, cap)
        prompt = correction_prompt.SYSTEM + '\n\n' + correction_prompt.user_prompt(view, domain)
        messages = [{'role': 'user', 'content': prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        ids = tokenizer(rendered, add_special_tokens=False,
                        return_token_type_ids=False)['input_ids']
        if len(ids) <= max_input_tokens:
            return {'prompt': prompt, 'messages': messages, 'rendered': rendered,
                    'input_ids': ids,
                    'prompt_token_ids_sha256': hashlib.sha256(json.dumps(
                        ids, separators=(',', ':')).encode()).hexdigest(),
                    'prompt_metadata': {'policy': GEMMA_POLICY,
                        'input_tokens': len(ids), 'max_input_tokens': max_input_tokens,
                        'field_character_cap': cap, 'truncated_fields': clipped}}
    raise ValueError('Gemma correction prompt cannot fit the locked input budget')


def main():
    original_tokenizer = transformers.AutoTokenizer
    original_direct_main = adapter.direct_main
    original_inventory = adapter.source_inventory
    original_policy = correction_prompt.POLICY
    from transformers import Gemma3ForConditionalGeneration
    local_model = '/root/PBPF-models/gemma3_4b'
    proof = json.loads(Path('/root/eesd-migration-20260921/gemma3-4b-model-proof.json').read_text())
    if (proof['model_id'] != 'google/gemma-3-4b-it'
            or proof['revision'] != '093f9f388b31de276ce2de164bdc2081324b9767'
            or proof['snapshot'] != local_model):
        raise ValueError('Gemma local model proof differs')
    for entry in proof['files']:
        if (Path(local_model) / entry['file']).stat().st_size != entry['bytes']:
            raise ValueError('Gemma local model size differs from proof')

    class LocalGemmaTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            if model_id != 'google/gemma-3-4b-it':
                raise ValueError('Gemma model identity changed')
            return original_tokenizer.from_pretrained(local_model, local_files_only=True)

    class LocalGemmaModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            if model_id != 'google/gemma-3-4b-it':
                raise ValueError('Gemma model identity changed')
            kwargs.pop('revision', None)
            return Gemma3ForConditionalGeneration.from_pretrained(local_model, **kwargs)

    def patched_main(original):
        namespace = dict(original.__globals__)
        original_import = builtins.__import__
        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            module = original_import(name, globals, locals, fromlist, level)
            if name == 'transformers' and 'AutoTokenizer' in fromlist:
                return SimpleNamespace(AutoTokenizer=LocalGemmaTokenizer,
                    AutoModelForCausalLM=LocalGemmaModel,
                    BitsAndBytesConfig=transformers.BitsAndBytesConfig)
            return module
        namespace['__builtins__'] = {**vars(builtins), '__import__': local_import}
        namespace.update(execute_stdin=direct.execute_stdin,
                         execute_call=direct.execute_call,
                         prepare_correction_prompt=gemma_prompt)
        return FunctionType(original.__code__, namespace, original.__name__,
                            original.__defaults__, original.__closure__)

    def inventory(stage):
        values = original_inventory(stage)
        if stage == 'generate':
            values['scripts/generate_eesd_direct_gemma_corrections.py'] = adapter.sha(__file__)
        return values

    try:
        adapter.direct_main = patched_main
        adapter.source_inventory = inventory
        correction_prompt.POLICY = GEMMA_POLICY
        return adapter.run('generate')
    finally:
        correction_prompt.POLICY = original_policy
        adapter.source_inventory = original_inventory
        adapter.direct_main = original_direct_main


if __name__ == '__main__':
    raise SystemExit(main())
