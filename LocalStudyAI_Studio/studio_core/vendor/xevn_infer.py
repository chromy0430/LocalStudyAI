"""Xevn single-pass decisions. Requires torch, transformers>=5.2, safetensors.

python xevn_infer.py --model quevirus/Xevn-4B --request request.json
Request: {"state": "...", "questions": {"q": {"type": "choice",
"instructions": "...", "criteria": {"a": "...", "b": "..."}}}}
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DEFAULT_TEMPS = {'choice': 1.1, 'noul': 1.2, 'score': 1.2}


def to_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)


def build(question, state):
    state = to_text(state)
    kind, criteria = question['type'], question.get('criteria')
    if kind == 'noul':
        if isinstance(criteria, list):
            criteria = {c.get('key'): c.get('description') for c in criteria if isinstance(c, dict)}
        if isinstance(criteria, dict) and criteria.get('true') is not None and criteria.get('false') is not None:
            options = [('yes', 'Yes: ' + to_text(criteria['true'])), ('no', 'No: ' + to_text(criteria['false']))]
        else:
            options = [('yes', 'Yes'), ('no', 'No')]
    elif kind == 'choice':
        if not isinstance(criteria, dict):
            raise ValueError('choice criteria must be an object')
        options = [(str(k), to_text(v) if v not in (None, '') else str(k)) for k, v in criteria.items()]
    elif kind == 'score':
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError('score criteria must contain 2-10 levels')
        options = [(str(i), to_text(v)) for i, v in enumerate(criteria)]
    else:
        raise ValueError('unknown question type')
    if not 2 <= len(options) <= 26:
        raise ValueError('single-pass inference supports 2-26 options')
    return dict(state=state, instructions=to_text(question.get('instructions', '')), type=kind, options=options)


def prompt_parts(example):
    opts = '\n'.join(f'{LETTERS[i]}. {d}' if d == k or example['type'] == 'noul'
                     else f'{LETTERS[i]}. {k}: {d}' for i, (k, d) in enumerate(example['options']))
    hint = ' (levels in ascending order)' if example['type'] == 'score' else ''
    return f"{example['state']}\n\n", f"{example['instructions']}{hint}\n{opts}\nAnswer:"


def token_ids(tokenizer, example):
    a, b = prompt_parts(example)
    return tokenizer.encode(a, add_special_tokens=False) + tokenizer.encode(b, add_special_tokens=False)


def letter_ids(tokenizer, count):
    ids = [tokenizer.encode(' ' + c, add_special_tokens=False) for c in LETTERS[:count]]
    if any(len(x) != 1 for x in ids):
        raise ValueError('tokenizer does not have single-token option letters')
    return [x[0] for x in ids]


def readout(logits, example, temps=None):
    """FP32 head logits are supplied by the backend; score mixing is fixed at zero."""
    temperature = (temps or DEFAULT_TEMPS)[example['type']]
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperatures must be positive and finite')
    # Match serve: softmax, clip, then per-type temperature power.
    peak = max(logits)
    p = [math.exp(float(v) - peak) for v in logits]
    total = sum(p)
    p = [max(1e-12, min(1., v / total)) ** (1 / temperature) for v in p]
    total = sum(p)
    p = [v / total for v in p]
    kind = example['type']
    keys = [k for k, _ in example['options']]
    if kind == 'noul':
        yes = p[0]
        if .2 < yes < .8:
            yes = .2 if yes < .5 else .8
        return {'answer': yes >= .5, 'probs': {'yes': yes, 'no': 1 - yes}}
    answer = keys[max(range(len(p)), key=p.__getitem__)] if kind == 'choice' else sum(i * v for i, v in enumerate(p))
    return {'answer': answer, 'probs': dict(zip(keys, p))}


class Xevn:
    def __init__(self, model='quevirus/Xevn-4B', dtype='bfloat16', device='cpu', threads=None):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model, dtype=getattr(torch, dtype), attn_implementation='eager').to(device).eval()
        path = Path(model) / 'temps.json'
        if not path.exists() and not Path(model).exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(model, 'temps.json'))
        calibration = json.loads(path.read_text()) if path.exists() else {'temps': DEFAULT_TEMPS}
        if calibration.get('score_lambda', 0) != 0:
            raise ValueError('this release requires score_lambda=0')
        self.temps = {**DEFAULT_TEMPS, **calibration.get('temps', {})}

    def logits(self, example):
        torch = self.torch
        ids = torch.tensor([token_ids(self.tokenizer, example)], device=self.device)
        letters = letter_ids(self.tokenizer, len(example['options']))
        with torch.inference_mode():
            hidden = self.model.model.language_model(input_ids=ids, use_cache=False).last_hidden_state[:, -1, :].float()
            # Select the tied head rows before promoting to FP32, avoiding a full vocab projection.
            head = self.model.lm_head.weight[letters].float()
            return torch.nn.functional.linear(hidden, head)[0].cpu().tolist()

    def predict(self, state, question):
        example = build(question, state)
        return readout(self.logits(example), example, self.temps)

    def request(self, request):
        if not isinstance(request.get('questions'), dict) or not request['questions']:
            raise ValueError('questions must be a non-empty object')
        return {key: self.predict(request['state'], q) for key, q in request['questions'].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='quevirus/Xevn-4B')
    ap.add_argument('--request', required=True, type=Path)
    ap.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    print(json.dumps(Xevn(args.model, args.dtype, args.device).request(json.loads(args.request.read_text())), ensure_ascii=False))


if __name__ == '__main__':
    main()
