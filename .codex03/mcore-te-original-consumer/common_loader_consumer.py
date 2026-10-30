"""Consume the genuine retained checkpoint through normal MCore imports/API."""
import argparse
import datetime
import enum
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    checkpoint = Path(args.checkpoint).resolve()
    observed = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                for p in checkpoint.iterdir() if p.is_file()}
    expected = json.loads(Path(args.expected).read_text())['common']
    sys.path.insert(0, str(Path(args.source).resolve()))
    import torch
    import megatron.core
    from megatron.core.dist_checkpointing import serialization
    value = serialization.load_common_state_dict(str(checkpoint))
    assert sorted(value) == sorted(expected['common_keys'])
    comparisons = {}
    for key, reference in expected['selected_args'].items():
        actual = getattr(value['args'], key)
        if isinstance(reference, dict) and 'type' in reference:
            assert isinstance(actual, enum.Enum), (key, type(actual))
            actual = {'type': type(actual).__module__ + '.' + type(actual).__name__,
                      'name': actual.name, 'value': actual.value}
        assert actual == reference, (key, actual, reference)
        comparisons['args.' + key] = actual
    for key, reference in expected['common'].items():
        assert value[key] == reference, (key, value[key], reference)
        comparisons[key] = value[key]
    modules = {}
    for name in ['megatron.core', 'megatron.core.dist_checkpointing.serialization',
                 'megatron.core.dist_checkpointing.strategies.torch',
                 'megatron.core.enums', 'megatron.core.transformer.enums']:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        modules[name] = {'path': str(path), 'sha256': digest(path)}
    after = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in checkpoint.iterdir() if p.is_file()}
    assert observed == after, 'Original read-only inputs changed'
    assert not torch.cuda.is_initialized()
    result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'result': 'NORMAL_MCORE_COMMON_LOADER_PASS',
              'seconds': time.monotonic() - start,
              'python': sys.version, 'torch': torch.__version__,
              'cuda_initialized': torch.cuda.is_initialized(),
              'call': 'serialization.load_common_state_dict',
              'input_bodies_before_after_equal': True, 'inputs': observed,
              'expected_binding_sha256': digest(args.expected),
              'modules': modules, 'comparisons': comparisons,
              'limits': 'Common metadata only; no model/optimizer tensors, RNG restore, '
                        'TE operation, GPU/NPU, full continuation or whole acceptance.'}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
