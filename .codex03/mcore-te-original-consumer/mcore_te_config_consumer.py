"""Normal original configuration API, using actual saved full-root arguments."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
start = time.monotonic()
sys.path.insert(0, str(Path(a.source).resolve()))
import torch
from megatron.core.dist_checkpointing import serialization
from megatron.training.arguments import core_transformer_config_from_args

saved = serialization.load_common_state_dict(a.checkpoint)
original_args = saved['args']
config = core_transformer_config_from_args(original_args)
expected = {'num_layers': 12, 'hidden_size': 512, 'num_attention_heads': 8,
            'tensor_model_parallel_size': 4, 'pipeline_model_parallel_size': 1,
            'bf16': True, 'hidden_dropout': 0.1, 'attention_dropout': 0.1}
observed = {}
for field, value in expected.items():
    observed[field] = getattr(config, field)
    assert observed[field] == value, (field, observed[field], value)
assert not torch.cuda.is_initialized()
result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'seconds': time.monotonic() - start,
          'source_code_cohort': '2fc25bc987f32eff9c3afbf5f3f17cb3d9a2753b',
          'original_api': 'core_transformer_config_from_args(saved_original_args)',
          'config_type': type(config).__module__ + '.' + type(config).__name__,
          'observed': observed, 'pipeline_dtype': str(config.pipeline_dtype),
          'params_dtype': str(config.params_dtype),
          'cuda_initialized': torch.cuda.is_initialized(),
          'runtime': {'python': sys.version, 'torch': torch.__version__},
          'input_common_sha256': hashlib.sha256((Path(a.checkpoint) / 'common.pt').read_bytes()).hexdigest(),
          'limits': 'Configuration only; no TE layer/model construction or forward, optimizer, device or whole-software qualification.'}
Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
