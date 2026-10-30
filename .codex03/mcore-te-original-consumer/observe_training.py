"""Record actual loaded source identities without changing the original program."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ENTRY = sys.argv.pop(1)
sys.argv[0] = ENTRY

def record_loaded_origins():
    rows = []
    for name, module in sorted(list(sys.modules.items())):
        if not (name.startswith(('megatron', 'transformer_engine')) or name in ('gpt_builders', 'model_provider', 'mamba_builders', 'torch', 'torch_npu', 'numpy')):
            continue
        path = getattr(module, '__file__', None)
        row = {'module': name, 'file': path, 'package_paths': list(getattr(module, '__path__', []))}
        if path:
            try:
                p = Path(path).resolve()
                body = p.read_bytes()
                row.update(file=str(p), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            except Exception as error:
                row['read_error'] = repr(error)
        rows.append(row)
    entry = Path(ENTRY)
    result = {'pid': os.getpid(), 'rank': os.environ.get('RANK'), 'python': sys.version, 'interpreter': sys.executable, 'sys_path': sys.path, 'entry': ENTRY, 'entry_sha256': hashlib.sha256(entry.read_bytes()).hexdigest(), 'modules': rows}
    output = Path(os.environ['AP218_ORIGIN_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ('rank-' + str(os.environ.get('RANK', 'cpu')) + '-pid-' + str(os.getpid()) + '.json')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('AP218_ACTUAL_ORIGIN_RECORD=' + str(destination), flush=True)

atexit.register(record_loaded_origins)
runpy.run_path(ENTRY, run_name='__main__')
