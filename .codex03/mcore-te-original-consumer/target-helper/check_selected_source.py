"""Read-only no-Torch-import selected package/ABI/body gate before original D/E."""
import argparse,datetime,hashlib,importlib.machinery,json,sys,sysconfig
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--helper-receipt',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.source).resolve()
def spec(n,path):
 s=importlib.machinery.PathFinder.find_spec(n,path);assert s is not None,n;return s
m=spec('megatron',sys.path);c=spec('megatron.core',list(m.submodule_search_locations));ds=spec('megatron.core.datasets',list(c.submodule_search_locations));h=spec('megatron.core.datasets.helpers_cpp',list(ds.submodule_search_locations))
assert Path(c.origin).resolve()==root/'megatron/core/__init__.py',('wrong_source',c.origin)
assert Path(h.origin).resolve().parent==root/'megatron/core/datasets',('wrong_helper_path',h.origin)
assert str(h.origin).endswith(sysconfig.get_config_var('EXT_SUFFIX')),('wrong_helper_ABI',h.origin)
pins=json.loads((Path(__file__).parent/'SELECTED_SOURCE_FILES.json').read_text());observed={}
for name,expected in pins.items():
 b=(root/name).read_bytes();digest=hashlib.sha256(b).hexdigest();assert digest==expected['sha256'],(name,digest,expected);observed[name]={'bytes':len(b),'sha256':digest}
receipt=json.loads(Path(a.helper_receipt).read_text());binary=Path(h.origin);actual=hashlib.sha256(binary.read_bytes()).hexdigest();assert actual==receipt['binary']['sha256'],('helper_copy_mismatch',actual)
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_origin':c.origin,'helper_origin':h.origin,'helper_sha256':actual,'SOABI':sysconfig.get_config_var('SOABI'),'source_files':observed,'no_torch_or_torch_npu_import':True,'device_initialization':False};Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
