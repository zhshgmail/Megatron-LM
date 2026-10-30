"""Build unchanged Megatron dataset helper with the selected runtime Python ABI."""
import argparse,datetime,hashlib,importlib.metadata,importlib.util,json,os,platform,shutil,subprocess,sys,sysconfig
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--expected-python',default='3.12');p.add_argument('--expected-machine',default='aarch64');p.add_argument('--build',action='store_true');a=p.parse_args()
source=Path(a.source).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);ds=source/'megatron/core/datasets'
def body(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
expected={'helpers.cpp':None,'Makefile':None}
# These expected hashes are supplied by this packet's SOURCE_FILES.json and must match.
pins=json.loads((Path(__file__).parent/'SOURCE_FILES.json').read_text())
for name in expected:
 actual=body(ds/name);assert actual['sha256']==pins[name]['sha256'],(name,actual,pins[name])
include=Path(sysconfig.get_path('include'));pybind=None
try:
 import pybind11
 pybind=Path(pybind11.get_include())
except ImportError:
 # Genuine existing Torch headers may supply pybind11 without importing Torch or initializing devices.
 try:
  candidate=Path(importlib.metadata.distribution('torch').locate_file('torch/include'))
  if (candidate/'pybind11/pybind11.h').is_file():pybind=candidate
 except importlib.metadata.PackageNotFoundError:pass
observed={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,'machine':platform.machine(),'SOABI':sysconfig.get_config_var('SOABI'),'EXT_SUFFIX':sysconfig.get_config_var('EXT_SUFFIX'),'Python_h':body(include/'Python.h') if (include/'Python.h').is_file() else None,'pybind_include':str(pybind) if pybind else None,'cxx':shutil.which('g++'),'make':shutil.which('make'),'source_files':{n:body(ds/n) for n in expected},'source_git':subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],text=True,capture_output=True).stdout.strip(),'build_requested':a.build,'imports_torch_or_torch_npu':False,'device_initialization':False}
for package in ['torch','torch-npu','transformer-engine','numpy','pybind11']:
 try:observed[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError:observed[package]=None
missing=[]
if platform.machine()!=a.expected_machine:missing.append('machine does not match selected target')
if '.'.join(map(str,sys.version_info[:2]))!=a.expected_python:missing.append('Python minor ABI does not match selected target')
if not observed['Python_h']:missing.append('selected Python development header Python.h')
if pybind is None:missing.append('existing pybind11 headers')
if not observed['cxx']:missing.append('g++')
if not observed['make']:missing.append('make')
observed['missing']=missing
(out/'PREFLIGHT.json').write_text(json.dumps(observed,indent=2)+'\n');print(json.dumps(observed,indent=2))
if missing:sys.exit(2)
if not a.build:sys.exit(0)
# Build in isolated owned output; no source edits and no stale wheel binary copied.
for name in expected:shutil.copyfile(ds/name,out/name)
cmd=['make','-C',str(out),'CPPFLAGS=-I'+str(include)+' -I'+str(pybind),'LIBEXT='+sysconfig.get_config_var('EXT_SUFFIX')]
r=subprocess.run(cmd);assert r.returncode==0,r.returncode
binary=out/('helpers_cpp'+sysconfig.get_config_var('EXT_SUFFIX'))
spec=importlib.util.spec_from_file_location('helpers_cpp',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert hasattr(module,'build_sample_idx_int32') and hasattr(module,'build_sample_idx_int64')
observed.update(build_argv=cmd,binary=body(binary),module_import=True,placement='Root must place the exact verified binary beside authoritative megatron/core/datasets/helpers.py in its owned target source/site before D/E. This script does not mutate source/site.')
(out/'BUILD_RESULT.json').write_text(json.dumps(observed,indent=2)+'\n');print(json.dumps({'binary':observed['binary'],'module_import':True}))
