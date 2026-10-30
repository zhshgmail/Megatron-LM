"""Normal GPTDataset/native helper + real original WikiText103 input CPU consumer."""
import argparse,hashlib,json,sys,time,datetime,itertools,subprocess,gc
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic()
sys.path.insert(0,str(Path(a.source).resolve()))
import numpy as np
import torch
torch.set_num_threads(2)
from megatron.core.dist_checkpointing import serialization
from megatron.core.datasets.gpt_dataset import GPTDataset,GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.datasets import helpers_cpp
from megatron.training.tokenizer.tokenizer import build_tokenizer
from megatron.training.datasets.data_samplers import MegatronPretrainingSampler
saved=serialization.load_common_state_dict(a.checkpoint);args=saved['args'];original_paths={'vocab_file':args.vocab_file,'merge_file':args.merge_file}
args.vocab_file=str(Path(a.data)/'tokenizer/vocab.json');args.merge_file=str(Path(a.data)/'tokenizer/merges.txt');tokenizer=build_tokenizer(args)
assert args.seq_length==1024 and args.micro_batch_size==4 and args.global_batch_size==32 and args.train_iters==100
sizes=[args.train_iters*args.global_batch_size,(args.train_iters//args.eval_interval+1)*args.eval_iters*args.global_batch_size,args.eval_iters*args.global_batch_size]
assert sizes==[3200,320,320]
config_values={n:getattr(args,n) for n in ['reset_position_ids','reset_attention_mask','eod_mask_loss']}
config_values.update(random_seed=args.seed,sequence_length=args.seq_length,create_attention_mask=args.create_attention_mask_in_dataloader,mmap_bin_files=args.mmap_bin_files,num_dataset_builder_threads=args.num_dataset_builder_threads)
config=GPTDatasetConfig(**config_values,tokenizer=tokenizer,path_to_cache=a.cache,blend_per_split=[([str(Path(a.data)/'indexed'/f'{n}_text_document')],None) for n in ['train','validation','test']])
assert not config.mock
records=[];resume_comparison=None
for i,(name,split,count) in enumerate(zip(['train','validation','test'],list(Split),sizes)):
 prefix=str(Path(a.data)/'indexed'/f'{name}_text_document'); low=GPTDataset.build_low_level_dataset(prefix,config)
 ds=GPTDataset(low,prefix,np.arange(len(low),dtype=np.int32),count,split,config)
 def consume(offset,n):
  sampler=MegatronPretrainingSampler(len(ds),offset,args.micro_batch_size,0,1)
  seed=args.dataloader_worker_seed+split.value
  loader=torch.utils.data.DataLoader(ds,batch_sampler=sampler,num_workers=args.num_workers,pin_memory=True,persistent_workers=False,generator=torch.Generator(device='cpu').manual_seed(seed))
  iterator=iter(loader); hashes=[];seen=0;cpu_before=torch.get_rng_state().clone()
  for batch in itertools.islice(iterator,n//args.micro_batch_size):
   assert batch['tokens'].shape==(4,1024) and batch['labels'].shape==(4,1024)
   assert torch.equal(batch['tokens'][:,1:],batch['labels'][:,:-1])
   assert int(batch['tokens'].min())>=0 and int(batch['tokens'].max())<tokenizer.vocab_size
   h=hashlib.sha256()
   for key in sorted(batch):
    x=batch[key].contiguous();h.update(key.encode());h.update(str(x.dtype).encode());h.update(x.numpy().tobytes())
   hashes.append(h.hexdigest());seen+=len(batch['tokens'])
  del iterator,loader;gc.collect();assert seen==n
  assert torch.equal(torch.get_rng_state(),cpu_before)
  return hashes
 full=consume(0,count)
 if i==0:
  resumed=consume(1600,1600);assert resumed==full[400:]
  wrong=next(iter(MegatronPretrainingSampler(len(ds),1596,4,0,1)))
  assert wrong!=next(iter(MegatronPretrainingSampler(len(ds),1600,4,0,1)))
  # Actual wrong-offset dataset bodies must discriminate, not only the integer cursor.
  def body_hash(k):
   h=hashlib.sha256()
   for key,value in sorted(ds[k].items()):h.update(key.encode());h.update(value.numpy().tobytes())
   return h.hexdigest()
  wrong_body=body_hash(1596);expected_body=body_hash(1600);assert wrong_body!=expected_body
  resume_comparison={'checkpoint_consumed_train_samples':saved['args'].consumed_train_samples,'expected_cursor':1600,'uninterrupted_second_half_equal':True,'full_batches':len(full),'resume_batches':len(resumed),'wrong_offset':1596,'wrong_offset_body_sha':wrong_body,'expected_body_sha':expected_body,'optimizer_or_rng_restore_executed':False}
 records.append({'split':name,'requested':count,'dataset_length':len(ds),'low_level_rows':len(low),'batches':len(full),'all_fields_digest':hashlib.sha256(''.join(full).encode()).hexdigest(),'first_batch_sha':full[0],'last_batch_sha':full[-1]})
assert not torch.cuda.is_initialized()
helper=Path(helpers_cpp.__file__)
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-start,'source_commit':subprocess.check_output(['git','-C',a.source,'rev-parse','HEAD'],text=True).strip(),'normal_APIs':['GPTDataset.build_low_level_dataset','GPTDataset','MegatronPretrainingSampler','torch.utils.data.DataLoader','build_tokenizer'],'runtime':{'python':sys.version,'torch':torch.__version__},'helper':{'path':str(helper),'bytes':helper.stat().st_size,'sha256':hashlib.sha256(helper.read_bytes()).hexdigest()},'config':config_values,'sample_counts':sizes,'num_workers':args.num_workers,'worker_seed':args.dataloader_worker_seed,'micro_batch':args.micro_batch_size,'global_batch':args.global_batch_size,'cpu_DP':1,'original_target_TP':4,'results':records,'cursor_resume':resume_comparison,'cuda_initialized':torch.cuda.is_initialized(),'limits':'Real input/data interface only. CPU DP1 sampler with original original TP4 data-parallel size1. No model, TE, optimizer, device RNG restoration, gradients or NPU execution. R1 restore branch remains NOT_RUN.'}
Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
