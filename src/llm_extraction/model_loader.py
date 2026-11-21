"""Load the pinned instruction model on a CUDA GPU (SLURM/Kaggle) or Apple-Silicon MPS (local)."""
import time
from .runtime import require_compute_runtime, accelerator


class QwenExtractor:
    def __init__(self,config):
        self.backend=require_compute_runtime()
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
        self.device_type=accelerator(torch)
        self.torch=torch;self.config=config
        torch.set_num_threads(4);set_seed(config['seed'])
        self.tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
        if self.device_type=='cuda':
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype=torch.float16  # Apple MPS
        options=dict(torch_dtype=dtype,low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True)
        # Kaggle T4 x2 exposes separate devices, not a single combined memory pool.
        # Accelerate partitions the unchanged full-precision model across them.
        if self.device_type=='cuda' and self.backend=='kaggle' and torch.cuda.device_count()>1:
            options['device_map']='balanced'
            options['max_memory']={i:str(max(1,int(torch.cuda.mem_get_info(i)[0]/1024**3)-2))+'GiB'
                                   for i in range(torch.cuda.device_count())}
            self.model=AutoModelForCausalLM.from_pretrained(config['model'],revision=config['revision'],**options).eval()
            self.devices=list(range(torch.cuda.device_count()))
        else:
            self.model=AutoModelForCausalLM.from_pretrained(config['model'],revision=config['revision'],**options).to(self.device_type).eval()
            self.devices=[0] if self.device_type=='cuda' else []
        self.input_device=self.model.get_input_embeddings().weight.device
        self.dtype=str(next(self.model.parameters()).dtype)

    def device_names(self):
        if self.device_type=='cuda':return [self.torch.cuda.get_device_name(i) for i in self.devices]
        return ['Apple MPS']

    def empty_cache(self):
        if self.device_type=='cuda':self.torch.cuda.empty_cache()
        else:self.torch.mps.empty_cache()

    def generate(self,messages):
        torch=self.torch
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(text,return_tensors='pt',truncation=False).to(self.input_device)
        length=inputs['input_ids'].shape[1]
        if length+self.config['max_new_tokens']>self.config['max_context_tokens']:
            raise ValueError('Prompt exceeds frozen context budget; no truncation')
        for device in self.devices:
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=self.config['max_new_tokens'],do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        for device in self.devices:torch.cuda.synchronize(device)
        if self.device_type=='mps':torch.mps.synchronize()
        generated=output[0,length:]
        if self.device_type=='cuda':peaks={str(i):torch.cuda.max_memory_allocated(i) for i in self.devices}
        else:peaks={'mps':torch.mps.driver_allocated_memory()}
        return dict(response=self.tokenizer.decode(generated,skip_special_tokens=True),
                    input_tokens=length,output_tokens=len(generated),latency_seconds=time.perf_counter()-started,
                    reached_token_limit=len(generated)>=self.config['max_new_tokens'],
                    execution_backend=self.backend,gpu=self.device_names(),dtype=self.dtype,
                    per_device_peak_gpu_bytes=peaks,peak_gpu_bytes=sum(peaks.values()),
                    peak_memory_definition='Sum of per-device allocated-memory peaks (CUDA) or MPS driver memory')
