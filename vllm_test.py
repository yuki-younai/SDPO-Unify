#from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.engine.llm_engine import LLMEngine

#import flash_attn_2_cuda as flash_attn_gpu
#测试torch cuda·
import torch
print(torch.cuda.is_available())


