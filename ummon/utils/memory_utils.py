import os, psutil, subprocess
import numpy as np
import time
import torch
from torch.autograd import Variable


def get_proc_memory_info():
    try:
        process = psutil.Process(os.getpid())
        percentage = process.memory_percent()
        memory = process.memory_info()[0] / float(2 ** 30)
        return {"mem" : memory,
              "usage" : percentage}
    except Exception:
        return 0.
  
def get_cuda_memory_info():
    """
    Get the current gpu usage.
    
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    try:
        if torch.cuda.is_available() == False:
            return 0.
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ]).decode('utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map
    except Exception:
        return 0.
    


