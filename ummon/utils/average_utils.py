import numpy as np
import torch

def online_average(value, count, avg):
    # BACKWARD COMPATIBILITY FOR TORCH < 0.4
    if type(value) is not float and not isinstance(value, np.float):
        if type(value) == torch.Tensor:
            value = value.item()
        else:
            value = value.data[0]
    navg = avg + (value - avg) / count
    return navg
     
def moving_average(t, ma, value, buffer):
    """
    Helper method for computing moving averages.
    
    Arguments
    ---------
    * t (int) : The timestep
    * ma (float) : Current moving average
    * value (float) : Current value
    * buffer (List<float>) : The buffer of size N
    
    Return
    ------
    * moving_average (float) : The new computed moving average.
    
    """
    # BACKWARD COMPATIBILITY FOR TORCH < 0.4
    if type(value) is not float and not isinstance(value, np.float):
        if type(value) == torch.Tensor:
            value = value.item()
        else:
            value = value.data[0]
    
    n = buffer.shape[0]
    if ma is None:
        moving_average = value
        buffer += value
    else:
        moving_average = ma + (value / n) - (buffer[t % n] / n)
    buffer[t % n] = value
    return moving_average