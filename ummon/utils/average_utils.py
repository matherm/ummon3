import numpy as np
import torch


__all__ = ['online_average', 'OnlineAverage', 'moving_average']

def online_average(value, count, avg):
    # BACKWARD COMPATIBILITY FOR TORCH < 0.4
    if type(value) is not float and not isinstance(value, np.float):
        if type(value) == torch.Tensor:
            value = value.item()
        else:
            value = value.data[0]
    navg = avg + (value - avg) / count
    return navg


class OnlineAverage():
    """docstring for OnlineAverage

    This online average function computes a running average.
    It can handle a single value or a list. 
    The list can also be a subpart of the complete list which should be averaged.    

    """

    def __init__(self):
        super().__init__()
        self.n_ = 0.0

    def __call__(self, value, avg):
        if type(value) is torch.Tensor:
            mvalue = torch.mean(value).item()
            try:
                items = len(value)
            except TypeError as e:
                items = 1
            except Exception as e:
                raise e
        elif type(value) is list or type(value) is np.ndarray:
            mvalue = np.mean(value)
            items = len(value)
        else:
            mvalue = value
            items = 1
        self.n_ += items
        return avg + (mvalue - avg) * items / self.n_

    def reset(self):
        self.n_ = 0


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
