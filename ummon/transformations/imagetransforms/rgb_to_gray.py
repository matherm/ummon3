import torch


class GraytoRGB():
    
    
    r"""
    Small transformation to transform a single gray channel im to a rgb-like 3 channel image.
    It simply copies the rgb image to all three layers.

    Usage:
        trans = GraytoRGB()
        im = trans(torch.zeros(1,224,224)) # shape (3, 224, 224)

    Args:
        im (Tensor): The image with shape :obj:`[1, width, height]`.

    """
    def __call__(self, im):
        assert im.dim() == 3
        assert im.shape[0] == 1
        return im.repeat(3, 1, 1)


class RGBtoGray():
    r"""
    Small transformation to transform a single rgb image to gray scale.
    It simply copies the rgb image to all three layers.
    0.299*R + 0.587*G + 0.114*B


    Usage:
        trans = RGBtoGray()
        im = trans(torch.zeros(1,224,224)) # shape (3, 224, 224)
        
    Args:
        im (Tensor): The image with shape :obj:`[1, width, height]`.

    """
    def __call__(self, im):
        assert im.dim() == 3
        assert im.shape[0] == 3
        return torch.stack([0.299 * im[0], 0.587 * im[1], 0.114 * im[2]]).sum(dim=0, keepdim=True)