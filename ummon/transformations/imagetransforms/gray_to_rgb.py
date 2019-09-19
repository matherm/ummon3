
class To3Channel():
    r"""
    Small transformation to transform a single gray channel im to a rgb-like 3 channel image.
    It simply copies the rgb image to all three layers.

    Usage:
        trans = To3Channel()
        im = trans(torch.zeros(1,224,224)) # shape (3, 224, 224)

    Args:
        im (Tensor): The image with shape :obj:`[1, width, height]`.

    """
    def __call__(self, im):
        assert im.dim() == 3
        assert im.shape[0] == 1
        return im.repeat(3, 1, 1)