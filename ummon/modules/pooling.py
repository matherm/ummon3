import numpy as np
import torch
import torch.nn as nn

__all__ = [ 'MaxPool', 'AvgPool', 'MaxPool1d' ]

# Max pooling layer class
class MaxPool(nn.MaxPool2d):
    '''
    Max pooling layer class::
    
        poo0 = MaxPool([n,p,q], kernel_size, stride, padding)
    
    creates a max pooling layer. This method selects the local maximum in the 
    neighborhood determined by 'kernel_size' as the output value. The error is sparsely 
    backpropagated, i.e., only the pixels where the maximum occurred is updated with the 
    backpropagated error, all others are ignored.
    
    Applies max pooling to subsample an input tensor and provides a certain translation 
    invariance. The input tensor for this layer must be in a non-flattened format. Pooling is 
    controlled by 3 attributes: the stride between window centers in x- and y-direction
    ('stride': either a number or 2-tuple), and the pooling window size ('kernel_size': 
    either a number or 2-tuple). You can set an additional padding region ('padding':
    either a number or 2-tuple) filled with zeroes for treating the image boundary 
    regions.
    
    You obtain the classical pooling in the valid region (as, e.g., in LeNet) by setting
    stride and kernel size to the same values and by setting 'padding' to zero. If you
    want a behaviour similar to a strided convolution with zero padding set 'padding'
    to half the filter size in each direction. For examples, see the tests in ummon_test.py.
    '''    
    # constructor
    def __init__(self, insize, kernel_size, stride, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        if type(kernel_size) == int:
            self._kernel_size = [kernel_size, kernel_size]
        else:
            self._kernel_size = list(kernel_size)
        for s in self._kernel_size:
            s = int(s)
            if s < 1:
                raise ValueError('Kernel size must be > 0.')
        if len(self._kernel_size) != 2:
            raise TypeError('Provided kernel size must have 1 or 2 elements.')
        self._kernel_size = tuple(self._kernel_size)
        if type(stride) == int:
            self._stride = [stride, stride]
        else:
            self._stride = list(stride)
        for s in self._stride:
            s = int(s)
            if s < 1:
                raise ValueError('Stride must be > 0.')
        if len(self._stride) != 2:
            raise TypeError('Provided stride must have 1 or 2 elements.')
        self._stride = tuple(self._stride)
        if type(padding) == int:
            self._padding = [padding, padding]
        else:
            self._padding = list(padding)
        for s in self._padding:
            s = int(s)
            if s < 0:
                raise ValueError('Padding must be >= 0.')
        if len(self._padding) != 2:
            raise TypeError('Provided padding must have 1 or 2 elements.')
        self._padding = tuple(self._padding)
        
        super(MaxPool, self).__init__(kernel_size, stride, padding, dilation, 
            return_indices, ceil_mode)
        
        # output size
        if self._stride[0] == self._kernel_size[0] and self._stride[1] == self._kernel_size[1]: # valid pooling
            out_height = (self.insize[1] + 2*self._padding[0])//self._stride[0]
            out_width = (self.insize[2] + 2*self._padding[1])//self._stride[1]
        else:
            out_height = (self.insize[1] + 2*self._padding[0] - 2*(self._kernel_size[0]//2) - 1) // self._stride[0] + 1
            out_width =  (self.insize[2] + 2*self._padding[1] - 2*(self._kernel_size[1]//2) - 1) // self._stride[1] + 1
        self.outsize = [self.insize[0], out_height, out_width]
        
        # network stats
        self.num_neurons = self.outsize[0] * self.outsize[1] * self.outsize[2]
        self.num_weights = self.num_adj_weights = 0
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{},{}]->[bs,{},{},{}];sz={};str={};pad={})'.format(
            self.insize[0], self.insize[1], self.insize[2], self.outsize[0], self.outsize[1], 
            self.outsize[2], self._kernel_size, self._stride, self._padding)
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        if self._stride[0] == self._kernel_size[0] and self._stride[1] == self._kernel_size[1]: # valid pooling
            y0 = outp[1]*self._stride[0]
            x0 = outp[2]*self._stride[1]
            y1 = outp[4]*self._stride[0] + self._kernel_size[0] - 1
            x1 = outp[5]*self._stride[1] + self._kernel_size[1] - 1
            
        else: 
            y0 = outp[1]*self._stride[0] - self._padding[0]
            if y0 < 0:
                y0 = 0
            x0 = outp[2]*self._stride[1] - self._padding[1]
            if x0 < 0: 
                x0 = 0
            y1 = outp[4]*self._stride[0] + self._kernel_size[0] - 1 - self._padding[0]
            if y1 >= self.insize[1]:
                y1 = self.insize[1] - 1
            x1 = outp[5]*self._stride[1] + self._kernel_size[1] - 1 - self._padding[1]
            if x1 >= self.insize[2]: 
                x1 = self.insize[2] - 1
        
        return [outp[0], y0, x0, outp[3], y1, x1]



# Max pooling layer class
class MaxPool1d(nn.MaxPool1d):
    '''
    1D Max pooling layer class::
    
        poo0 = MaxPool1d([n,p], kernel_size, stride, padding)
    
    creates a max pooling layer. This method selects the local maximum in the 
    neighborhood determined by 'kernel_size' as the output value. The error is sparsely 
    backpropagated, i.e., only the pixels where the maximum occurred is updated with the 
    backpropagated error, all others are ignored.
    
    Applies max pooling to subsample an input tensor and provides a certain translation 
    invariance. The input tensor for this layer must be in a non-flattened format. Pooling is 
    controlled by 3 attributes: the stride between window centers ('stride'), and the 
    pooling window size ('kernel_size'). You can set an additional padding region 
    ('padding') filled with zeroes for treating the image boundary regions.
    
    You obtain the classical pooling in the valid region (as, e.g., in LeNet) by setting
    stride and kernel size to the same values and by setting 'padding' to zero. If you
    want a behaviour similar to a strided convolution with zero padding set 'padding'
    to half the filter size in each direction. For examples, see the tests in ummon_test.py.
    '''    
    # constructor
    def __init__(self, insize, kernel_size, stride, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 2:
            raise TypeError('Input size list must have 2 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        self._kernel_size = int(kernel_size)
        if self._kernel_size < 1:
            raise ValueError('Kernel size must be > 0.')
        self._stride = int(stride)
        if self._stride < 1:
            raise ValueError('Stride must be > 0.')
        self._padding = int(padding)
        if self._padding < 0:
            raise ValueError('Padding must be >= 0.')
        
        super(MaxPool1d, self).__init__(kernel_size, stride, padding, dilation, 
            return_indices, ceil_mode)
        
        # output size
        if self._stride == self._kernel_size: # valid pooling
            out_width = (self.insize[1] + 2*self._padding)//self._stride
        else:
            out_width =  (self.insize[1] + 2*self._padding - 2*(self._kernel_size//2) - 1) // self._stride + 1
        self.outsize = [self.insize[0], out_width]
        
        # network stats
        self.num_neurons = self.outsize[0] * self.outsize[1]
        self.num_weights = self.num_adj_weights = 0
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{}]->[bs,{},{}];sz={};str={};pad={})'.format(
            self.insize[0], self.insize[1], self.outsize[0], self.outsize[1], 
            self._kernel_size, self._stride, self._padding)
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        if self._stride == self._kernel_size: # valid pooling
            x0 = outp[1]*self._stride
            x1 = outp[3]*self._stride + self._kernel_size - 1            
        else: 
            x0 = outp[1]*self._stride - self._padding
            if x0 < 0: 
                x0 = 0
            x1 = outp[3]*self._stride + self._kernel_size - 1 - self._padding
            if x1 >= self.insize[1]: 
                x1 = self.insize[1] - 1
        
        return [outp[0], x0, outp[2], x1]


# Average pooling layer class
class AvgPool(nn.AvgPool2d):
    '''
    Average pooling layer class::
    
        poo0 = AvgPool([n,p,q], kernel_size, stride, padding, count_include_pad)
    
    creates an average pooling layer. This method takes the average value in the pooling window 
    as the output value. 
    
    Applies average pooling to subsample an input tensor and provides a certain translation 
    invariance. The input tensor for this layer must be in a non-flattened format. Pooling is 
    controlled by 3 attributes: the stride between window centers in x- and y-direction
    ('stride': either a number or 2-tuple), and the pooling window size ('kernel_size': 
    either a number or 2-tuple). You can set an additional padding region ('padding':
    either a number or 2-tuple) filled with zeroes for treating the image boundary 
    regions. In this case, you should set 'count_include_pad' either to True or False,
    depending on whether you want to include the padded region in the average or not.
    
    
    You obtain the classical pooling in the valid region (as, e.g., in LeNet) by setting
    stride and kernel size to the same values and by setting 'padding' to zero. If you
    want a behaviour similar to a strided convolution with zero padding set 'padding'
    to half the filter size in each direction. For examples, see the tests in ummon_test.py.
    '''    
    # constructor
    def __init__(self, insize, kernel_size, stride, padding=0, count_include_pad=False):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        if type(kernel_size) == int:
            self._kernel_size = [kernel_size, kernel_size]
        else:
            self._kernel_size = list(kernel_size)
        for s in self._kernel_size:
            s = int(s)
            if s < 1:
                raise ValueError('Kernel size must be > 0.')
        if len(self._kernel_size) != 2:
            raise TypeError('Provided kernel size must have 1 or 2 elements.')
        self._kernel_size = tuple(self._kernel_size)
        if type(stride) == int:
            self._stride = [stride, stride]
        else:
            self._stride = list(stride)
        for s in self._stride:
            s = int(s)
            if s < 1:
                raise ValueError('Stride must be > 0.')
        if len(self._stride) != 2:
            raise TypeError('Provided stride must have 1 or 2 elements.')
        self._stride = tuple(self._stride)
        if type(padding) == int:
            self._padding = [padding, padding]
        else:
            self._padding = list(padding)
        for s in self._padding:
            s = int(s)
            if s < 0:
                raise ValueError('Padding must be >= 0.')
        if len(self._padding) != 2:
            raise TypeError('Provided padding must have 1 or 2 elements.')
        self._padding = tuple(self._padding)
        
        super(AvgPool, self).__init__(kernel_size, stride, padding, count_include_pad)
        
        # output size
        if self._stride[0] == self._kernel_size[0] and self._stride[1] == self._kernel_size[1]: # valid pooling
            out_height = (self.insize[1] + 2*self._padding[0])//self._stride[0]
            out_width = (self.insize[2] + 2*self._padding[1])//self._stride[1]
        else:
            out_height = (self.insize[1] + 2*self._padding[0] - 2*(self._kernel_size[0]//2) - 1) // self._stride[0] + 1
            out_width =  (self.insize[2] + 2*self._padding[1] - 2*(self._kernel_size[1]//2) - 1) // self._stride[1] + 1
        self.outsize = [self.insize[0], out_height, out_width]
        
        # network stats
        self.num_neurons = self.outsize[0] * self.outsize[1] * self.outsize[2]
        self.num_weights = self.num_adj_weights = 0
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{},{}]->[bs,{},{},{}];sz={};str={};pad={})'.format(
            self.insize[0], self.insize[1], self.insize[2], self.outsize[0], self.outsize[1], 
            self.outsize[2], self._kernel_size, self._stride, self._padding)
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        if self._stride[0] == self._kernel_size[0] and self._stride[1] == self._kernel_size[1]: # valid pooling
            y0 = outp[1]*self._stride[0]
            x0 = outp[2]*self._stride[1]
            y1 = outp[4]*self._stride[0] + self._kernel_size[0] - 1
            x1 = outp[5]*self._stride[1] + self._kernel_size[1] - 1
            
        else: 
            y0 = outp[1]*self._stride[0] - self._padding[0]
            if y0 < 0:
                y0 = 0
            x0 = outp[2]*self._stride[1] - self._padding[1]
            if x0 < 0: 
                x0 = 0
            y1 = outp[4]*self._stride[0] + self._kernel_size[0] - 1 - self._padding[0]
            if y1 >= self.insize[1]:
                y1 = self.insize[1] - 1
            x1 = outp[5]*self._stride[1] + self._kernel_size[1] - 1 - self._padding[1]
            if x1 >= self.insize[2]: 
                x1 = self.insize[2] - 1
        
        return [outp[0], y0, x0, outp[3], y1, x1]


# Unpooling layer class
# class Unpool(Layer):
#     '''
#     Unpooling layer class::
#     
#         unp = Unpool(id, [n,p,q], cnet,['name 1',..], padding, ystride, xstride,
#             ysize, xsize)
#     
#     creates an upooling layer with ID string 'id' for an input tensor of size [n,p,q]. The
#     layer is registered in the network 'cnet' and connected to the input layers 'name 1', 
#     'name 2 ' etc. 
#     
#     Applies unpooling to upscale an input tensor. The input tensor for this node can have 
#     an arbitrary size. Unpooling is controlled by 4 attributes: the stride between window 
#     centers in x- and y-direction ('ystride' and 'xstride'), and the pooling window size 
#     ('ysize' and 'xsize'). If the input dimensions are [mini_batch_size, nmaps, rows, cols], 
#     the output dimensions are [mini_batch_size, nmaps, r, c] with r = rows*ystride and 
#     c = cols*xstride. The unpooling operation corresponds roughly to an inversion of
#     average pooling.
#     
#     You can set two types of padding for treating the enlarged pixel regions:
#     
#     1. 'valid': the input pixels are just enlarged by the stride factors (zero order
#     interpolation), so the pooling window size arguments are ignored. 
#     
#     2. 'same': the windows corresponding to the enlarged pixels are allowed to extend 
#     beyond the image boundaries and to overlap. The first window center is placed at 
#     position (0,0), all others are placed according to xstride and ystride. The window 
#     size must be odd. 
#     
#     Attributes:
#     
#     * padding (read only): padding type of the layer
#     * ystride: stride in y-direction (read only)
#     * xstride: stride in x-direction (read only)
#     * ysize: window size in y-direction (read only)
#     * xsize: window size in x-direction (read only)
#     
#     '''
#     # padding types
#     padding_types = [ 'valid', 'same' ]
#     
#     # constructor
#     def __init__(self, id, insize, netw, inputs=[], padding='valid', ystride=3, xstride=3, 
#         ysize=3, xsize=3):
#         super(Unpool, self).__init__(id, insize, netw, inputs)
#         
#         # check layer parameters
#         if len(insize) != 3:
#             _lg.error('Input size list must have 3 elements.', TypeError)
#         ystride = int(ystride)
#         if padding not in self.padding_types:
#             _lg.error('Unknown padding type: {}'.format(padding))
#         self._padding = padding
#         if ystride < 2:
#             _lg.error('ystride must be >= 2.', ValueError)
#         self._ystride = ystride
#         xstride = int(xstride)
#         if xstride < 2:
#             _lg.error('xstride must be >= 2.', ValueError)
#         self._xstride = xstride
#         ysize = int(ysize)
#         if ysize < 2:
#             _lg.error('ysize must be >= 2.', ValueError)
#         self._ysize = ysize
#         xsize = int(xsize)
#         if xsize < 2:
#             _lg.error('xsize must be >= 2.', ValueError)
#         self._xsize = xsize
#         if padding == 'same' and (not ysize % 2 or not xsize % 2):
#             _lg.error('Window size must be odd for SAME padding.', ValueError)
#         
#         # output size
#         out_height = self.insize[2] * self.ystride
#         out_width = self.insize[3] * self.xstride
#         self.outsize = [netw.mini_batch_size, self.insize[1], out_height, out_width]
#         
#         # add layer to network
#         self.num_neurons = self.insize[1] * self.insize[2] * self.insize[3]
#         netw.register_node(self)
#         _lg.info("Unpooling layer     ({}->{},{},dy={},dx={}) '{}'".format(self.insize, 
#             self.outsize, self.padding, self.ystride, self.xstride, self.id))
#     
#     
#     # Symbolically defined forward path
#     def forward(self, input, **kwargs):
#         '''
#         Computes symbolically the unpooling operation. Called by Network._compile().
#         '''
#         upscaled = input
#         if self.padding == 'valid':
#             if self.xstride > 1:
#                 upscaled = T.extra_ops.repeat(upscaled, self.xstride, 3)
#             if self.ystride > 1:
#                 upscaled = T.extra_ops.repeat(upscaled, self.ystride, 2)
#         else:
#             if self.ystride > 1 or self.xstride > 1:
#                 W = T.zeros(shape=tuple([self.insize[1],self.insize[1],self.ysize,self.xsize]), 
#                     dtype=input.dtype)
#                 for i in range(self.insize[1]):
#                     W = T.set_subtensor(W[i,i,:,:], T.ones(shape=tuple([self.ysize,self.xsize])))
#                 upscaled = T.zeros(shape=tuple(self.outsize), dtype=input.dtype)
#                 upscaled = T.set_subtensor(upscaled[:,:,::self.ystride,::self.xstride], 
#                     input)
#                 upscaled = T.nnet.conv2d(upscaled, W, tuple(self.outsize), 
#                     tuple([self.outsize[1],self.insize[1],self.ysize, self.xsize]), "half", (1,1), False)
#         
#         upscaled = (1.0/(self.xstride*self.ystride)) * upscaled
#         return upscaled
#     
#     
#     # Get the input block of a given output block
#     def get_input_block(self, outp):
#         '''
#         Returns the input block that feeds into the specified output block.
#         '''
#         y0 = outp[1]//self.ystride
#         x0 = outp[2]//self.xstride
#         y1 = (outp[4] + 1 - self.ysize)//self.ystride
#         x1 = (outp[5] + 1 - self.xsize)//self.xstride
#         
#         return [outp[0], y0, x0, outp[3], y1, x1]
#     
#     
#     # Attribute: padding type (read only)
#     @property
#     def padding(self):
#         return self._padding
#     
#     # Attribute: stride in y-direction (read only)
#     @property
#     def ystride(self):
#         return self._ystride
#     
#     # Attribute: stride in x-direction (read only)
#     @property
#     def xstride(self):
#         return self._xstride
#     
#     # Attribute: window size in y-direction (read only)
#     @property
#     def ysize(self):
#         return self._ysize
#     
#     # Attribute: window size in x-direction (read only)
#     @property
#     def xsize(self):
#         return self._xsize
# 
