import numpy as np
import torch
import torch.nn as nn


#__all__ = [ 'Conv', 'Deconv' ]
__all__ = [ 'Conv', 'Conv1d' ]


# Convolutional layer class
class Conv(nn.Conv2d):
    '''
    Convolutional layer::
    
        conv0 = Conv([m,n,p], [c,r,s], [ystride,xstride], [ypadding, xpadding], bias, init)
    
    creates a convolutional layer for unflattened 4d input tensors. The weight matrix is 
    initialized according to 'init' which must be one of the initialization methods 
    described in pyTorch.
    
    This layer implements a convolutional layer with a set of 'c' learned filters that
    are applied to a sequence of m-channel images. The output of each filter is
    the cross-correlation of the learned m x r x s filter mask with the multi channel
    image where 'm' is the number of input channels, 'r' the filter mask height and 's' 
    the filter mask width. 
    
    The output image can be downscaled by a factor of 'ystride' in y- and 'xstride' in 
    x-direction. In this way, a strided convolution is possible. This is useful for large 
    input images. You can use a single number if the strides in both directions is equal.
    
    Additional boundary pixels with value 0 are allocated when 'padding' is set to value 
    larger than 0. Note that in this case the same number of rows and columns are added, independently
    of the filter size. Alternatively, you can two different paddings in y- and x-direction
    by setting 'padding' to 2-tupke. If you set 'padding' to zero you obtain the classical
    'valid' padding used, e.g. in LeNet which leads to smaller output images. 
    By setting 'padding' to half the filter size you
    get the 'half' padding used in Theano. This corresponds to standard zero padding as
    used in image processing and preserves the image size.
    
    In addition, each filter also has a trainable bias term similar to standard linear 
    layers. If you do not want a bias term set the argument 'bias' to False.
    
    The layer accepts only input tensors of size [mini_batch_size, m, n, p]. If the input 
    is in flattened format, an Unflatten node 
    has to be inserted as input layer. 
    
    Attributes:
    
    * w: weight tensor of the layer
    * b: bias vector
    
    '''
    # constructor
    def __init__(self, insize, fsize, stride=1, padding=0, dilation=1, groups=1, 
        bias=True, init='xavier_normal_'):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        self.filtsize = list(fsize)
        if len(self.filtsize) != 3:
            raise TypeError('Filter size list must have 3 elements.')
        for s in self.filtsize:
            s = int(s)
            if s < 1:
                raise ValueError('Filter size must be > 0.')
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
        
        # create pytorch conv2d
        super(Conv, self).__init__(self.insize[0], self.filtsize[0], (self.filtsize[1], 
            self.filtsize[2]), self._stride, self._padding, dilation, groups, bias)
        
        # initialize weights
        init = str(init)
        nn.init.__dict__[init](self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias,0.0)
        
        # output size
        out_height = (self.insize[1] + 2*self._padding[0] - self.filtsize[1]) // \
            self._stride[0] + 1
        out_width = (self.insize[2] + 2*self._padding[1] - self.filtsize[2]) // \
            self._stride[1] + 1
        self.outsize = [self.filtsize[0], out_height, out_width]
        
        # network stats
        self.num_neurons = self.outsize[0] * self.outsize[1] * self.outsize[2]
        self.num_weights = self.num_neurons*self.filtsize[1]*self.filtsize[2]*self.insize[0] 
        self.num_adj_weights = self.filtsize[0]*self.insize[0]*self.filtsize[1]* \
            self.filtsize[2]
        if self.bias is not None:
            self.num_weights += self.num_neurons
            self.num_adj_weights += self.filtsize[0]
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{},{}]->[bs,{},{},{}],mask[{},{},{},{}],str=({},{}),pad={}'.format(
            self.insize[0], self.insize[1], self.insize[2], self.outsize[0], self.outsize[1], 
            self.outsize[2], self.filtsize[0], self.insize[0], self.filtsize[1], 
            self.filtsize[2], self._stride[0], self._stride[1], self._padding) + ',bias=' + \
            str(self.bias is not None) + ')' 
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        y0 = outp[1]*self._stride[0] - self._padding[0]
        if y0 < 0:
            y0 = 0
        x0 = outp[2]*self._stride[1] - self._padding[1]
        if x0 < 0: 
            x0 = 0
        y1 = outp[4]*self._stride[0] + self.filtsize[1] - 1 - self._padding[0]
        if y1 >= self.insize[1]:
            y1 = self.insize[1] - 1
        x1 = outp[5]*self._stride[1] + self.filtsize[2] - 1 - self._padding[1]
        if x1 >= self.insize[2]: 
            x1 = self.insize[2] - 1
        
        return [0, y0, x0, self.insize[0] - 1, y1, x1]
    
    
    # get weights
    @property
    def w(self):
        return self.weight.data.numpy()
    
    
    # set weights
    @w.setter
    def w(self, wmat):
        if type(wmat) != np.ndarray:
            raise TypeError('Provided weight matrix is not a *NumPy* array')
        if wmat.shape != (self.filtsize[0], self.insize[0], self.filtsize[1], self.filtsize[2]):
            raise TypeError('Provided weight tensor has wrong size.')
        self.weight.data = torch.from_numpy(wmat)
    
    
    # get bias
    @property
    def b(self):
        if self.bias is not None:
            return self.bias.data.numpy()
        else:
            return None
    
    
    # set bias
    @b.setter
    def b(self, bvec):
        if self.bias is not None:
            if type(bvec) != np.ndarray:
                raise TypeError('Provided bias vector is not a *NumPy* array')
            if bvec.ndim == 1:
                bvec = bvec.reshape((len(bvec),1)).copy()
            if bvec.shape != (self.filtsize[0], 1):
                raise TypeError('Provided bias vector has wrong size.')
            self.bias.data = torch.from_numpy(bvec)


# 1d Convolutional layer class
class Conv1d(nn.Conv1d):
    '''
    Convolutional layer::
    
        conv0 = Conv([m,n,p], [c,r,s], [ystride,xstride], [ypadding, xpadding], bias, init)
    
    creates a convolutional layer for unflattened 4d input tensors. The weight matrix is 
    initialized according to 'init' which must be one of the initialization methods 
    described in pyTorch.
    
    This layer implements a convolutional layer with a set of 'c' learned filters that
    are applied to a sequence of m-channel images. The output of each filter is
    the cross-correlation of the learned m x r x s filter mask with the multi channel
    image where 'm' is the number of input channels, 'r' the filter mask height and 's' 
    the filter mask width. 
    
    The output image can be downscaled by a factor of 'ystride' in y- and 'xstride' in 
    x-direction. In this way, a strided convolution is possible. This is useful for large 
    input images. You can use a single number if the strides in both directions is equal.
    
    Additional boundary pixels with value 0 are allocated when 'padding' is set to value 
    larger than 0. Note that in this case the same number of rows and columns are added, independently
    of the filter size. Alternatively, you can two different paddings in y- and x-direction
    by setting 'padding' to 2-tupke. If you set 'padding' to zero you obtain the classical
    'valid' padding used, e.g. in LeNet which leads to smaller output images. 
    By setting 'padding' to half the filter size you
    get the 'half' padding used in Theano. This corresponds to standard zero padding as
    used in image processing and preserves the image size.
    
    In addition, each filter also has a trainable bias term similar to standard linear 
    layers. If you do not want a bias term set the argument 'bias' to False.
    
    The layer accepts only input tensors of size [mini_batch_size, m, n, p]. If the input 
    is in flattened format, an Unflatten node 
    has to be inserted as input layer. 
    
    Attributes:
    
    * w: weight tensor of the layer
    * b: bias vector
    
    '''
    # constructor
    def __init__(self, insize, fsize, stride=1, padding=0, dilation=1, groups=1, 
        bias=True, init='xavier_normal_'):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 2:
            raise TypeError('Input size list must have 2 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        self.filtsize = list(fsize)
        if len(self.filtsize) != 2:
            raise TypeError('Filter size list must have 2 elements.')
        for s in self.filtsize:
            s = int(s)
            if s < 1:
                raise ValueError('Filter size must be > 0.')
        self._stride = int(stride)
        if self._stride < 1:
            raise ValueError('Stride must be > 0.')
        self._padding = int(padding)
        if self._padding < 0:
            raise ValueError('Padding must be >= 0.')
        
        # create pytorch conv1d
        super(Conv1d, self).__init__(self.insize[0], self.filtsize[0], self.filtsize[1], 
            self._stride, self._padding, dilation, groups, bias)
        
        # initialize weights
        init = str(init)
        nn.init.__dict__[init](self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias,0.0)
        
        # output size
        out_size = (self.insize[1] + 2*self._padding - self.filtsize[1]) // \
            self._stride + 1
        self.outsize = [self.filtsize[0], out_size]
        
        # network stats
        self.num_neurons = self.outsize[0] * self.outsize[1]
        self.num_weights = self.num_neurons*self.filtsize[1]*self.insize[0] 
        self.num_adj_weights = self.filtsize[0]*self.insize[0]*self.filtsize[1]
        if self.bias is not None:
            self.num_weights += self.num_neurons
            self.num_adj_weights += self.filtsize[0]
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{}]->[bs,{},{}],mask[{},{},{}],str={},pad={}'.format(
            self.insize[0], self.insize[1], self.outsize[0], self.outsize[1], 
            self.filtsize[0], self.insize[0], self.filtsize[1], 
            self._stride, self._padding) + ',bias=' + \
            str(self.bias is not None) + ')' 
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        x0 = outp[1]*self._stride - self._padding
        if x0 < 0:
            x0 = 0
        x1 = outp[3]*self._stride + self.filtsize[1] - 1 - self._padding
        if x1 >= self.insize[1]:
            x1 = self.insize[1] - 1
        
        return [0, x0, self.insize[0] - 1, x1]
    
    
    # get weights
    @property
    def w(self):
        return self.weight.data.numpy()
    
    
    # set weights
    @w.setter
    def w(self, wmat):
        if type(wmat) != np.ndarray:
            raise TypeError('Provided weight matrix is not a *NumPy* array')
        if wmat.shape != (self.filtsize[0], self.insize[0], self.filtsize[1]):
            raise TypeError('Provided weight tensor has wrong size.')
        self.weight.data = torch.from_numpy(wmat)
    
    
    # get bias
    @property
    def b(self):
        if self.bias is not None:
            return self.bias.data.numpy()
        else:
            return None
    
    
    # set bias
    @b.setter
    def b(self, bvec):
        if self.bias is not None:
            if type(bvec) != np.ndarray:
                raise TypeError('Provided bias vector is not a *NumPy* array')
            if bvec.ndim == 1:
                bvec = bvec.reshape((len(bvec),1)).copy()
            if bvec.shape != (self.filtsize[0], 1):
                raise TypeError('Provided bias vector has wrong size.')
            self.bias.data = torch.from_numpy(bvec)
# 
# 
# # Deconvolution layer class
# class Deconv(Layer):
#     '''
#     Deconvolution layer::
#     
#         deconv = Deconv(id, [m,n,p], cnet, ['name 1', 'name 2',..], [c,r,s], init, padding, 
#             ystride, xstride, wdecay)
#     
#     creates a deconvolution layer with ID string 'id' for unflattened 4d input tensors.
#     It is designed to invert the effects of a Conv layer in the sense that it produces
#     the same output size as the input size of a Conv Layer with same filter mask size and
#     padding type. Thus, Deconv is preferably meant for being used in autoencoders, not as
#     an exact implementation of a flipped correlation layer.
# 
#     The layer is registered in the network 'cnet' and connected to the input
#     layers 'name 1', 'name 2 ' etc. The weight matrix is initialized
#     according to 'init' which must be one of the initialization methods described in
#     the base class Layer. The actual initialization is done in Network.init_weights().
#     
#     This node implements a deconvolution layer with a set of 'c' learned filters that
#     are applied to a sequence of m-channel images. The output of each filter is
#     the convolution of the learned m x r x s filter mask with the multi channel
#     image where 'm' is the number of input channels, 'r' the filter mask height and 's' 
#     the filter mask width. Filter width and height must be odd numbers, so, e.g., a 
#     c x 2 x 2 filter is not allowed. In addition, you can specify the strides 'ystride'
#     and 'xstride' which lead to an enlargement of the output image by these factors.
#     
#     The user can set the boundary treatment to the following values:
#     
#     1. 'valid': The c x m x n volume is xy-flipped and scanned over the image. The output
#     image is scaled such as to revert the effect of 'valid' convolution, i.e., the output
#     image is larger than the input image by filter size - 1. If strides > 1 are used the
#     output image is first upsampled by the stride factors and shifted such that the spikes
#     are always at the center of the striding filter mask. The upsampled image is then
#     convolved over the entire image with 'half' padding.
#     
#     2. 'half': here, the volume is scanned over the entire image. Input regions outside
#     the image are set to zero. Without strides, the output image has the same size as the
#     input image, otherwise it is first upsampled and then convolved.
#     
#     In addition, each filter also has a trainable bias term similar to standard linear 
#     layers.
#     
#     The node accepts only input tensors of size [mini_batch_size, q, n, p] and produces
#     output tensors of size [mini_batch_size, c, (q - 1)*ystride* + r, (p - 1)*xstride + s]
#     for 'valid' boundary conditions and [mini_batch_size, c, ystride*q, xstride*p] for
#     'half' boundary conditions. If the input is in flattened format, an Unflatten node has
#     to be inserted as input layer. You can set an L2 weight decay 'wdecay' for the weights
#     for this layer individually.
#     
#     Attributes:
#     
#     * weights: weight tensor of the layer
#     * bias: bias vector
#     * filtsize: filter size, [ num_maps, filter_height, filter_width ] (read only)
#     * padding: padding type (either 'valid' or 'half', read only)
#     * ystride: stride in y-direction (read only)
#     * xstride: stride in x-direction (read only)
# 
#     '''
#     # padding types
#     padding_types = [ 'valid', 'half' ]
#     
#     # constructor
#     def __init__(self, id, insize, netw, inputs=[], fsize=[], init='xavier_gaussian', 
#         padding='valid', ystride=1, xstride=1, wdecay=0.0):
#         super(Deconv, self).__init__(id, insize, netw, inputs, init)
#         
#         # check layer parameters
#         if len(insize) != 3:
#             _lg.error('Input size list must have 3 elements.', TypeError)
#         self.filtsize = list(fsize)
#         if len(self.filtsize) != 3:
#             _lg.error('Filter size list must have 3 elements.', TypeError)
#         if padding not in self.padding_types:
#             _lg.error('Unknown padding type: {}'.format(padding))
#         self._padding = padding
#         for s in self.filtsize:
#             s = int(s)
#             if s < 1:
#                 _lg.error('Filter size must be > 0.', ValueError)
#         if self.padding == 'half':
#             if self.filtsize[1] < 3 or self.filtsize[2] < 3:
#                 _lg.error('Filter size for HALF padding must be at least 3.', ValueError)
#             if not self.filtsize[1] % 2 or not self.filtsize[2] % 2:
#                 _lg.error('Filter size for HALF padding must be at odd.', ValueError)
#         ystride = int(ystride)
#         if ystride < 1:
#             _lg.error('ystride must be > 0.', ValueError)
#         self._ystride = ystride
#         xstride = int(xstride)
#         if xstride < 1:
#             _lg.error('xstride must be > 0.', ValueError)
#         self._xstride = xstride
#         
#         # output size
#         if self.padding == 'valid':
#             out_height = (self.insize[2] - 1) * self.ystride + self.filtsize[1]
#             out_width  = (self.insize[3] - 1) * self.xstride + self.filtsize[2]
#         else:
#             out_height = self.insize[2] * self.ystride
#             out_width  = self.insize[3] * self.xstride
#         self.outsize = [netw.mini_batch_size, self.filtsize[0], out_height, out_width]
#         
#         # add layer to network
#         self.trainable = True
#         self.num_neurons = self.outsize[1] * self.outsize[2] * self.outsize[3]
#         self.num_weights = self.num_neurons*self.filtsize[1]*self.filtsize[2] + self.num_neurons
#         self.num_adj_weights = \
#             self.filtsize[0]*self.insize[1]*self.filtsize[1]*self.filtsize[2] + self.filtsize[0]
#         netw.register_node(self)
#         _lg.info("Deconv              ({}->{},mask[{},{},{},{}],{},str=[{},{}],wdec={:.3f}) '{}'".format(
#             self.insize, self.outsize, self.filtsize[0], self.insize[1], self.filtsize[1], 
#             self.filtsize[2], self.padding, self.ystride, self.xstride, wdecay, self.id))
#         
#         # init shared variables
#         self._W = self.add_adj_weight(np.zeros((self.filtsize[0], self.insize[1],
#             self.filtsize[1], self.filtsize[2]), dtype=np.float32), "W", wdecay)
#         self._b = self.add_adj_weight(np.zeros((self.filtsize[0], 1), dtype=np.float32), 
#             "b", 0.0)
#     
#     
#     # Symbolically defined forward path
#     def forward(self, input, **kwargs):
#         '''
#         Computes symbolically the convolution y = W * x + b. Called by Network._compile().
#         '''
#         btensor = T.reshape(self._b, tuple((1, self.outsize[1], 1, 1)))
#         if self.ystride == 1 and self.xstride == 1 and self.padding == 'half':
#             convd = T.nnet.conv2d(input, self._W, tuple(self.insize), tuple(self.weights.shape),
#                 self.padding, (1,1), True)
#         else: # strided convolution
#             sz = tuple((self.insize[0], self.insize[1], self.outsize[2], self.outsize[3]))
#             upscaled = T.zeros(shape=sz, dtype=input.dtype)
#             if self.padding == 'valid':
#                 miny = self.filtsize[1]//2
#                 minx = self.filtsize[2]//2
#                 maxy = (self.insize[2] - 1) * self.ystride + miny + 1
#                 maxx = (self.insize[3] - 1) * self.xstride + minx + 1
#                 upscaled = T.set_subtensor(upscaled[:,:,
#                     miny:maxy:self.ystride,minx:maxx:self.xstride], input)
#             else:
#                 upscaled = T.set_subtensor(upscaled[:,:,::self.ystride,::self.xstride], input)
#             upscaled = (1.0/(self.xstride*self.ystride)) * upscaled
#             convd = T.nnet.conv2d(upscaled, self._W, sz, tuple(self.weights.shape),
#                 'half', (1,1), True)
#         return convd + T.patternbroadcast(btensor, (True, False, True, True))
#     
#     
#     # initialize weights
#     def init_weights(self):
#         '''
#         Initializes the weights according to 'self.init_type'.
#         '''
#         # compute width of distribution
#         fan_in = self.insize[1] * self.filtsize[1] * self.filtsize[2]
#         fan_out = self.filtsize[0] * self.filtsize[1] * self.filtsize[2]
#         
#         # Gaussian init
#         if self.init_type == 'xavier_gaussian':
#             dev = math.sqrt(2.0/(fan_in + fan_out))
#             self.weights = global_rng2.normal(0.0, dev, (self.filtsize[0], self.insize[1],
#                 self.filtsize[1], self.filtsize[2])).astype('float32')
#             self.bias = 0.1 * np.ones((self.filtsize[0], 1), dtype=np.float32)
#         
#         # Uniform init
#         elif self.init_type == 'xavier_uniform':
#             dev = math.sqrt(6.0/(fan_in + fan_out))
#             self.weights = global_rng2.uniform(-dev, dev, (self.filtsize[0], self.insize[1],
#                 self.filtsize[1], self.filtsize[2])).astype('float32')
#             self.bias = 0.1 * np.ones((self.filtsize[0], 1), dtype=np.float32)
#         
#         # Truncated normal
#         else:
#             dev = math.sqrt(2.0/(fan_in + fan_out))
#             self.weights = np.reshape(truncnorm.rvs(-1.0, 1.0, 0.0, dev, 
#                 self.filtsize[0]*self.insize[1]*self.filtsize[1]*self.filtsize[2]), 
#                 (self.filtsize[0], self.insize[1], self.filtsize[1], self.filtsize[2])).astype('float32')
#             self.bias = 0.1 * np.ones((self.filtsize[0], 1), dtype=np.float32)
#     
#     
#     # Get the input block of a given output block
#     def get_input_block(self, outp):
#         '''
#         Returns the input block that feeds into the specified output block.
#         '''
#         if self.padding == 'valid':
#             y0 = outp[1] // self.ystride
#             x0 = outp[2] // self.xstride
#             y1 = (outp[4] + self.filtsize[1] - 1) // self.ystride
#             x1 = (outp[5] + self.filtsize[2] - 1) // self.xstride
#             
#         else: # half padding
#             y0 = (outp[1] - self.filtsize[1]//2) // self.ystride
#             if y0 < 0:
#                 y0 = 0
#             x0 = (outp[2] - self.filtsize[2]//2) // self.xstride
#             if x0 < 0: 
#                 x0 = 0
#             y1 = (outp[4] + self.filtsize[1]//2) // self.ystride
#             if y1 >= self.insize[2]:
#                 y1 = self.insize[2] - 1
#             x1 = (outp[5] + self.filtsize[2]//2) // self.xstride
#             if x1 >= self.insize[3]: 
#                 x1 = self.insize[3] - 1
#         
#         return [0, y0, x0, self.insize[1] - 1, y1, x1]
#     
#     
#     # get weights
#     @property
#     def weights(self):
#         return self._W.get_value()
#     
#     # set weights
#     @weights.setter
#     def weights(self, w):
#         if type(w) != np.ndarray:
#             _lg.error('Provided weight matrix is not a *NumPy* array')
#         if w.shape != (self.filtsize[0], self.insize[1], self.filtsize[1], self.filtsize[2]):
#             _lg.error('Provided weight tensor has wrong size.', TypeError)
#         if w.dtype != 'float32':
#             w = w.astype('float32')
#         self._W.set_value(w)
#     
#     # get bias
#     @property
#     def bias(self):
#         return self._b.get_value()
#     
#     # set bias
#     @bias.setter
#     def bias(self, b):
#         if type(b) != np.ndarray:
#             _lg.error('Provided bias vector is not a *NumPy* array')
#         if b.ndim == 1:
#             b = b.reshape((len(b),1)).copy()
#         if b.shape != (self.filtsize[0], 1):
#             _lg.error('Provided bias vector has wrong size.', TypeError)
#         if b.dtype != 'float32':
#             b = b.astype('float32')
#         self._b.set_value(b)
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
