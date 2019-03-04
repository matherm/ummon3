import torch
import numpy as np

class EmbeddInEmpty():
    
    def __init__(self, final_size=(60,60)):
        self.final_size= final_size

    def __call__(self, image):
        '''
        This method pastes the given image into a greater image specified by final_size.
        
        Arguments
        --------
        Image as [Channels, Height, Width]
        
        Returns
        -------
        New Image as [Channels, (final_size)]
        '''
        assert isinstance(image, torch.Tensor)
        assert image.numpy().ndim == 3
        final_size = self.final_size
    
        greater_im = torch.FloatTensor(image.size(0), final_size[0], final_size[1]).zero_() 
        # COMPUTE RANDOM POSITION
        x = int(np.random.uniform(0,final_size[0] - image.size(1)))
        y = int(np.random.uniform(0,final_size[1] - image.size(2)))
        # PLACE IMAGE INTO NEW POSITION
        greater_im[:,x:x+image.size(1),y:y+image.size(2)] = image  
        return greater_im