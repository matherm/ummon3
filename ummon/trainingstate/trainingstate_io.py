from .trainingstate_dict import TrainingStateDict
import torch, os, shutil
import torch.nn as nn
import numpy as np

class Trainingstate(TrainingStateDict):

    def __init__(self, filename = None, force_weights_to_cpu = True, model_keep_epochs=False):
        super().__init__()
        
        self.extension = ".pth.tar"
        self.train_pattern = "_best_training_loss"
        self.valid_pattern = "_best_valid_loss"
        self.combined_retraining_pattern = "_comb_retrn"
        self.force_weights_to_cpu = force_weights_to_cpu
        self.model_keep_epochs = model_keep_epochs
        self.filename = filename 
        if self.filename is not None and ("/dev/null/" in self.filename or "" == self.filename or " " == self.filename):
            self.filename = None 

        if filename is not None:
            if self.extension not in filename:
                self.filename = str(filename + self.extension)
            if os.path.exists(self.filename):
                self.load_state_(self.filename, force_weights_to_cpu)


    def load_state_(self, filename, force_weights_to_cpu=True):
        """
        Loads the state from file.
        
        Arguments
        ------
        *filename (String) : A file on disk representing a trainingstate
        *OPTIONAL force_weights_to_cpu (bool) : PyTorch speciality 
                                                that converts CUDA floats to CPU floats (default True)
        
        """
        assert filename is not None
        
        if force_weights_to_cpu:
            state = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            state = torch.load(filename)        

        self.state = state
        
        
    def save_state(self):
        """
        Saves the state to file and maintaines a copy of the best validation and training model.
        
        """
        if self.filename is None:
            return

        filename = self.filename
        extension = self.extension
        train_pattern = self.train_pattern
        valid_pattern = self.valid_pattern
        keep_epochs = self.model_keep_epochs
        
        # CREATE FOLDERS
        if "/" in filename and not os.path.exists(filename[0:filename.rfind("/")]):
            os.makedirs(filename[0:filename.rfind("/")])

        # UPDATE NAME            
        filename = filename.replace(train_pattern, '').replace(valid_pattern, '')
        
        short_filename = filename.split(extension)[0]
        if keep_epochs:
            epoch = self.state["lrate[]"][-1][0]
            filename = short_filename + "_epoch_" + str(epoch) + extension
            torch.save(self.state, filename)
        else:
            filename = short_filename + extension
            torch.save(self.state, filename)  
        
        if self.is_best_training_model():
            shutil.copyfile(filename, str(short_filename + train_pattern + extension))

        if self.is_best_validation_model():
            shutil.copyfile(filename, str(short_filename + valid_pattern + extension))
            
    
    def load_weights_best_training_(self, model, optimizer=None):
        """
        Loads the persisted weights into a given model.
        
        Attention
        ---------
        The model needs to be the same as the persisted model.
        
        Arguments
        ---------
        *model (torch.nn.Module) : A model that needs to be filled with the stored weights.
        *optimizer (torch.optim.Optimizer) : A optimizer that needs to be repointed to the new weights
        
        """
        assert isinstance(model, nn.Module)
        assert self.filename is not None

        self.load_state_(str(self.filename + self.train_pattern + self.extension), self.force_weights_to_cpu)
        self.load_weights_(model, optimizer)           
    
    def load_weights_best_validation_(self, model, optimizer=None):
        """
        Loads the persisted weights into a given model.
        
        Attention
        ---------
        The model needs to be the same as the persisted model.
        
        Arguments
        ---------
        *model (torch.nn.Module) : A model that needs to be filled with the stored weights.
        *optimizer (torch.optim.Optimizer) : A optimizer that needs to be repointed to the new weights
        
        """
        assert isinstance(model, nn.Module)
        assert self.filename is not None
        
        self.load_state_(str(self.filename + self.valid_pattern + self.extension), self.force_weights_to_cpu)
        self.load_weights_(model, optimizer)           
    
        
    def load_weights_(self, model, optimizer=None):
        """
        Loads the persisted weights into a given model.
        
        Attention
        ---------
        The model needs to be the same as the persisted model.
        
        Arguments
        ---------
        *model (torch.nn.Module) : A model that needs to be filled with the stored weights.
        *optimizer (torch.optim.Optimizer) : A optimizer that needs to be repointed to the new weights
        
        """
        assert self.state is not None
        assert isinstance(model, nn.Module)
            
        model.load_state_dict(self.state["model_state"])            
        
        if optimizer is not None:
            Trainingstate.update_optimizer_weights_(model, optimizer)
    
    def load_optimizer_(self, optimizer, cuda=False):
        """
        Loads the persisted weights into a given optimizer.
        
        Arguments
        ---------
        *model (torch.optim.Optimizer) : A optimizer that needs to be filled with the stored weights.
        
        Return
        ------
        *model (torch.optim.Optimizer) : The initialized optimizer.
        
        """
        assert self.state is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
            
        optimizer.load_state_dict(self.state["optimizer_state"])
        
    
    def load_scheduler_(self, scheduler):
        assert self.state is not None
        
        if isinstance(scheduler, StepLR_earlystop):
            scheduler.load_state_dict(self.state['scheduler_state'])
        

    @staticmethod
    def transform_model(model, optimizer, precision, use_cuda = False):
        """
        Transforms the model weights to an artbitray precision or device like cuda.

        Arguments
        ---------
        *model (nn.module) : The model to be transformed.
        *optimizer (torch.optim.Optimizer) : A optimizer that needs to be repointed to the new weights.
        *precision (np.dtype) : The target dtype
        *use_cuda (bool) : Shall model be transformed to cuda.

        Return
        ------
        *model (nn.module) : The transformed model.

        """
        assert isinstance(model, nn.Module)
        assert precision == np.float32 or precision == np.float64
     
        # Computational configuration
        if precision == np.float32:
            model = model.float()
        if precision == np.float64:
            model = model.double()
        if precision == np.int32:
            # TODO: Custom model conversion FPGA-Teamproject
            pass
        if use_cuda:
            assert torch.cuda.is_available() == True
            model = model.cuda()
        else:
            model = model.cpu()

        if optimizer is not None:
            Trainingstate.update_optimizer_weights_(model, optimizer)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        if use_cuda:
                            state[k] = v.cuda()
                        else:
                            state[k] = v.cpu()

        return model
    
    @staticmethod
    def update_optimizer_weights_(model, optimizer):
        """
        Repoints the weights of an optimizer to a new model.
       
        Helper method to repair references after the model's parameters have changed.
        This happens when the model is converted to CUDA or older weights are loaded.
        When this happens, the optimizer optimizes old weights as he does not have the current weights.
        Therefore we need to repoint the optimizers weights to the new model.
        """
        # Delete the current weights
        del optimizer.param_groups[:]
    
        param_groups = list(model.parameters())
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
    
        for param_group in param_groups:
                optimizer.add_param_group(param_group)

from ..schedulers.early_stopping import StepLR_earlystop