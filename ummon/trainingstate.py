##################################################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################
import torch
import torch.nn as nn
import shutil
import numpy as np
import ummon.utils as uu
from collections import OrderedDict

__all__ = [ "Trainingstate" ]

class Trainingstate():
    """
    Small wrapper class for persisting trainingstate as a Python Dictionary. 
    It combines `model.state_dict()` and `optimzer.state_dict()` plus some information during training like 
    the current learning rate and the loss function.
    
    E.g. Trainingstate().state["loss[]"] gives you a list of tuples as [(epoch, loss)] over the whole training.
    
    Members
    -------
    
    state : Dictionary
            Python Dictionary that holdes the training state
            
    
    Methods
    -------
    
    update_state()  : Method for updating training state
    load_state()    : Method for loading a persisted state
    save_state()    : Method for persisting the current training state.
    
    """
    
    def __init__(self, filename = None, force_weights_to_cpu = True):
        assert filename != ' ' and filename != ''
        
        self.state = None
        self.extension = ".pth.tar"
        self.train_pattern = "_best_training_loss"
        self.valid_pattern = "_best_valid_loss"
        self.combined_retraining_pattern = "_comb_retrn"
        self.force_weights_to_cpu = force_weights_to_cpu
        self.filename = filename

        if filename is not None:
            if self.extension not in filename:
                self.filename = str(filename + self.extension)
            self.load_state(self.filename, force_weights_to_cpu)
        
    def update_state(self, 
                     epoch, model, loss_function, optimizer, training_loss, 
                     training_accuracy = 0,
                     training_dataset = None,
                     training_batchsize = 0,
                     trainer_instance = "",
                     precision = np.float32,
                     detailed_loss = {},
                     validation_loss = None, 
                     validation_accuracy = 0., 
                     validation_dataset = None,
                     samples_per_second = None,
                     scheduler = None,
                     combined_retraining = False,
                     args = {}):
        """
        Updates the trainingstate with the given parameters.
        If no validation data available, validation data will not be persisted and therefor no dummy data 
        occurs in the resulting dictionary.
        
        """
        # COMPATIBILITY FOR TORCH < v.0.4
        if type(validation_loss) == torch.Tensor:
            validation_loss = validation_loss.item()
        if type(training_loss) == torch.Tensor:
            training_loss = training_loss.item()


        # INITIALIZE NEW STATE
        if self.state is None:
            if validation_accuracy is not None and validation_loss is not None and validation_dataset is not None:
                validation_accuracy_list = [(epoch, validation_accuracy, len(validation_dataset))]
                validation_loss_list = [(epoch, validation_loss, len(validation_dataset))]
                best_validation_accuracy = (epoch, validation_accuracy, len(validation_dataset))
                best_validation_loss = (epoch, validation_loss, len(validation_dataset))
                validation_dataset = uu.get_data_information(validation_dataset)
            else:
                validation_accuracy_list = []
                validation_loss_list = []
                best_validation_accuracy = (epoch, 0., 0.)
                best_validation_loss = (epoch, np.finfo(np.float32).max, 0)
                validation_dataset = None
            self.state = {  
                         "model_desc" : str(model),
                         "model_trainable_params" : uu.count_parameters(model),
                         "loss_desc"  : str(loss_function),
                         "cuda" : next(model.parameters()).is_cuda, 
                         "trainer_instance" : trainer_instance,
                         "precision" : precision,
                         "dataset_training" : uu.get_data_information(training_dataset),
                         "dataset_validation" : validation_dataset,
                         "samples_per_second[]" : [(epoch, samples_per_second)],
                         "init_optimizer_state" : optimizer.state_dict(),
                         "lrate[]" : [(epoch, optimizer.state_dict()["param_groups"][0]["lr"])],
                         "model_state" : model.state_dict(),
                         "optimizer_state": optimizer.state_dict(),
                         "best_training_loss" : (epoch, training_loss, training_batchsize),
                         "best_training_accuracy" : (epoch, training_accuracy, training_batchsize),
                         "training_loss[]" : [(epoch, training_loss, training_batchsize)],
                         "training_accuracy[]" : [(epoch, training_accuracy, training_batchsize)],
                         "best_validation_loss" :  best_validation_loss,
                         "best_validation_accuracy" : best_validation_accuracy,
                         "validation_loss[]" : validation_loss_list,
                         "validation_accuracy[]" : validation_accuracy_list ,
                         "detailed_loss[]" : [(epoch, detailed_loss)],
                         "scheduler_state" : scheduler.state_dict() if isinstance(scheduler, StepLR_earlystop) else None,
                         "combined_retraining" : combined_retraining,
                         "args[]" : [args]
                          }
        else:
            # APPEND STATE
            if validation_accuracy is not None and validation_loss is not None and validation_dataset is not None:                
                if "dataset_validation" in self.state:
                    dataset_validation_info = self.state["dataset_validation"] 
                else:
                    dataset_validation_info = uu.get_data_information(validation_dataset)                
                if validation_loss < self.state["best_validation_loss"][1]:
                    best_validation_loss = (epoch, validation_loss, len(validation_dataset)) 
                else: 
                    best_validation_loss = self.state["best_validation_loss"]
                if validation_accuracy > self.state["best_validation_accuracy"][1]:
                    best_validation_acc = (epoch, validation_accuracy, len(validation_dataset)) 
                else:
                    best_validation_acc  = self.state["best_validation_accuracy"]
                validation_loss_list = [*self.state["validation_loss[]"], (epoch, validation_loss, len(validation_dataset))]
                validation_acc_list = [*self.state["validation_accuracy[]"], (epoch, validation_accuracy, len(validation_dataset))]
            else:
                dataset_validation_info = self.state["dataset_validation"] 
                best_validation_loss = self.state["best_validation_loss"]
                best_validation_acc = self.state["best_validation_accuracy"]
                validation_loss_list = self.state["validation_loss[]"]
                validation_acc_list = self.state["validation_accuracy[]"]

            if "dataset_training" in self.state:
                dataset_training_info = self.state["dataset_training"] 
            else: 
                dataset_training_info = uu.get_data_information(training_dataset)
            if "samples_per_second[]" in self.state:
                samples_per_second_info = [*self.state["samples_per_second[]"], (epoch, samples_per_second)] 
            else:
                samples_per_second_info = [(epoch, samples_per_second)]
            if training_loss< self.state["best_training_loss"][1]:
                best_training_loss = (epoch, training_loss, training_batchsize) 
            else:
                best_training_loss = self.state["best_training_loss"]
            if training_accuracy > self.state["best_training_accuracy"][1]:
                best_training_acc = (epoch, training_accuracy, training_batchsize) 
            else:
                best_training_acc = self.state["best_training_accuracy"]
            if "detailed_loss[]" in self.state:
                detailed_loss_info = [*self.state["detailed_loss[]"], (epoch, detailed_loss)] 
            else:
                detailed_loss_info = [(epoch, detailed_loss)]
            if len(args) == 0:
                args = self.state["args[]"]
            else:
                args = [*self.state["args[]"]]
            self.state = {  
                         "model_desc" : str(model),
                         "model_trainable_params" : uu.count_parameters(model),
                         "loss_desc"  : str(loss_function),
                         "cuda" : next(model.parameters()).is_cuda, 
                         "trainer_instance" : trainer_instance,
                         "precision" : precision,
                         "dataset_training" : dataset_training_info,
                         "dataset_validation" : dataset_validation_info,
                         "samples_per_second[]" : samples_per_second_info,
                         "init_optimizer_state" : self.state["init_optimizer_state"],
                         "lrate[]" : [*self.state["lrate[]"], (epoch, optimizer.state_dict()["param_groups"][0]["lr"])],
                         "model_state" : model.state_dict(),
                         "optimizer_state": optimizer.state_dict(),
                         "best_training_loss" : best_training_loss,
                         "best_training_accuracy" : best_training_acc,
                         "training_loss[]" : [*self.state["training_loss[]"], (epoch, training_loss, training_batchsize)],
                         "training_accuracy[]" : [*self.state["training_accuracy[]"], (epoch, training_accuracy, training_batchsize)],
                         "best_validation_loss" : best_validation_loss,
                         "best_validation_accuracy" : best_validation_acc,
                         "validation_loss[]" : validation_loss_list,
                         "validation_accuracy[]" : validation_acc_list,
                         "detailed_loss[]" : detailed_loss_info,
                         "scheduler_state" : scheduler.state_dict() if isinstance(scheduler, StepLR_earlystop) else None,
                         "combined_retraining" : self.state["combined_retraining"],
                         "args[]" : args
                          }
    def __repr__(self):
        formatted_string = [str(str(key) + "\t" + str(value) + "\n") for key,value in sorted(self.state.items())]
        return str(''.join(formatted_string))
    
    
    def __str__(self):
        formatted_string = [str(str(key) + "\t" + str(value) + "\n") for key,value in sorted(self.state.items())]
        return str(''.join(formatted_string))
    
    
    def __getitem__(self, item):
         return self.state[item]
    
    def get_summary(self):
        """
        Returns a summary of the trainingstate
        
        Return
        ------
        *summary (dict) : A summary containing 'epochs', 'Best Training Loss' and 'Best Validation Loss'
        
        """
        if self.state is None:
            return {}

        return OrderedDict({
                    "Epochs"               : self.state["training_loss[]"][-1][0],
                    "Best Training Loss"   : self.state["best_training_loss"],
                    "Best Validation Loss" : self.state["best_validation_loss"],
                  })
        
        
    def get_loss(self):
        """
        Returns a summary of the trainingstate's losses
        
        Return
        ------
        *summary (dict) : A summary containing 'epochs', 'Best Training Loss' and 'Best Validation Loss', 'validation_loss[]', 'training_loss[]'
        
        """
        if self.state is None:
            return {}

        return OrderedDict({
                    "Epochs"               : self.state["training_loss[]"][-1][0],
                    "Best Training Loss"   : self.state["best_training_loss"],
                    "Best Validation Loss" : self.state["best_validation_loss"],
                    "validation_loss[]"    : self.state["validation_loss[]"],
                    "training_loss[]"      : self.state["training_loss[]"],
                  })
    
    
    def load_state(self, filename = None, force_weights_to_cpu = True):
        """
        Loads the state from file.
        
        Arguments
        ------
        *filename (String) : A file on disk representing a trainingstate
        *OPTIONAL force_weights_to_cpu (bool) : PyTorch speciality 
                                                that converts CUDA floats to CPU floats (default True)
        
        """
        assert filename is not None or self.filename is not None
        if filename == None:
            filename = self.filename
        if self.extension not in filename:
            filename = str(filename + self.extension)
        
        if force_weights_to_cpu:
            self.state = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            self.state = torch.load(filename)        

        # UPDATE NAME            
        self.filename = filename.replace(self.train_pattern, '').replace(self.valid_pattern, '')
        
    def save_state(self, filename = None, keep_epochs = False):
        """
        Saves the state to file and maintaines a copy of the best validation and training model.
        
        Arguments
        ------
        *filename (String) : A file on disk representing a trainingstate
        *OPTIONAL keep_epochs (bool) : When TRUE the state of every epoch is stored 
                                       with pattern "MODEL_epoch_{NUMBER}.pth.tar" (default False).
        """
        assert filename is not None or self.filename is not None
        if filename == None:
            filename = self.filename
        if self.extension not in filename:
            filename = str(filename + self.extension)

        # UPDATE NAME            
        self.filename = filename.replace(self.train_pattern, '').replace(self.valid_pattern, '')
        
        short_filename = filename.split(self.extension)[0]
        if keep_epochs:
            epoch = self.state["lrate[]"][-1][0]
            filename = short_filename + "_epoch_" + str(epoch) + self.extension
            torch.save(self.state, filename)
        else:
            filename = short_filename + self.extension
            torch.save(self.state, filename)  
        
        def is_best_train(state):
            if len(state["training_loss[]"]) == 0:
                return False
            return state["training_loss[]"][-1][1] == state["best_training_loss"][1]
        
        def is_best_valid(state):
            if len(state["validation_loss[]"]) == 0:
                return False
            return state["validation_loss[]"][-1][1] == state["best_validation_loss"][1]
        
        if is_best_train(self.state):
            shutil.copyfile(filename, str(short_filename + self.train_pattern + self.extension))

        if is_best_valid(self.state):
            shutil.copyfile(filename, str(short_filename + self.valid_pattern + self.extension))
            
    
    def load_weights_best_training_(self, model, optimizer):
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
        short_filename = self.filename.split(self.extension)[0]

        if self.train_pattern in short_filename:
            self.load_state(str(short_filename + self.extension), self.force_weights_to_cpu)
        else:
            self.load_state(str(short_filename + self.train_pattern + self.extension), self.force_weights_to_cpu)
           
        self.load_weights_(model, optimizer)           
    
    def load_weights_best_validation_(self, model, optimizer):
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
        short_filename = self.filename.split(self.extension)[0]
        
        if self.valid_pattern in short_filename:
            self.load_state(str(short_filename + self.extension), self.force_weights_to_cpu)
        else:
            self.load_state(str(short_filename + self.valid_pattern + self.extension), self.force_weights_to_cpu)
        
        self.load_weights_(model, optimizer)           
    
    
    def load_weights_(self, model, optimizer):
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
 

from .schedulers import StepLR_earlystop
