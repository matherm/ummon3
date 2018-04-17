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
from ummon.utils import Torchutils

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
        self.force_weights_to_cpu = force_weights_to_cpu

        if filename is not None:
            if self.extension not in filename:
                self.filename = str(filename + self.extension)
            else:
                self.filename = filename
            self.short_filename = filename.split(self.extension)[0]
            self.load_state(self.filename, force_weights_to_cpu)
        
    def update_state(self, 
                     epoch, model, loss_function, optimizer, training_loss, 
                     validation_loss = None, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = None, 
                     validation_batchsize = 0,
                     regression = False,
                     precision = np.float32,
                     detailed_loss = {},
                     training_dataset = None,
                     validation_dataset = None,
                     samples_per_second = None,
                     args = None):
        if validation_accuracy is None:
            validation_accuracy = -1
        if validation_batchsize is None:
            validation_batchsize = 0
        if validation_loss is None:
            validation_loss  = np.finfo(np.float32).max
        
        # INITIALIZE NEW STATE
        if self.state is None:
            self.state = {  
                             "model_desc" : str(model),
                             "model_trainable_params" : Torchutils.count_parameters(model),
                             "loss_desc"  : str(loss_function),
                             "cuda" : next(model.parameters()).is_cuda, 
                             "regression" : regression,
                             "precision" : precision,
                             "dataset_training" : Torchutils.get_data_information(training_dataset),
                             "dataset_validation" : Torchutils.get_data_information(validation_dataset),
                             "samples_per_second[]" : [(epoch, samples_per_second)],
                             "init_optimizer_state" : optimizer.state_dict(),
                             "lrate[]" : [(epoch, optimizer.state_dict()["param_groups"][0]["lr"])],
                             "model_state" : model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                             "best_training_loss" : (epoch, training_loss, training_batchsize),
                             "best_training_accuracy" : (epoch, training_accuracy, training_batchsize),
                             "training_loss[]" : [(epoch, training_loss, training_batchsize)],
                             "training_accuracy[]" : [(epoch, training_accuracy, training_batchsize)],
                             "best_validation_loss" : (epoch, validation_loss, validation_batchsize),
                             "best_validation_accuracy" : (epoch, validation_accuracy, validation_batchsize),
                             "validation_loss[]" : [(epoch, validation_loss, validation_batchsize)],
                             "validation_accuracy[]" : [(epoch, validation_accuracy, validation_batchsize)],
                             "detailed_loss[]" : [(epoch, detailed_loss)],
                             "args[]" : [args]
                          }
        else:
            # APPEND STATE
            if "dataset_training" in self.state:
                dataset_training_info = self.state["dataset_training"] 
            else: 
                dataset_training_info = Torchutils.get_data_information(training_dataset)
            if "dataset_validation" in self.state:
                dataset_validation_info = self.state["dataset_validation"] 
            else:
                dataset_validation_info = Torchutils.get_data_information(validation_dataset)                
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
            if validation_loss < self.state["best_validation_loss"][1]:
                best_validation_loss = (epoch, validation_loss, validation_batchsize) 
            else: 
                best_validation_loss = self.state["best_validation_loss"]
            if validation_accuracy > self.state["best_validation_accuracy"][1]:
                best_validation_acc = (epoch, validation_accuracy, validation_batchsize) 
            else:
                best_validation_acc  = self.state["best_validation_accuracy"]
            if "detailed_loss[]" in self.state:
                detailed_loss_info = [*self.state["detailed_loss[]"], (epoch, detailed_loss)] 
            else:
                detailed_loss_info = [(epoch, detailed_loss)]
            self.state = {  
                             "model_desc" : str(model),
                             "model_trainable_params" : Torchutils.count_parameters(model),
                             "loss_desc"  : str(loss_function),
                             "cuda" : next(model.parameters()).is_cuda, 
                             "regression" : regression,
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
                             "validation_loss[]" : [*self.state["validation_loss[]"], (epoch, validation_loss, validation_batchsize)],
                             "validation_accuracy[]" : [*self.state["validation_accuracy[]"], (epoch, validation_accuracy, validation_batchsize)],
                             "detailed_loss[]" : detailed_loss_info,
                             "args[]" : [*self.state["args[]"]]
                          }
        
    def get_summary(self):
        summary = {
                    "Epochs"               : self.state["training_loss[]"][-1][0],
                    "Best Training Loss"   : self.state["best_training_loss"],
                    "Best Validation Loss" : self.state["best_validation_loss"],
                  }
        return summary   
    
           
  
    def load_state(self, filename = None, force_weights_to_cpu = True):
        if filename is None:
            filename = self.filename
        
        if force_weights_to_cpu:
            self.state = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            self.state = torch.load(filename)        
            
        
    def save_state(self, filename = None, keep_epochs = False):
        if filename is None:
            filename = self.filename
            short_filename = self.short_filename
        else:
            if self.extension not in filename:
                filename = str(filename + self.extension)
            short_filename = filename.split(self.extension)[0]
            self.short_filename = short_filename
            self.filename = filename
        assert filename is not None
        
        file_extension = self.extension
        if keep_epochs:
            epoch = self.state["lrate[]"][-1][0]
            filename = short_filename + "_epoch_" + str(epoch) + file_extension
            torch.save(self.state, filename)
        else:
            filename = short_filename + file_extension
            torch.save(self.state, filename)  
        
        def is_best_train(state):
            return state["training_loss[]"][-1][1] == state["best_training_loss"][1]
        
        def is_best_valid(state):
            return state["validation_loss[]"][-1][1] == state["best_validation_loss"][1]
        
        if is_best_train(self.state):
            shutil.copyfile(filename, str(short_filename + self.train_pattern + file_extension))

        if is_best_valid(self.state):
            shutil.copyfile(filename, str(short_filename + self.valid_pattern + file_extension))
            
     
    def __repr__(self):
        return str(self.state)
    
    
    def __str__(self):
        return str(self.state)
    
    
    def __getitem__(self, item):
         return self.state[item]
    
    
    def reset_to_best_validation_model(self, model, filename = None):
        assert self.state is not None
        assert isinstance(model, nn.Module)
        if filename is None:
            filename = self.filename
            short_filename = self.short_filename
        else:
            if self.extension not in filename:
                filename = str(filename + self.extension)
            short_filename = filename.split(self.extension)[0]
            
        self.load_state(str(short_filename + self.valid_pattern + self.extension), self.force_weights_to_cpu)
        
        # RESTORE STATE    
        model.load_state_dict(self.state["model_state"])            
       
        return model    
    
    def reset_to_best_training_model(self, model, filename = None):
        assert self.state is not None
        assert isinstance(model, nn.Module)
        if filename is None:
            filename = self.filename
            short_filename = self.short_filename
        else:
            if self.extension not in filename:
                filename = str(filename + self.extension)
            short_filename = filename.split(self.extension)[0]
            
        self.load_state(str(short_filename + self.train_pattern + self.extension), self.force_weights_to_cpu)
        
        # RESTORE STATE    
        model.load_state_dict(self.state["model_state"])            
       
        return model
    
    
    def load_weights(self, model, precision, use_cuda = False):
        assert self.state is not None
        assert isinstance(model, nn.Module)
        assert precision == np.float32 or precision == np.float64
            
        # RESTORE STATE    
        model.load_state_dict(self.state["model_state"])            
       
        # Computational configuration
        return Torchutils.transform_model(model, precision, use_cuda)  
    
    
    def load_optimizer(self, optimizer):
        assert self.state is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
            
        optimizer.load_state_dict(self.state["optimizer_state"])
        
        return optimizer
    