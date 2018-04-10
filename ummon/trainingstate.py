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
    
    def __init__(self, filename = None, force_weights_to_cpu = False):
        
        self.state = None
        self.extension = ".pth.tar"
        
        if not filename is None:
            self.load_state(filename, force_weights_to_cpu)
    
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
        if self.state is None:
            self.state = {  
                             "model_desc" : str(model),
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
             self.state = {  
                             "model_desc" : str(model),
                             "loss_desc"  : str(loss_function),
                             "cuda" : self.state["cuda"], 
                             "regression" : regression,
                             "precision" : precision,
                             "dataset_training" : self.state["dataset_training"] if "dataset_training" in self.state else Torchutils.get_data_information(training_dataset),
                             "dataset_validation" : self.state["dataset_validation"] if "dataset_validation" in self.state else Torchutils.get_data_information(validation_dataset),
                             "samples_per_second[]" : [*self.state["samples_per_second[]"], (epoch, samples_per_second)] if "samples_per_second[]" in self.state else [(epoch, samples_per_second)],
                             "init_optimizer_state" : self.state["init_optimizer_state"],
                             "lrate[]" : [*self.state["lrate[]"], (epoch, optimizer.state_dict()["param_groups"][0]["lr"])],
                             "model_state" : model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                             "best_training_loss" : (epoch, training_loss, training_batchsize) if training_loss< self.state["best_training_loss"][1] else self.state["best_training_loss"],
                             "best_training_accuracy" : (epoch, training_accuracy, training_batchsize) if training_accuracy > self.state["best_training_accuracy"][1] else self.state["best_training_accuracy"],
                             "training_loss[]" : [*self.state["training_loss[]"], (epoch, training_loss, training_batchsize)],
                             "training_accuracy[]" : [*self.state["training_accuracy[]"], (epoch, training_accuracy, training_batchsize)],
                             "best_validation_loss" : (epoch, validation_loss, validation_batchsize) if validation_loss < self.state["best_validation_loss"][1] else self.state["best_validation_loss"],
                             "best_validation_accuracy" : (epoch, validation_accuracy, validation_batchsize) if validation_accuracy > self.state["best_validation_accuracy"][1] else self.state["best_validation_accuracy"],
                             "validation_loss[]" : [*self.state["validation_loss[]"], (epoch, validation_loss, validation_batchsize)],
                             "validation_accuracy[]" : [*self.state["validation_accuracy[]"], (epoch, validation_accuracy, validation_batchsize)],
                             "detailed_loss[]" : [*self.state["detailed_loss[]"], (epoch, detailed_loss)] if "detailed_loss[]" in self.state else [(epoch, detailed_loss)],
                             "args[]" : [*self.state["args[]"], args]
                          }
        
    def get_summary(self):
        summary = {
                    "Epochs"               : self.state["training_loss[]"][-1][0],
                    "Best Training Loss"   : self.state["best_training_loss"],
                    "Best Validation Loss" : self.state["best_validation_loss"],
                  }
        return summary   
    
           
  
    def load_state(self, filename, force_weights_to_cpu = False):
        if force_weights_to_cpu:
            self.state = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            self.state = torch.load(filename)        
            
        
    def save_state(self, filename = "model", keep_epochs = False):
        if self.extension not in filename:
            filename = str(filename + self.extension)
        short_filename = filename.split(self.extension)[0]
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
            shutil.copyfile(filename, str(short_filename + '_best_training_loss' + file_extension))

        if is_best_valid(self.state):
            shutil.copyfile(filename, str(short_filename + '_best_validation_loss' + file_extension))
            
     
    def __repr__(self):
        return str(self.state)
    
    
    @staticmethod
    def initialize_model(logger, model, trainingstate, precision, use_cuda = False):
        assert isinstance(model, nn.Module)
        assert precision == np.float32 or precision == np.float64
        
        if trainingstate is not None:
            assert isinstance(trainingstate, Trainingstate)
            
            # RESTORE STATE    
            model.load_state_dict(trainingstate.state["model_state"])            
           
            # SANITY CHECK
            assert precision == trainingstate.state["precision"]
        
        # Computational configuration
        if precision == np.float32:
            model = model.float()
        if precision == np.float64:
            model = model.double()
        if use_cuda:
            if not torch.cuda.is_available():
                logger.error('CUDA is not available on your system.')
            model = model.cuda()
        else:
            model = model.cpu()
        return model