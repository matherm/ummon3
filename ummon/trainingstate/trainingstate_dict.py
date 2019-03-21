import os
import torch
import torch.nn as nn
import shutil
import numpy as np
import ummon.utils as uu
from collections import OrderedDict

__all__ = [ "TrainingStateDict" ]

class TrainingStateDict():
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
    
    def __init__(self):        
        self.state = None
            
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
            self.state = self._new_state(epoch, model, loss_function, optimizer, training_loss, 
                                            training_accuracy = training_accuracy,
                                            training_dataset = training_dataset,
                                            training_batchsize = training_batchsize,
                                            trainer_instance = trainer_instance,
                                            precision = precision,
                                            detailed_loss = detailed_loss,
                                            validation_loss = validation_loss, 
                                            validation_accuracy = validation_accuracy, 
                                            validation_dataset = validation_dataset,
                                            samples_per_second = samples_per_second,
                                            scheduler = scheduler,
                                            combined_retraining = combined_retraining,
                                            args = args)
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
                         "id" : self.state["id"],
                         "args[]" : args
                          }

    def _new_state(self, 
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
            state = {  
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
                         "id" : hash(np.random.uniform(0,1000000)),
                         "args[]" : [args]
                          }
            return state


    def __repr__(self):
        formatted_string = [str(str(key) + "\t" + str(value) + "\n") for key,value in sorted(self.state.items())]
        return str(''.join(formatted_string))
    
    
    def __str__(self):
        formatted_string = [str(str(key) + "\t" + str(value) + "\n") for key,value in sorted(self.state.items())]
        return str(''.join(formatted_string))
    
    
    def __getitem__(self, item):
         return self.state[item]

    def current_validation_loss(self):
        if len(self.state['validation_loss[]']) == 0:
            return None
        return self.state['validation_loss[]'][-1][1]

    def current_training_loss(self):
        return self.state["training_loss[]"][-1][1]

    def current_validation_acc(self):
        if len(self.state['validation_loss[]']) == 0:
            return None
        return self.state['validation_accuracy[]'][-1][1]
        
    def current_training_acc(self):
        return self.state["training_accuracy[]"][-1][1]

    def is_best_validation_model(self):
        if self.state is None:
            return False
        if len(self.state["validation_loss[]"]) == 0:
                return False
        is_best = self.current_validation_loss() == self.best_validation_loss()
        return is_best
        
    def is_best_training_model(self):
        if self.state is None:
            return False
        if len(self.state["training_loss[]"]) == 0:
                return False
        is_best = self.current_training_loss() == self.best_training_loss()
        return is_best
        
    def best_validation_loss(self):
        return self.state["best_validation_loss"][1]         
  
    def best_training_loss(self):
        return self.state["best_training_loss"][1]         

    def best_training_acc(self):
        return self.state["best_training_acc"][1]         
  
    def best_validation_acc(self):
        return self.state["best_validation_acc"][1]         
            
    def current_lrate(self):
        return self.state['lrate[]'][-1][1]

    def current_epoch(self):
        return self.state['lrate[]'][-1][0]

    def has_validation_data(self):
        return self.state["validation_loss[]"] == []

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
 
from ..schedulers.early_stopping import StepLR_earlystop