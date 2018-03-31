##################################################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################
import torch
import shutil

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
        
        if not filename is None:
            self.load_state(filename, force_weights_to_cpu)
    
    def update_state(self, 
                     epoch, model, loss_function, optimizer, training_loss, 
                     validation_loss = None, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = None, 
                     validation_batchsize = 0,
                     args = None):
        
        if self.state is None:
            self.state = {  
                             "model_desc" : str(model),
                             "loss_desc"  : str(loss_function),
                             "cuda" : next(model.parameters()).is_cuda, 
                             "init_optimizer_state" : optimizer.state_dict(),
                             "lrate[]" : [(epoch, optimizer.state_dict()["param_groups"][0]["lr"])],
                             "model_state" : model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                             "best_training_loss" : (epoch, training_loss),
                             "best_training_accuracy" : (epoch, training_accuracy),
                             "training_loss[]" : [(epoch, training_loss)],
                             "training_accuracy[]" : [(epoch, training_accuracy, training_batchsize)],
                             "best_validation_loss" : (epoch, validation_loss),
                             "best_validation_accuracy" : (epoch, validation_accuracy),
                             "validation_loss[]" : [(epoch, validation_loss)],
                             "validation_accuracy[]" : [(epoch, validation_accuracy, validation_batchsize)],
                             "args[]" : [args]
                          }
        else:
             self.state = {  
                             "model_desc" : str(model),
                             "loss_desc"  : str(loss_function),
                             "cuda" : self.state["cuda"], 
                             "init_optimizer_state" : self.state["init_optimizer_state"],
                             "lrate[]" : [*self.state["lrate[]"], (epoch, optimizer.state_dict()["param_groups"][0]["lr"])],
                             "model_state" : model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                             "best_training_loss" : (epoch, training_loss) if training_loss< self.state["best_training_loss"][1] else self.state["best_training_loss"],
                             "best_training_accuracy" : (epoch, training_accuracy, training_batchsize) if training_accuracy > self.state["best_training_accuracy"][1] else self.state["best_training_accuracy"],
                             "training_loss[]" : [*self.state["training_loss[]"], (epoch, training_loss)],
                             "training_accuracy[]" : [*self.state["training_accuracy[]"], (epoch, training_accuracy, training_batchsize)],
                             "best_validation_loss" : (epoch, validation_loss) if validation_loss < self.state["best_validation_loss"][1] else self.state["best_validation_loss"],
                             "best_validation_accuracy" : (epoch, validation_accuracy, validation_batchsize) if validation_accuracy > self.state["best_validation_accuracy"][1] else self.state["best_validation_accuracy"],
                             "validation_loss[]" : [*self.state["validation_loss[]"], (epoch, validation_loss)],
                             "validation_accuracy[]" : [*self.state["validation_accuracy[]"], (epoch, validation_accuracy, training_batchsize)],
                             "args[]" : [*self.state["args[]"], args]
                          }
        
    def get_summary(self):
        summary = {
                    "Epochs"                : self.state["training_loss[]"][-1][0],
                    "Best Training Loss"   : self.state["best_training_loss"],
                    "Best Validation Loss" : self.state["best_validation_loss"],
                  }
        return summary    
        
    def load_state(self, filename, force_weights_to_cpu = False):
        if force_weights_to_cpu:
            self.state = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            self.state = torch.load(filename)        
            
        
    def save_state(self, filename = "model.pth.tar", keep_epochs = False):
        if ".pth.tar" not in filename:
            filename = str(filename + ".pth.tar")
        if keep_epochs:
            epoch = self.state["lrate[]"][-1][0]
            filename_epoch = str(filename + "_epoch_" + epoch)
            torch.save(self.state, filename_epoch)
        else:
            torch.save(self.state, filename)  
        
        def is_best_train(state):
            return state["training_loss[]"][-1] == state["best_training_loss"]
        
        def is_best_valid(state):
            return state["validation_loss[]"][-1] == state["best_validation_loss"]
        
        if is_best_train(self.state):
            shutil.copyfile(filename, str(filename + '_best_training_loss.pth.tar'))

        if is_best_valid(self.state):
            shutil.copyfile(filename, str(filename + '_best_validation_loss.pth.tar'))
