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
    
    def __init__(self):
        
        self.state = None
    
    def update_state(self, model, optimizer, epoch, iteration, loss, accuracy):
        if self.state is None:
            self.state = {  
                             "model_desc" : str(model),
                             "init_optimizer_state" : optimizer.state_dict(),
                             "best_loss" : (epoch, loss),
                             "best_accuracy" : (epoch, accuracy),
                             "cuda" : next(model.parameters()).is_cuda, 
                             "loss[]" : [(epoch, loss)],
                             "lrate[]" : [optimizer.state_dict()["param_groups"][0]["lr"]],
                             "accuracy[]" : [(epoch, accuracy)],
                             "model_state" : model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                          }
        else:
             self.state = {  
                             "model_desc" : str(model),
                             "init_optimizer_state" : self.state["init_optimizer_state"],
                             "best_loss" : (epoch, loss) if loss < self.state["best_loss"][1] else self.state["best_loss"],
                             "best_accuracy" : (epoch, accuracy) if accuracy < self.state["best_accuracy"][1] else self.state["best_accuracy"],
                             "cuda" : self.state["cuda"], 
                             "loss[]" : [*self.state["loss[]"], (epoch, loss)],
                             "lrate[]" : [*self.state["lrate[]"], optimizer.state_dict()["param_groups"][0]["lr"]],
                             "accuracy[]" : [*self.state["accuracy[]"], (epoch, accuracy)],
                             "model_state" : model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                          }
        
    def load_state(self, filename, force_cpu = False):
        if force_cpu:
            self.state = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            self.state = torch.load(filename)        
        
    def save_state(self, filename):
        torch.save(self.state, filename)
        
        def is_best(state):
            return state["loss"][-1] == state["best_loss"]
        
        if is_best(self.state):
            shutil.copyfile(filename, 'model_best.pth.tar')

        
