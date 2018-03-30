#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################

import argparse
parser = argparse.ArgumentParser(description='ummon3 Analyzer - For analysing and evaluating models')
parser.add_argument('--state', default="model_best.pth.tar", metavar="",
                    help="The state file (default: model_best.pth.tar)")
parser.add_argument('--plot', action='store_true', dest='plot',
                    help="Shall python plot intermediate tests with matplotlib (default: False)")
argv = parser.parse_args()
sys.argv = [sys.argv[0]]

import time
import torch
import torch.nn as nn
from ummon.trainingstate import Trainingstate
from torch.utils.data import DataLoader
from torch.autograd import Variable

class Analyzer:
    """
    This class provides a generic analyzer for PyTorch-models. For a given PyTorch-model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    
    
    Constructor
    -----------
    model           :   torch.nn.module
                        The torch module to use
    training_state  :   ummon.trainingstate (dictionary)
                        The training state
             
    Methods
    -------
    predict()           :  Predicts any given data
    accuracy()          :  Computes accuracy            
             
    """
    def __init__(self, is_classifier = True):
        self.name = "ummon.Analyzer"
        
        self.is_classifier = is_classifier
    
    def evaluate(self, model, loss_function, dataset):
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(model, nn.Module)

        evaluation_dict = {}
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, sampler=None, batch_sampler=None)  
        for i, data in enumerate(dataloader, 0):
            
                # Take time
                t = time.time()

                # Get the inputs
                inputs, targets = data
                
                # Execute Model
                model.eval()
                output = model(Variable(inputs)).cpu()
                model.train()
                
                # Compute Loss
                targets = Variable(targets)
                loss = loss_function(output, targets).cpu()
                
                # Compute classification accuracy
                if self.is_classifier:
                    classes = self.classify(output)
                    acc = self.compute_accuracy(classes, targets)
                    evaluation_dict["accuracy"] = acc
                
                evaluation_dict["samples_per_seconds"] = dataloader.batch_size / (time.time() - t)
                evaluation_dict["loss"] = loss
                
        return evaluation_dict
   
    @staticmethod
    def classify(output):
        # Get index of class with max probability
        classes = output.data.max(1, keepdim=True)[1] 
        return classes
            
    @staticmethod
    def inference(model, dataset):
        raise NotImplementedError
        pass
   
    @staticmethod
    def predict(model, dataset):
        raise NotImplementedError
        pass
    
    @staticmethod
    def compute_accuracy(classes, targets):
        correct = classes.eq(targets.data.view_as(classes))
        accuracy = 100. * correct.sum() / len(targets)
        return accuracy
    
    @staticmethod
    def compute_loss(model, dataset, loss):
        raise NotImplementedError
        pass
    
    @staticmethod
    def compute_roc(model, dataset):
        raise NotImplementedError
        pass

    @staticmethod
    def get_summary(trainingstate):
        summary = {
                    "Epochs"                : trainingstate["training_loss"][-1][0],
                    "Best Training Error"   : trainingstate["best_training_loss"],
                    "Best Validation Error" : trainingstate["best_validation_loss"],
                    "Best Test Error"       : trainingstate["best_test_loss"]
                  }
        return summary
     
        
if __name__ == "__main__":
    pass
