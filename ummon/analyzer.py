##########################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
##########################################################################


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
    def __init__(self, model, training_state):
        self.name = "ummon.Analyzer"
   
    def load(self, model, training_state_filename):
        pass

    def predict(model, dataset):
        pass

if __name__ == "__main__":
    print("This is", Analyzer().name)