##########################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
##########################################################################

class Logger:
    def __init__(self):
        self.name = "ummon.Logger"
        
    
    def print_training_state(self):
        pass
    
        
        
if __name__ == "__main__":
    print("This is", Logger().name)