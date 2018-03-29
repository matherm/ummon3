##########################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
##########################################################################

class Logger:
    """
    This class provides a generic Logger for logging training information to file/console.
    
    Constructor
    -----------
    direname : String
               The directory used for saving logs (if NULL logs are only printed to console)       
             
    Methods
    -------
        
         
    """
    def __init__(self, direname):
        self.name = "ummon.Logger"
        
        if direname:
            self.dirname = direname
            self.filelog = True
        else:
            self.filelog = False
            
    def log_to_file(self, training_state):
        pass
    
    def log_to_console(self, training_state):
        pass
    
        
        
if __name__ == "__main__":
    print("This is", Logger().name)