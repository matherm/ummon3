#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################


class Visualizer:
    def __init__(self):
        self.name = "ummon.Visualizer"

if __name__ == "__main__":
    print("This is", Visualizer().name)
