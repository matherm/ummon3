#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################

"""

ummon3 Examples

Check the trainingstate of a learned model

Run command:
    
    python check-trainingstate.py --view MNIST1.pth.tar

"""

#
# IMPORTS
from ummon.trainingstate import Trainingstate
import pprint

class DefaultValues(dict):
    def __init__(self):
        dict.__init__(self, {
                        "view" : "",
                        "verbose" : False
                        })
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__  


def example(argv = DefaultValues()):
   
    if argv.view is not "":
        ts = Trainingstate(argv.view)
        if argv.verbose == True:
            # DELETE MODEL PARAMETERS
            ts.state.pop("model_state" , None)
            # PRETTY PRINT
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(ts.state)
        else:
            print(ts.get_summary())
    else:
        print("No model was given..") 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ummon3 - example - trainingstate checker')
    parser.add_argument('--view', default="", metavar="",
                        help="Print summary about a trained model")
    parser.add_argument('-v', action='store_true', dest='verbose', 
                        help="Verbose (default: False)")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]                    
    
    example(argv)
    
    
