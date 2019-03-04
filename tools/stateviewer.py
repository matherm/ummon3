"""

ummon3 Tools

Check the trainingstate of a learned model

Run command:
    
    python -m ummon.tools.stateviewer MNIST1.pth.tar

"""

#
# IMPORTS
from ummon.trainingstate import Trainingstate
import pprint
import sys

class DefaultValues(dict):
    def __init__(self):
        dict.__init__(self, {
                        "model" : "MNIST1",
                        "verbose" : False
                        })
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__  


def view(model, verbose = False):
   
    if model is not "":
        ts = Trainingstate(model)
        if verbose == True:
            # DELETE MODEL PARAMETERS
            ts.state.pop("model_state" , None)
            # PRETTY PRINT
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(ts.state)
        else:
            print(ts.get_summary())
    else:
        raise Exception("No model was given..") 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ummon3 - tools - trainingstate checker')
    parser.add_argument('model', default="", metavar="",
                        help="Print summary about a trained model")
    parser.add_argument('-v', action='store_true', dest='verbose', 
                        help="Verbose (default: False)")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]                    
    
    view(argv.model, argv.verbose)