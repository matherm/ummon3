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
import argparse
parser = argparse.ArgumentParser(description='ummon3 - example - trainingstate checker')
parser.add_argument('--view', default="", metavar="",
                    help="Print summary about a trained model")
argv = parser.parse_args()
sys.argv = [sys.argv[0]]                    

#
# IMPORTS
from ummon.trainingstate import Trainingstate


if __name__ == "__main__":
    
    if argv.view is not "":
        ts = Trainingstate(argv.view)
        print(ts.get_summary())
    else:
        print("No model was given..")
