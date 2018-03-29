##########################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
##########################################################################


class Trainingstate():
    
    def __init__():
        


     training_state = {"args" : args, 
                         "error[]" : [error],
                         "classification_loss[]" : [classification_loss.data[0]],
                         "policy_loss[]": [_policy_loss.data[0]],
                         "value_loss[]" : [_value_loss.data[0]],
                         "total_loss[]" : [total_loss],
                         "best_total_loss" : total_loss,
                         "best_error" : error,
                         "avg_scale[]" : [avg_scale],
                         "avg_locx[]" : [avg_x_loc],
                         "avg_locy[]" : [avg_y_loc],
                         "model" : model.state_dict(),}
    training_state = { "args" : args, 
                         "error[]" : [*learning_state["error[]"], error], 
                         "classification_loss[]" : [*learning_state["classification_loss[]"], classification_loss.data[0]],
                         "policy_loss[]": [*learning_state["policy_loss[]"], _policy_loss.data[0]],
                         "value_loss[]" : [*learning_state["value_loss[]"], _value_loss.data[0]],
                         "total_loss[]" : [*learning_state["total_loss[]"], total_loss],
                         "best_total_loss" : total_loss if total_loss < learning_state["best_total_loss"] else learning_state["best_total_loss"],
                         "best_error" : error if error < learning_state["best_error"] else learning_state["best_error"],
                         "avg_scale[]" : [*learning_state["avg_scale[]"], avg_scale],
                         "avg_locx[]" : [*learning_state["avg_locx[]"], avg_x_loc],
                         "avg_locy[]" : [*learning_state["avg_locy[]"], avg_y_loc],
                         "model" : model.state_dict(),}


