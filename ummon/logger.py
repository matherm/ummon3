#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################

import time
from tabulate import tabulate

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
    def __init__(self, direname = None, log_batch_interval=500, log_epoch_interval=1):
        self.name = "ummon.Logger"
        
        # MEMBERS
        self.dirname = direname
        self.filelog = True if direname is not None else False
        self.log_batch_interval = log_batch_interval
        self.log_epoch_interval = log_epoch_interval
            
    def log_one_batch(self, epoch, batch, batches, loss, t):
        if batch % self.log_batch_interval == 0:
            print("\rEpoch: {} - Batch/Batches: {:05}/{:05} - Loss: {:04.5f}. [{:02} s] ".format(
                                                                                epoch,
                                                                                batch, 
                                                                                batches, 
                                                                                loss.data[0],
                                                                                int(time.time() - t)),end='')
            sys.stdout.flush()


    def log_epoch(self, epoch, batch, batches, loss, batchsize, t):
         if epoch % self.log_epoch_interval == 0:
             print("\rEpoch: {} - Batch/Batches: {:05}/{:05} - Loss: {:04.5f}. [{:02} s ({} samples/s)] ".format(
                                                                        epoch,
                                                                        batch,
                                                                        batches, 
                                                                        loss.data[0],
    	                                			                    int(time.time() - t),
                                                                        int((batches * batchsize/(time.time() - t)))))
    
    def log_evaluation(self, learningstate, samples_per_seconds):
        epoch = learningstate["training_loss[]"][-1][0]
        lrate = learningstate["lrate[]"][-1][1]
        loss =  learningstate["validation_loss[]"][-1][1]
        acc  =  learningstate["validation_accuracy[]"][-1][1]
        batchsize = learningstate["validation_accuracy[]"][-1][2]
        is_best = learningstate["validation_accuracy[]"][-1][1] == learningstate["best_validation_accuracy"][1]
        
        print("\r" , " " * 125)
        print("Model Evaluation","Epoch #", epoch, "lrate :", lrate)
        print("------------------------------------------")  
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Error: {:.2f}%.'.format(loss, acc * batchsize, batchsize, acc, 1. - acc, "[BEST]" if is_best else None))
        print("Loss: {:04.5f}".format(loss))
        print('Throughput is:', samples_per_seconds, "samples/s")

            
    def print_args(self, args):
        print("\n[Arguments]")
        table = [[ arg, getattr(args, arg)] for arg in vars(args)]
        print(tabulate(table, headers=['Key', 'Value']))
        

    def print_problem_summary(self, model, loss_function, optimizer, dataloader_train, dataset_validation = None, dataset_test = None):
        print("\n[Parameters]")
        table_params = [["lrate" , optimizer.state_dict()["param_groups"][0]["lr"]],
                        ["using_cuda"  , "True" if next(model.parameters()).is_cuda else "False"],
                        ["batch_size" , dataloader_train.batch_size]]
        print(tabulate(table_params, headers=['Key', 'Value']))
        
        print("\n[Model]")
        print(model)
        
        print("\n[Loss]")
        print(loss_function)
        
        print("\n[Data]")
        table_data = [['Training'  , len(dataloader_train.dataset) if dataloader_train is not None else 0, str(dataloader_train.dataset[0][0].numpy().shape) if dataloader_train is not None else "---"], 
                      ['Validation', len(dataset_validation) if dataset_validation is not None else 0, str(dataset_validation[0][0].numpy().shape) if dataset_validation is not None else "---"],
                      ['Test'      , len(dataset_test)  if dataset_test is not None else 0,  str(dataset_test[0][0].numpy().shape)] if dataset_test is not None else "---"]
        print(tabulate(table_data, headers=['Dataset', 'Samples', "Dimensions (per sample)"]))

        
if __name__ == "__main__":
    print("This is", Logger().name)