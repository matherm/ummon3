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
    logfile :               String
                            The directory used for saving logs (if NULL logs are only printed to console)       
    log_batch_interval :    int
                             Specifies how often information about mini-batches are logged.   
             
    Methods
    -------
    log_one_batch()         :   Logs information about a single mini-batch
    log_epoch()             :   Logs information about a single epoch
    log_evaluation()        :   Logs information about an model evaluation
    info()                  :   Logs information with Tag [INFO]
    error()                 :   Logs errors with Tag [ERROR]
    print_args()            :   Pretty print for argument dicts
    print_problem_summary() :   Pretty print of a learning problem including learning rate, epochs, etc.
         
    """
    def __init__(self, logfile = None, log_batch_interval=500):
        self.name = "ummon.Logger"
        
        # MEMBERS
        self.logfile = logfile
        self.filelog = True if logfile is not None else False
        self.log_batch_interval = log_batch_interval
            
    def log_one_batch(self, epoch, batch, batches, loss, t):
        if batch % self.log_batch_interval == 0:
            print("\r[INFO] Epoch: {} - Batch/Batches: {:05}/{:05} - Loss: {:04.5f}. [{:02} s] ".format(
                                                                                epoch,
                                                                                batch, 
                                                                                batches, 
                                                                                loss,
                                                                                int(time.time() - t)),end='')
            sys.stdout.flush()
            if self.filelog:
                with open(self.logfile, "a") as logfile:
                     print("[INFO] Epoch: {} - Batch/Batches: {:05}/{:05} - Loss: {:04.5f}. [{:02} s] ".format(
                                                                                epoch,
                                                                                batch, 
                                                                                batches, 
                                                                                loss,
                                                                                int(time.time() - t)),file=logfile)
           


    def log_epoch(self, epoch, batch, batches, loss, batchsize, t):
        print("\r[INFO] Epoch: {} - Batch/Batches: {:05}/{:05} - Loss: {:04.5f}. [{:02} s ({} samples/s)] ".format(
                                                                        epoch,
                                                                        batch,
                                                                        batches, 
                                                                        loss,
    	                                			                    int(time.time() - t),
                                                                        int((batches * batchsize/(time.time() - t)))))
        if self.filelog:
                with open(self.logfile, "a") as logfile:
                    print("[INFO] Epoch: {} - Batch/Batches: {:05}/{:05} - Loss: {:04.5f}. [{:02} s ({} samples/s)] ".format(
                                                                        epoch,
                                                                        batch,
                                                                        batches, 
                                                                        loss,
    	                                			                    int(time.time() - t),
                                                                        int((batches * batchsize/(time.time() - t)))), file=logfile)
    
    def log_evaluation(self, learningstate, samples_per_seconds):
        epoch = learningstate.state["training_loss[]"][-1][0]
        lrate = learningstate.state["lrate[]"][-1][1]
        loss =  learningstate.state["validation_loss[]"][-1][1]
        acc  =  learningstate.state["validation_accuracy[]"][-1][1]
        batchsize = learningstate.state["validation_accuracy[]"][-1][2]
        is_best = learningstate.state["validation_accuracy[]"][-1][1] == learningstate.state["best_validation_accuracy"][1]
        
        print("\n[EVAL] Model Evaluation","Epoch #", epoch, "lrate :", lrate)
        print("       ----------------------------------------")  
        print('       Test set: loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Error: {:.2f}%.'.format(loss, int(acc * batchsize), batchsize, acc * 100, (1. - acc) * 100), "[BEST]" if is_best else None)
        print('       Throughput is {:.0f} samples/s\n'.format(samples_per_seconds))

        if self.filelog:
                with open(self.logfile, "a") as logfile:
                    print("\n[EVAL] Model Evaluation","Epoch #", epoch, "lrate :", lrate, file=logfile)
                    print("       ----------------------------------------", file=logfile)  
                    print('       Test set: loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Error: {:.2f}%.'.format(loss, int(acc * batchsize), batchsize, acc * 100, (1. - acc) * 100), "[BEST]" if is_best else None, file=logfile)
                    print('       Throughput is {:.0f} samples/s\n'.format(samples_per_seconds), file=logfile)


    def info(self, text):
        print("[INFO]", text)
        
        if self.filelog:
                with open(self.logfile, "a") as logfile:
                    print("[INFO]", text, file=logfile)
        
    def error(self, text):
        print("[ERROR]", text)
        
        if self.filelog:
                with open(self.logfile, "a") as logfile:
                    print("[ERROR]", text, file=logfile)

        
    def print_args(self, args):
        table = [[ arg, getattr(args, arg)] for arg in vars(args)]
        print("\n[Arguments]")
        print(tabulate(table, headers=['Key', 'Value']))
       
        if self.filelog:
            with open(self.logfile, "a") as logfile:
                print("\n[Arguments]", file=logfile)
                print(tabulate(table, headers=['Key', 'Value']), file=logfile)



    def print_problem_summary(self, model, loss_function, optimizer, dataloader_train, dataset_validation = None, epochs = 0, early_stopping = False, dataset_test = None):
        table_params = [["lrate" , optimizer.state_dict()["param_groups"][0]["lr"]],
                        ["batch_size" , dataloader_train.batch_size],
                        ["epochs" , epochs],
                        ["using_cuda"  , "True" if next(model.parameters()).is_cuda else "False"],
                        ["early_stopping" , early_stopping]]

        table_data = [['Training'  , len(dataloader_train.dataset) if dataloader_train is not None else 0, str(dataloader_train.dataset[0][0].numpy().shape) if dataloader_train is not None else "---"], 
                      ['Validation', len(dataset_validation) if dataset_validation is not None else 0, str(dataset_validation[0][0].numpy().shape) if dataset_validation is not None else "---"],
                      ['Test'      , len(dataset_test)  if dataset_test is not None else 0,  str(dataset_test[0][0].numpy().shape)] if dataset_test is not None else "---"]

        print("\n[Parameters]")
        print(tabulate(table_params, headers=['Key', 'Value']))
        
        print("\n[Model]")
        print(model)
        
        print("\n[Loss]")
        print(loss_function)
        
        print("\n[Data]")
        print(tabulate(table_data, headers=['Dataset', 'Samples', "Dimensions (per sample)"]))

        if self.filelog:
            with open(self.logfile, "a") as logfile:
                print("\n[Parameters]", file=logfile)
                print(tabulate(table_params, headers=['Key', 'Value']), file=logfile)
                
                print("\n[Model]", file=logfile)
                print(model, file=logfile)
                
                print("\n[Loss]", file=logfile)
                print(loss_function, file=logfile)
                
                print("\n[Data]", file=logfile)
                print(tabulate(table_data, headers=['Dataset', 'Samples', "Dimensions (per sample)"]), file=logfile)


        
if __name__ == "__main__":
    print("This is", Logger().name)