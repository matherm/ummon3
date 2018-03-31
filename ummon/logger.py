#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import logging, warnings, os, time, socket
from platform import platform
import numpy as np
import torch
from .__version__ import version

__all__ = [ 'Logger' ]

# formatter for colored messages
class _ColoredFormatter(logging.Formatter):
    '''
    Formatter needed for displaying colored log messages. Intended for internal use only.
    '''
    # types required for colored output
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    COLORS = {
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': MAGENTA
    }
    
    def __init__(self, msg):
        logging.Formatter.__init__(self, msg)
    
    def format(self, record):
        levelname = record.levelname
        message = record.msg
        if levelname in self.COLORS:
            message_color = self.COLOR_SEQ % (30 + self.COLORS[levelname]) + message + self.RESET_SEQ
            record.msg = message_color
        return logging.Formatter.format(self, record)


# logger
class Logger(logging.getLoggerClass()):
    '''
    Logger used for Ummon::
    
    Parameters:
    
    * name: Logger name (default: 'logger')
    * loglevel: one of the standard log levels of the logging module (default: 
      logging.DEBUG or 10)
    * logdir: directory for the logfiles. When set to '', nothing is logged to file.
    
    Messages can be displayed using the methods error(), warn(), info() and debug().
    Warnings are output in yellow, errors in red using the methods warn(message) and 
    error(message). warn() uses the warning mechanism from the standard warnings module
    which are rerouted to the logger. error(message exception) throws an exception after 
    displaying the message the type of which can be provided as second argument (default:
    Exception). User level and debug information are displayed using info(message) and
    debug(message). All messages can be disabled by setting the loglevel appropriately.
    The logger supports additional logging to a file which is started using the method
    start_logfile() and stopped by stop_logfile(). The logfile records all
    levels, independently of what is set for console output.
    '''
    
    def __init__(self, name='ummon', loglevel=logging.DEBUG, logdir=''):
        self.name = str(name)
        self.loglevel = int(loglevel)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.loglevel)
        self.logdir = str(logdir)
    
    # setup logging
    def __enter__(self):
        ch = logging.StreamHandler()
        ch.setLevel(self.loglevel)
        formatter = _ColoredFormatter('%(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False    # do not propagate messages to ancestor handlers (avoids  
                                    # double output on consoles/notebooks)
        warnings.showwarning = self.send_warnings_to_log
        
        # start logging to file
        if self.logdir != '':
            self.start_logfile()
        
        # ummon startup message
        self.info('ummon               Ver.{}, IOS Konstanz'.format(version))
        
        return self
    
    # stop logging
    def __exit__(self, *err):
        
        # stop logging to file
        if self.logdir != '':
            self.stop_logfile()
        
        ch = self.logger.handlers[0]
        self.logger.removeHandler(ch)
    
    # reroute warnings to log
    def send_warnings_to_log(self, message, category, filename, lineno, file=None, line=None):
        self.logger.warning(str(message))
        return
    
    # error
    def error(self, msg='', errtype=Exception):
        err_msg = 'ERROR: ' + msg
        self.logger.error(err_msg)
        raise errtype(err_msg)
    
    # warning
    def warn(self, msg=''):
        warn_msg = 'WARNING: ' + msg
        warnings.warn(warn_msg)
    
    # info
    def info(self, msg=''):
        self.logger.info(msg)
    
    # debug
    def debug(self, msg=''):
        self.logger.debug(msg)
    
    # set logging directory
    def set_logdir(self, pathname):
        
        # check whether logging directory exists
        if not os.path.exists(pathname):
            raise IOError('Path to logging directory does not exist.')
        self.logdir = pathname
    
    # start logging to file
    def start_logfile(self):
        fname = os.path.join(self.logdir, 
            'ummon_{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
        file_handler = logging.FileHandler(fname)
        formatter = logging.Formatter("%(asctime)s: [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.debug('SYSTEM INFORMATION ({})'.format(socket.gethostname()))
        self.debug('Platform {}'.format(platform()))
        self.debug('CUDA {}'.format(torch.version.cuda))
        self.debug('CuDNN {}'.format(torch.backends.cudnn.version()))
        self.debug('Python {}'.format(sys.version.split('\n')))
        self.debug('Numpy {}'.format(np.__version__))
        self.debug('Torch {}'.format(torch.__version__))
    
    # stop logging to file    
    def stop_logfile(self):
        if len(self.logger.handlers) < 2: # called before file logging started
            return # do nothing in this case
        self.logger.handlers[-1].stream.close()
        self.logger.removeHandler(self.logger.handlers[-1])

import time
from tabulate import tabulate

class Logger2:
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
    print("This is", Logger2().name)