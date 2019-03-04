import logging, warnings, os, time, socket
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from platform import platform
import ummon
import ummon.utils
import re, sys

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
    def __init__(self, name='ummon.Logger', loglevel=logging.DEBUG, logdir='', filename='ummon',
        log_batch_interval=500, log_epoch_interval=1):
        self.name = str(name)
        self.loglevel = int(loglevel)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.loglevel)
        self.logdir = str(logdir)
        self.filename = str(filename).replace('.log', '')
        log_batch_interval = int(log_batch_interval,)
        self.log_epoch_interval= int(log_epoch_interval) if log_epoch_interval > 0 else 1
        self.log_batch_interval = log_batch_interval if log_batch_interval > 0 else 500
    
    
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
            str(self.filename + '_{}.log').format(time.strftime("%Y%m%d-%H%M%S")))
        file_handler = logging.FileHandler(fname)
        formatter = logging.Formatter("%(asctime)s: [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.debug('[System]')
        self.debug('Host                {}'.format(socket.gethostname()))
        self.debug('Platform            {}'.format(platform()))
        self.debug('CUDA                {}'.format(torch.version.cuda))
        self.debug('CuDNN               {}'.format(torch.backends.cudnn.version()))
        self.debug('Python              {}'.format(sys.version.split('\n')))
        self.debug('Numpy               {}'.format(np.__version__))
        self.debug('Torch               {}'.format(torch.__version__))
        self.debug('Torchvision         {}'.format(torchvision.__version__))
        self.debug('ummon               {}'.format(ummon.version))
        self.debug(' ')
    
    
    # stop logging to file    
    def stop_logfile(self):
        if len(self.logger.handlers) < 2: # called before file logging started
            return # do nothing in this case
        self.logger.handlers[-1].stream.close()
        self.logger.removeHandler(self.logger.handlers[-1])
    
    
    # log one batch # 
    def log_one_batch(self, epoch, batch, batches, loss, batchsize, time_dict):
        if batch % self.log_batch_interval == 0:
            self.debug('Epoch: {} - {:05}/{:05} - Loss: {:04.5f}. [{:3} s]'.format(
                epoch, batch, batches, loss, 
                int(time_dict["total"])))
                
    
    # log at end of epoch
    def log_epoch(self, epoch, batch, batches, loss, batchsize, time_dict, 
        evalstr=None, evaluation_dict = None):
        if epoch % self.log_epoch_interval == 0:
            self.info('Epoch: {} - {}. [{}s] @{} samples/s '.format(
                epoch, evalstr, 
                int(time_dict["total"]),
                int((batches * batchsize/(time_dict["total"])))))
            
            # Detailed loss information comming from either __repr__(loss) or after_eval_hook()
            if evaluation_dict is not None and re.match(".*\d\.\d.*", str(evaluation_dict["detailed_loss"])):
                if type(evaluation_dict["detailed_loss"]) == dict:
                    for k, v in evaluation_dict["detailed_loss"].items():
                        if type(v) == str:
                            self.debug("       >> loss details: {:20} {}".format(k, v))
                        else:
                            self.debug("       >> loss details: {:20} {:.2f}".format(k, v))
                else:
                    self.debug('       __repr__(loss): {}'.format(evaluation_dict["detailed_loss"]))

    
    # output description of learning task
    def print_problem_summary(self, trainer, model, loss_function, optimizer, train_dataset, batch_size,
        valid_dataset = None, epochs = 0, early_stopping = False, combined_retraining = 0, dataset_test = None):
        
        if hasattr(loss_function, "size_average"):
            size_average = loss_function.size_average
        else:
            size_average = None
        
        self.debug(' ')
        self.debug('[Trainer]')
        self.debug(str(type(trainer)).replace("<class '", "").replace("'>", ""))
        
        self.debug(' ')
        self.debug('[Model]')
        for lin in model.__repr__().splitlines():
            self.debug(lin)
        self.debug('{0:20}{1}'.format("Trainable params:", ummon.utils.count_parameters(model)))   
        
        self.debug(' ')
        self.debug('[Loss]')
        for lin in loss_function.__repr__().splitlines():
            self.debug(lin)
        
        self.debug(' ')
        self.debug('[Data]')
        self.debug('{0:18}{1:8}    {2:18} {3} {4:18}'.format('Training', 
            ummon.utils.get_size_information(train_dataset), 
            ummon.utils.get_shape_information(train_dataset), 
            ummon.utils.get_type_information(train_dataset),
            ummon.utils.get_numerical_information(train_dataset)))
        self.debug('{0:18}{1:8}    {2:18} {3} {4:18}'.format('Validation', 
            ummon.utils.get_size_information(valid_dataset), 
            ummon.utils.get_shape_information(valid_dataset), 
            ummon.utils.get_type_information(valid_dataset),
            ummon.utils.get_numerical_information(valid_dataset))) 
        if dataset_test is not None:
            self.debug('{0:18}{1:8}    {2:18} {3} {4:18}'.format('Test', 
                ummon.utils.get_size_information(dataset_test), 
                ummon.utils.get_shape_information(dataset_test), 
                ummon.utils.get_type_information(dataset_test),
                ummon.utils.get_numerical_information(dataset_test)))
          
        self.debug(' ')
        self.debug('[Parameters]')
        self.debug('{0:20}{1:.2e}'.format("lrate" , 
                   optimizer.state_dict()["param_groups"][0]["lr"]))
        self.debug('{0:20}{1}'.format("batch_size" , batch_size))
        self.debug('{0:20}{1}'.format("epochs" , epochs))
        self.debug('{0:20}{1}'.format("combined_retraining" , combined_retraining))
        self.debug('{0:20}{1}'.format("using_cuda"  , next(model.parameters()).is_cuda))
        self.debug('{0:20}{1}'.format("early_stopping" , early_stopping))
        self.debug('{0:20}{1}'.format("precision" , next(model.parameters()).cpu().data.numpy().dtype))
        for i,line in enumerate(optimizer.__repr__().splitlines()):
            if i == 0:
                self.debug('{0:20}{1}'.format("optimizer" , line.replace(" (","")))
            elif i < len(optimizer.__repr__().splitlines()) - 1:
                self.debug('{0:20}{1}'.format("   optimizer-param" , line.replace(" ", "")))
            else:
                pass
     
        if size_average is not None:
            self.debug('{0:20}{1}'.format("size_average" , loss_function.size_average))
            if loss_function.size_average:
                self.warn("Numerical issue detected. Consider setting size_average=False (loss(size_average = False) and adjusting learning rate so that lrate=lrate/batch_size")
        self.debug('')
        
        
    # output description of inference task
    def print_summary(self, model, validation_dataset = None):
        
        self.debug(' ')
        self.debug('[Model]')
        for lin in model.__repr__().splitlines():
            self.debug(lin)
        self.debug('{0:20}{1}'.format("Trainable params:", ummon.utils.count_parameters(model)))   
        
        self.debug(' ')
        self.debug('[Data]')
        self.debug('{0:18}{1:8}    {2:18} {3} {4:18}'.format('Validation', 
            ummon.utils.get_size_information(validation_dataset), 
            ummon.utils.get_shape_information(validation_dataset), 
            ummon.utils.get_type_information(validation_dataset)),
            ummon.utils.get_numerical_information(validation_dataset))
       
        self.debug(' ')
        self.debug('[Parameters]')
        self.debug('{0:20}{1}'.format("using_cuda"  , next(model.parameters()).is_cuda))
        self.debug('{0:20}{1}'.format("precision" , next(model.parameters()).cpu().data.numpy().dtype))
        self.debug('')
      
    
    
    # print arguments when called as shell program
    def print_args(self, args):
        table = [[ arg, getattr(args, arg)] for arg in vars(args)]
        self.debug(' ')
        self.debug('[Arguments]')
        for key, val in table:
            self.debug('{0:20}{1}'.format(key, val))


if __name__ == "__main__":
    print("This is", Logger().name)
