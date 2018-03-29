import unittest
import torch
import numpy
from ummon.trainer import Trainer
from ummon.analyzer import Analyzer

class TestUmmon(unittest.TestCase):

    def test_sine(self):
        # Fit Sine Example..
        #my_trainer = Trainer(model, X, y, loss, optimizer, logger)
        # Get trained model
        #file_best_model = my_trainer.fit()       
        
        #Analyzer().predict(model)  
        
        #assertEquals MSE
        self.assertTrue(False)
        
        
if __name__ == '__main__':
    unittest.main()
