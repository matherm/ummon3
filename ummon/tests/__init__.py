import unittest
import ummon

def my_module_suites():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(ummon)
    return suite