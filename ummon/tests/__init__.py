import unittest
import ummon
import os

def find_ummon_tests():
    """
    Finds all ummon tests within the directory.
    Tests are identified by `*_test.py` file pattern.
    """
    ummon_directory = os.path.dirname(ummon.__file__)
    suite = unittest.TestLoader().discover(ummon_directory, pattern="*_test.py")
    return suite

def run():
    unittest.TextTestRunner(verbosity=2).run(find_ummon_tests())