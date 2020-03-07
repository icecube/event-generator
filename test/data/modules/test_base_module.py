#!/usr/local/bin/python3

import unittest
import os

from egenerator.data.modules.base import BaseModule
from egenerator.data.tensor import DataTensorList


class TestDummyWeightModule(unittest.TestCase):

    """Test base module class.
    """

    def test_abstract_class(self):

        # check if error is correctly rasied when trying to instantiate class
        with self.assertRaises(NotImplementedError) as context:
            dummy_module = BaseModule()


if __name__ == '__main__':
    unittest.main()
