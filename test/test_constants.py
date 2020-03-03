#!/usr/local/bin/python3

import unittest
from egenerator.constants import MY_DUMMY_CONSTANT


class TestBasicFunction(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(MY_DUMMY_CONSTANT, 42)


if __name__ == '__main__':
    unittest.main()
