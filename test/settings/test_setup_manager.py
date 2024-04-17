#!/usr/local/bin/python3

import unittest
import os

from egenerator.settings.setup_manager import SetupManager


class TestSetupManagerInitializer(unittest.TestCase):
    """Test initialization of setup manager class.
    Make sure correct exceptions are raised.
    """

    def test_wrong_config_files_data_type(self):

        with self.assertRaises(ValueError) as context:
            setup_manager = SetupManager(None)

        self.assertTrue("Wrong data type:" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            setup_manager = SetupManager("a")

        self.assertTrue("Wrong data type:" in str(context.exception))

    def test_empty_setup_manager(self):

        with self.assertRaises(ValueError) as context:
            setup_manager = SetupManager(())

        self.assertTrue(
            "You must specify at least one config file!"
            in str(context.exception)
        )

    def test_create_setup_manager(self):
        config_files = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_files = [os.path.join(script_dir, "../../configs/test.yaml")]
        setup_manager = SetupManager(config_files)
        self.assertTrue(hasattr(setup_manager, "config"))

    def test_create_duplicate_check(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_files = [os.path.join(script_dir, "../../configs/test.yaml")]
        with self.assertRaises(ValueError) as context:
            setup_manager = SetupManager(config_files * 2)

        self.assertTrue(
            "Keys are defined multiple times" in str(context.exception)
        )


class TestSetupManager(unittest.TestCase):

    def setUp(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_files = [os.path.join(script_dir, "../../configs/test.yaml")]
        self.setup_manager = SetupManager(config_files)

    def test_get_config(self):
        config = self.setup_manager.get_config()
        self.assertTrue(isinstance(config, dict))

    def test_creation_of_setup_manager(self):
        config_true = {
            "unique_name": "event_selection_starting_events_big",
            "float_precision": "float32",
        }
        config = self.setup_manager.get_config()
        for key in config_true:
            self.assertEqual(config[key], config_true[key])

        keys = list(config_true.keys()) + [
            "float_precision",
            "tf_float_precision",
            "np_float_precision",
            "git_short_sha",
            "git_sha",
            "git_origin",
            "git_uncommited_changes",
            "pip_installed_packages",
            "config_name",
            "egenerator_dir",
        ]
        keys = set(keys)

        self.assertListEqual(sorted(keys), sorted(config.keys()))


if __name__ == "__main__":
    unittest.main()
