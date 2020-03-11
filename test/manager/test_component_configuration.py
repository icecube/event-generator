#!/usr/local/bin/python3

import unittest
import os
import numpy as np
from copy import deepcopy

import egenerator
from egenerator.settings import version_control
from egenerator.manager.component import Configuration, BaseComponent


class TestConfiguration(unittest.TestCase):

    def test_object_initialization(self):

        config_dicts = [
            {
                'class_string': 'dummy_class_string',
                'settings': {'setting1': 1337},
                'mutable_settings': {},
                'check_values': {},
                'sub_component_configurations': {},
                'dependent_sub_components': [],
            },
            {
                'class_string': 'dummy_class_string',
                'settings': {'setting4': 1337},
                'mutable_settings': {},
                'check_values': {'check_values': 42},
                'sub_component_configurations': {},
                'dependent_sub_components': [],
            },
        ]

        for config_dict in config_dicts:
            configuration = Configuration(**config_dict)

            # add event-generator version and git info after the Configuration
            # was configured to check if the correct version is saved.
            short_sha, sha, origin, changes = version_control.get_git_infos()
            config_dict['event_generator_version'] = egenerator.__version__
            config_dict['event_generator_git_short_sha'] = short_sha
            config_dict['event_generator_git_sha'] = sha
            config_dict['event_generator_origin'] = origin
            config_dict['event_generator_uncommitted_changes'] = changes

            self.assertEqual(configuration.settings, config_dict['settings'])
            self.assertEqual(configuration.mutable_settings,
                             config_dict['mutable_settings'])
            self.assertEqual(configuration.check_values,
                             config_dict['check_values'])
            self.assertEqual(configuration.sub_component_configurations,
                             config_dict['sub_component_configurations'])
            self.assertEqual(configuration.dependent_sub_components,
                             config_dict['dependent_sub_components'])
            self.assertEqual(configuration.class_string,
                             config_dict['class_string'])

            self.assertEqual(configuration.dict, config_dict)

            combined_dict = dict(config_dict['settings'])
            combined_dict.update(config_dict['mutable_settings'])
            self.assertEqual(configuration.config, combined_dict)

    def test_object_intialization_duplicates(self):

        with self.assertRaises(ValueError) as context:
            configuration = Configuration('dummy_class_string',
                                          {'setting1': 1337},
                                          {'setting1': 1337})
        self.assertTrue('Keys are defined multiple times'
                        in str(context.exception))

    def test_add_sub_components_type_error(self):
        configuration = Configuration('dummy_class_string',
                                      {'setting1': 1337})

        with self.assertRaises(TypeError) as context:
            configuration.add_sub_components({'setting1': 1337})
        self.assertTrue('Incorrect type' in str(context.exception))

    def test_add_sub_components_key_error(self):
        configuration = Configuration(
            'dummy_class_string',
            {'setting1': 1337},
            sub_component_configurations={'key_already_exists': 'dfa'})

        component = BaseComponent()
        # fake configuration:

        with self.assertRaises(KeyError) as context:
            configuration.add_sub_components(
                {'key_already_exists': component})
        self.assertTrue('Sub component' in str(context.exception) and
                        'already exists!' in str(context.exception))

    def test_add_sub_components_not_configured_error(self):
        configuration = Configuration('dummy_class_string', {'setting1': 1337})

        component = BaseComponent()
        # fake configuration:

        with self.assertRaises(ValueError) as context:
            configuration.add_sub_components(
                {'new_component': component})
        self.assertTrue('Component' in str(context.exception) and
                        'is not configured!' in str(context.exception))

    def test_add_sub_components(self):
        short_sha, sha, origin, changes = version_control.get_git_infos()
        true_dict = {
                'event_generator_version': egenerator.__version__,
                'event_generator_git_short_sha': short_sha,
                'event_generator_git_sha': sha,
                'event_generator_origin': origin,
                'event_generator_uncommitted_changes': changes,
                'class_string': 'dummy_class_string',
                'settings': {'setting1': 1337},
                'mutable_settings': {},
                'check_values': {},
                'dependent_sub_components': [],
                'sub_component_configurations': {
                    'new_component': {
                        'event_generator_version': egenerator.__version__,
                        'event_generator_git_short_sha': short_sha,
                        'event_generator_git_sha': sha,
                        'event_generator_origin': origin,
                        'event_generator_uncommitted_changes': changes,
                        'class_string': 'nested_class_string',
                        'settings': {'nested': 42},
                        'mutable_settings': {},
                        'check_values': {},
                        'sub_component_configurations': {},
                        'dependent_sub_components': [],
                    },
                },
            }
        configuration = Configuration('dummy_class_string', {'setting1': 1337})

        component = BaseComponent()
        # fake configuration:
        component._is_configured = True
        component._configuration = Configuration('nested_class_string',
                                                 {'nested': 42})

        configuration.add_sub_components({'new_component': component})
        self.assertDictEqual(configuration.dict, true_dict)

    def test_check_is_compatible(self):
        dict_1 = {
                'class_string': 'dummy_class_string',
                'settings': {'setting1': 1337},
                'mutable_settings': {'b': 1},
                'check_values': {},
                'dependent_sub_components': [],
                'sub_component_configurations': {
                    'new_component': {
                        'class_string': 'nested_class_string',
                        'settings': {'nested': 42},
                        'mutable_settings': {},
                        'check_values': {},
                        'sub_component_configurations': {},
                        'dependent_sub_components': [],
                    },
                },
            }
        dict_2 = dict(deepcopy(dict_1))
        dict_2['sub_component_configurations']['new_component'][
            'check_values'] = {'a': 32}

        dict_3 = dict(deepcopy(dict_1))
        dict_3['mutable_settings'] = {'a': 32}

        dict_4 = dict(deepcopy(dict_1))
        dict_4['settings'] = {'a': 32}

        dict_5 = dict(deepcopy(dict_1))
        dict_5['sub_component_configurations']['new_component2'] = {
                        'class_string': 'nested_class_string2',
                        'settings': {'nested': 42},
                        'mutable_settings': {},
                        'check_values': {},
                        'sub_component_configurations': {},
                        'dependent_sub_components': [],
                    }
        dict_6 = dict(deepcopy(dict_1))
        dict_6['mutable_settings']['b'] = 32

        configuration1 = Configuration(**dict_1)
        configuration2 = Configuration(**dict(deepcopy(dict_1)))
        configuration3 = Configuration(**dict_2)
        configuration4 = Configuration(**dict_3)
        configuration5 = Configuration(**dict_4)
        configuration6 = Configuration(**dict_5)
        configuration7 = Configuration(**dict_6)

        self.assertTrue(configuration1.is_compatible(configuration2))
        self.assertFalse(configuration1.is_compatible(configuration3))
        self.assertTrue(configuration1.is_compatible(configuration4))
        self.assertFalse(configuration3.is_compatible(configuration4))
        self.assertFalse(configuration1.is_compatible(configuration5))
        self.assertTrue(configuration1.is_compatible(configuration4))
        self.assertFalse(configuration1.is_compatible(configuration6))
        self.assertTrue(configuration1.is_compatible(configuration7))


if __name__ == '__main__':
    unittest.main()
