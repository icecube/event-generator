import unittest
import os

from egenerator import misc
from egenerator.manager.component import Configuration, BaseComponent


class DummyComponent(BaseComponent):

    def _configure(self, data, new_subcomponent=None, **kwargs):
        kwargs["data"] = data
        class_string = misc.get_full_class_string_of_object(self)
        configuration = Configuration(class_string, kwargs)
        if new_subcomponent is None:
            dependent_components = None
        else:
            dependent_components = {"new_subcomponent": new_subcomponent}
        return configuration, data, dependent_components


class WrongComponent(BaseComponent):

    def _configure(self, data, **kwargs):
        kwargs["data"] = data
        kwargs[list(kwargs.keys())[0]] = 3
        configuration = Configuration("WrongComponent", kwargs)
        return configuration, data, None


class MissingKeyComponent(BaseComponent):
    def _configure(self, **kwargs):
        configuration = Configuration("MissingKeyComponent", {})
        return configuration, None, None


class TestBaseComponent(unittest.TestCase):
    """Test base data handler class."""

    # def setUp(self):
    #     # create handler object
    #     data_handler = BaseDataHandler()

    #     # fake setup
    #     data_handler._is_setup = True
    #     data_handler._config = {'setting1': 1337}
    #     data_handler._skip_check_keys = ['ignore_key']
    #     data_handler._tensors = DataTensorList([DataTensor(name='data_tensor',
    #                                                        shape=[None, 86, 1],
    #                                                        tensor_type='data',
    #                                                        dtype='float32')])
    #     self.data_handler = data_handler

    def test_object_initialization(self):

        component = BaseComponent()
        self.assertEqual(component.data, None)
        self.assertEqual(component.is_configured, False)
        self.assertEqual(component.configuration, None)
        self.assertEqual(component.untracked_data, {})
        self.assertEqual(component.sub_components, {})

    def test_get_untracked_member_attributes(self):

        component = BaseComponent()
        untracked_attributes = component._get_untracked_member_attributes()
        self.assertEqual(untracked_attributes, [])
        component._check_member_attributes()

        component.untracked_var = 2
        component._private_var = None
        untracked_attributes = component._get_untracked_member_attributes()
        self.assertListEqual(
            sorted(untracked_attributes),
            sorted(["untracked_var", "_private_var"]),
        )

        with self.assertRaises(ValueError) as context:
            component._check_member_attributes()
        self.assertTrue(
            "Class contains member variables that it should not"
            in str(context.exception)
        )

    def test_configuration_errors_already_configured_error(self):
        component = BaseComponent()
        component._is_configured = True

        with self.assertRaises(ValueError) as context:
            component.configure()
        self.assertTrue(
            "Component is already configured!" in str(context.exception)
        )

    def test_configuration_errors_not_configured_error(self):
        component = BaseComponent()
        component_inherited = DummyComponent()

        with self.assertRaises(ValueError) as context:
            component.configure(component=component_inherited)
        self.assertTrue(
            "Component" in str(context.exception)
            and "is not configured!" in str(context.exception)
        )

    def test_configuration_errors_not_implemented_error(self):
        component = BaseComponent()

        with self.assertRaises(NotImplementedError):
            component.configure()

    def test_configuration_errors_wrong_data_type(self):
        component = DummyComponent()
        with self.assertRaises(TypeError) as context:
            component.configure(data=3, param=42)
        self.assertTrue(
            "Wrong type" in str(context.exception)
            and "Expected dict." in str(context.exception)
        )

    def test_configuration_errors_wrong_data1(self):
        component = WrongComponent()
        with self.assertRaises(ValueError) as context:
            component.configure(data={"12": 2}, param=42)
        self.assertTrue(
            "Values of" in str(context.exception)
            and "do not match:" in str(context.exception)
        )

    def test_configuration_errors_wrong_data2(self):
        component = MissingKeyComponent()
        with self.assertRaises(ValueError) as context:
            component.configure(data={"12": 2}, param=42)
        self.assertTrue(
            "Key" in str(context.exception)
            and "missing in configuration" in str(context.exception)
        )

    def test_configuration(self):
        component = DummyComponent()
        settings = dict(data={"12": 2}, param=42)
        component.configure(**settings)

        self.assertTrue(component.is_configured)
        self.assertDictEqual(component.configuration.settings, settings)

    def test_compatibility(self):
        component = DummyComponent()
        component2 = DummyComponent()
        settings = dict(data={"12": 2}, param=42)
        component.configure(**settings)
        component2.configure(**settings)

        self.assertTrue(component.is_compatible(component2))

    def test_load_and_save(self):
        """Test the saving and loading of a previously created component."""

        # create a sub componeont
        sub_component = DummyComponent()
        sub_component.configure(data={"foo": 2}, great_setting=42)

        component = DummyComponent()
        component2 = DummyComponent()
        settings = dict(
            data={"12": 2}, param=42, new_subcomponent=sub_component
        )
        component.configure(**settings)

        # save trafo model
        dir_path = os.path.join(
            os.path.dirname(__file__), "../../data/temp_test_files/component"
        )
        sub_dir_path = os.path.join(dir_path, "new_subcomponent")

        # remove it if it already exists
        for directory in [sub_dir_path, dir_path]:
            if os.path.exists(directory):
                os.remove(os.path.join(directory, "configuration.yaml"))
                os.remove(os.path.join(directory, "data.pickle"))
                os.removedirs(directory)

        component.save(dir_path, overwrite=False)
        component.save(dir_path, overwrite=True)

        # check error message when attempting to save unconfigured component
        with self.assertRaises(ValueError) as context:
            component2.save(dir_path, overwrite=False)
        self.assertTrue(
            "Component is not configured!" in str(context.exception)
        )

        # check error message when attempting to overwrite file
        with self.assertRaises(IOError) as context:
            component.save(dir_path, overwrite=False)
        self.assertTrue(" already exists!" in str(context.exception))

        # load component from file
        component2.load(dir_path)

        # check of compatibility
        self.assertTrue(component.is_compatible(component2))
        self.assertEqual(component.data, component2.data)

    def test_load_and_save_allow_untracked_attributes(self):

        component = DummyComponent()
        component.configure(data=None, great_setting=42)
        component.additional_key = 3

        dir_path = os.path.join(
            os.path.dirname(__file__), "../../data/temp_test_files/component2"
        )

        # remove it if it already exists
        if os.path.exists(dir_path):
            os.remove(os.path.join(dir_path, "configuration.yaml"))
            os.remove(os.path.join(dir_path, "data.pickle"))
            os.removedirs(dir_path)

        # check error message when attempting to save unconfigured component
        with self.assertRaises(AttributeError) as context:
            component.save(dir_path, allow_untracked_attributes=False)
        self.assertTrue(
            "Found untracked attributes:" in str(context.exception)
        )

        component.save(dir_path, allow_untracked_attributes=True)

    def test_load_alread_configured_error(self):

        component = DummyComponent()
        component.configure(data=None, great_setting=42)

        # check error message when attempting to save unconfigured component
        with self.assertRaises(ValueError) as context:
            component.load("dummy_path")
        self.assertTrue(
            "Component is already configured!" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
