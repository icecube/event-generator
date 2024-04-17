#!/usr/local/bin/python3

import unittest

from egenerator.data.tensor import DataTensor, DataTensorList


def create_tensors(names):
    """Generate a list of DataTensor objects.

    Parameters
    ----------
    names : list of str
        A list of strings. Must have length 7.

    Returns
    -------
    list of DataTensor
        A list of DataTensor objects.
    """
    tensors = [
        DataTensor(
            name=names[0],
            shape=[None],
            tensor_type="data",
            dtype="float32",
            vector_info={
                "type": "index",
                "reference": "value_tensor",
            },
        ),
        DataTensor(
            name=names[1],
            shape=[None],
            tensor_type="data",
            dtype="float32",
            vector_info={
                "type": "index",
                "reference": "value_tensor",
            },
        ),
        DataTensor(
            name=names[2],
            shape=[None, 1],
            tensor_type="data",
            dtype="float32",
            trafo_log=True,
            vector_info={
                "type": "index",
                "reference": "value_tensor",
            },
        ),
        DataTensor(
            name=names[3],
            shape=[None],
            tensor_type="data",
            dtype="float32",
            vector_info={
                "type": "index",
                "reference": "value_tensor2",
            },
        ),
        DataTensor(
            name=names[4],
            shape=[None],
            tensor_type="data",
            dtype="float32",
            vector_info={
                "type": "index",
                "reference": "value_tensor2",
                "additional_key": None,
            },
        ),
        DataTensor(
            name=names[5],
            shape=[None],
            tensor_type="data",
            dtype="float32",
            vector_info={
                "type": "index",
                "reference": "value_tensor2",
                "additional_key": None,
            },
        ),
        DataTensor(
            name=names[6],
            shape=[None],
            tensor_type="data",
            dtype="float32",
            vector_info={
                "type": "index",
                "reference": "value_tensor2",
                "additional_key": None,
            },
            new_member_var=42,
        ),
    ]
    return tensors


class TestDataTensor(unittest.TestCase):
    """Test data tensor class.
    Make sure correct exceptions are raised.
    """

    def test_unset_shape(self):
        with self.assertRaises(ValueError) as context:
            DataTensor(
                name="test_tensor",
                shape=None,
                tensor_type="data",
                dtype="float32",
            )

        self.assertTrue("Shape must be defined but" in str(context.exception))

    def test_unset_trafo_axis_shape(self):
        with self.assertRaises(ValueError) as context:
            DataTensor(
                name="test_tensor",
                shape=[1, None],
                tensor_type="data",
                trafo_log=True,
                dtype="float32",
            )

        self.assertTrue(
            "When using logarithm trafo, the shape along the "
            in str(context.exception)
        )

    def test_wrong_trafo_axis_type(self):
        with self.assertRaises(ValueError) as context:
            DataTensor(
                name="test_tensor",
                shape=[13],
                trafo_log_axis=1.0,
                tensor_type="wrong_type",
                dtype="float32",
            )

        self.assertTrue(
            "Trafo log axis must be an integer" in str(context.exception)
        )

    def test_wrong_type(self):
        with self.assertRaises(ValueError) as context:
            DataTensor(
                name="test_tensor",
                shape=[None],
                tensor_type="wrong_type",
                dtype=None,
            )

        self.assertTrue("Unknown type:" in str(context.exception))

    def test_wrong_dtype(self):
        with self.assertRaises(ValueError) as context:
            DataTensor(
                name="test_tensor", shape=[None], tensor_type="data", dtype=1
            )
        self.assertTrue(" != " in str(context.exception))

    def test_dtype_not_part_of_numpy(self):
        with self.assertRaises(ValueError) as context:
            DataTensor(
                name="test_tensor",
                shape=[None],
                tensor_type="data",
                dtype="fa",
            )
        self.assertTrue("Invalid dtype str:" in str(context.exception))

    def test_wrong_vector_info(self):
        with self.assertRaises(ValueError) as context:
            DataTensor(
                name="test_tensor",
                shape=[None],
                tensor_type="data",
                dtype="float32",
                vector_info={
                    "type": "wrong_type",
                },
            )

        self.assertTrue("Unknown vector type:" in str(context.exception))

    def test_equality_and_inequality(self):

        names = ["tensor_name"] * 5 + ["arbitray_name", "s"]
        tensors = create_tensors(names)

        self.assertFalse(tensors[0] is None)
        self.assertFalse(tensors[0] == 3)
        self.assertTrue(tensors[0] == tensors[1])
        self.assertFalse(tensors[0] == tensors[2])
        self.assertFalse(tensors[0] == tensors[3])
        self.assertFalse(tensors[1] == tensors[2])
        self.assertFalse(tensors[2] == tensors[3])
        self.assertFalse(tensors[3] == tensors[4])

        self.assertTrue(tensors[0] is not None)
        self.assertTrue(tensors[0] != 3)
        self.assertFalse(tensors[0] != tensors[1])
        self.assertTrue(tensors[0] != tensors[2])
        self.assertTrue(tensors[0] != tensors[3])
        self.assertTrue(tensors[1] != tensors[2])
        self.assertTrue(tensors[2] != tensors[3])
        self.assertTrue(tensors[3] != tensors[4])


class TestDataTensorList(unittest.TestCase):
    """Test data tensor list class.
    Make sure correct exceptions are raised.
    """

    def test_empty_list(self):
        tensor_list = DataTensorList([])
        self.assertEqual(tensor_list.list, [])
        self.assertEqual(tensor_list.names, [])
        self.assertEqual(tensor_list.shapes, [])
        self.assertEqual(tensor_list.exists, [])
        self.assertEqual(tensor_list._name_dict, {})
        self.assertEqual(tensor_list._index_dict, {})

        with self.assertRaises(ValueError) as context:
            names = ["t3", "t2", "arbitray_name", "t6", "t6", "weights", "s"]
            tensors = create_tensors(names)
            tensor_list = DataTensorList(tensors)

        self.assertTrue("Found duplicate names:" in str(context.exception))

    def test_duplicate_name(self):
        with self.assertRaises(ValueError) as context:
            names = ["t3", "t2", "arbitray_name", "t6", "t6", "weights", "s"]
            tensors = create_tensors(names)
            tensor_list = DataTensorList(tensors)

        self.assertTrue("Found duplicate names:" in str(context.exception))

    def test_initializer_and_sorting_of_tensors(self):
        names = ["t3", "t2", "arbitray_name", "t6", "labels", "weights", "s"]
        tensors = create_tensors(names)
        tensor_list = DataTensorList(tensors)
        self.assertListEqual(sorted(names), tensor_list.names)

    def test_get_methods(self):
        names = ["t3", "t2", "arbitray_name", "t6", "labels", "weights", "s"]
        tensors = create_tensors(names)
        tensor_list = DataTensorList(tensors)
        self.assertEqual(tensor_list.get_index("arbitray_name"), 0)
        self.assertEqual(tensor_list.get_index("weights"), 6)
        self.assertEqual(tensor_list.get_name(0), "arbitray_name")
        self.assertTrue(tensor_list.list[0] == tensor_list["arbitray_name"])
        self.assertTrue(tensor_list.list[1] == tensor_list["labels"])

    def test_equality_and_inequality(self):

        names = ["t3", "t2", "arbitray_name", "t6", "labels", "weights", "s"]
        tensors = create_tensors(names)
        tensor_list1 = DataTensorList(tensors[:3])
        tensor_list2 = DataTensorList(tensors[:3])
        tensor_list3 = DataTensorList(tensors[:4])
        tensor_list4 = DataTensorList(tensors[1:4])

        self.assertFalse(tensor_list1 == 4)
        self.assertTrue(tensor_list1 != 4)
        self.assertTrue(tensor_list1 == tensor_list2)
        self.assertFalse(tensor_list1 != tensor_list2)
        self.assertFalse(tensor_list1 == tensor_list3)
        self.assertTrue(tensor_list1 != tensor_list3)
        self.assertFalse(tensor_list3 == tensor_list1)
        self.assertTrue(tensor_list3 != tensor_list1)
        self.assertFalse(tensor_list2 == tensor_list4)
        self.assertTrue(tensor_list2 != tensor_list4)

    def test_copy_constructor(self):

        names = ["t3", "t2", "arbitray_name", "t6", "labels", "weights", "s"]
        tensors = create_tensors(names)
        tensor_list1 = DataTensorList(tensors[:3])
        tensor_list2 = DataTensorList(tensor_list1)
        tensor_list3 = DataTensorList(tensors[:4])

        self.assertTrue(tensor_list1 == tensor_list2)
        self.assertTrue(tensor_list3 != tensor_list2)

    def test_serialization_methods(self):
        names = ["t3", "t2", "arbitray_name", "t6", "labels", "weights", "s"]
        tensors = create_tensors(names)
        tensor_list1 = DataTensorList(tensors[1:3])
        tensor_list2 = DataTensorList(tensors[5:7])
        tensor_list3 = DataTensorList(tensor_list1.serialize())
        tensor_list4 = DataTensorList(tensor_list2.serialize())

        self.assertTrue(tensor_list1 != tensor_list2)
        self.assertTrue(tensor_list1 == tensor_list3)
        self.assertTrue(tensor_list2 == tensor_list4)


if __name__ == "__main__":
    unittest.main()
