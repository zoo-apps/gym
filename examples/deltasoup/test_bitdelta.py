import unittest

import numpy as np

from bitdelta import Adapter, PackedTensor, merge, pack


class BitDeltaTests(unittest.TestCase):
    def setUp(self):
        self.base = {"head": np.zeros((1, 3), dtype=np.float32)}
        self.schema = "three classes: A,B,C"

    def adapter(self, values):
        tuned = {"head": np.array([values], dtype=np.float32)}
        return pack(self.base, tuned, self.schema)

    def test_non_byte_aligned_tensor(self):
        adapter = self.adapter([1, -2, 3])
        self.assertEqual(adapter.tensors["head"].signs, bytes([5]))
        result = merge(self.base, [adapter], self.schema, 0)
        np.testing.assert_array_equal(result["head"], [[2, -2, 2]])

    def test_constant_zero_update(self):
        result = merge(self.base, [self.adapter([0, 0, 0])], self.schema, 0)
        np.testing.assert_array_equal(result["head"], self.base["head"])

    def test_trim_removes_both_tails_at_each_coordinate(self):
        values = ([100, -100, 100], [2, 2, -2], [-9, 9, -9])
        result = merge(self.base, [self.adapter(v) for v in values], self.schema, 1 / 3)
        np.testing.assert_array_equal(result["head"], [[2, 2, -2]])

    def test_wrong_parent_and_label_order_are_rejected(self):
        adapter = self.adapter([1, 2, 3])
        for base, schema in ((self.base, "three classes: C,B,A"),
                             ({"head": self.base["head"] + 1}, self.schema)):
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, "Incompatible"):
                merge(base, [adapter], schema, 0)

    def test_bad_shape_and_dtype_are_rejected(self):
        for value in (np.zeros(3, np.float32), np.zeros((1, 3), np.float64)):
            with self.subTest(shape=value.shape), self.assertRaisesRegex(ValueError, "shape or dtype"):
                pack(self.base, {"head": value}, self.schema)

    def test_duplicate_and_damaged_payloads_are_rejected(self):
        adapter = self.adapter([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge(self.base, [adapter, adapter], self.schema, 0)
        for tensor in (PackedTensor(1.0, b""), PackedTensor(-1.0, b"\x05"),
                       PackedTensor(float("nan"), b"\x05"), PackedTensor(1.0, b"\xff")):
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                merge(self.base, [Adapter(adapter.parent, {"head": tensor})], self.schema, 0)


if __name__ == "__main__":
    unittest.main()
