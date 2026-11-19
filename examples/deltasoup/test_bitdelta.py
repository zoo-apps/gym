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

    def test_large_finite_updates_have_a_finite_mean(self):
        for shape in ((), (3,), (1, 3)):
            with self.subTest(shape=shape):
                base = {"head": np.zeros(shape, np.float32)}
                values = [np.float32(2e38), np.float32(3e38)]
                updates = [pack(base, {"head": np.full(shape, v, np.float32)}, self.schema)
                           for v in values]
                result = merge(base, updates, self.schema, 0)["head"]
                expected = np.float32(np.mean(values, dtype=np.float64))
                np.testing.assert_array_equal(result, np.full(shape, expected, np.float32))
                self.assertEqual(result.dtype, np.float32)

    def test_unrepresentable_reconstruction_is_rejected(self):
        base = {"head": np.array([3e38, 0], np.float32)}
        tuned = {"head": np.array([3.1e38, 2e38], np.float32)}
        adapter = pack(base, tuned, self.schema)
        with self.assertRaisesRegex(ValueError, "float32 range"):
            merge(base, [adapter], self.schema, 0)

    def test_bad_shape_and_dtype_are_rejected(self):
        for value in (np.zeros(3, np.float32), np.zeros((1, 3), np.float64)):
            with self.subTest(shape=value.shape), self.assertRaisesRegex(ValueError, "shape or dtype"):
                pack(self.base, {"head": value}, self.schema)

    def test_duplicate_and_damaged_payloads_are_rejected(self):
        adapter = self.adapter([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge(self.base, [adapter, adapter], self.schema, 0)
        for tensor in (PackedTensor(1.0, b""), PackedTensor(-1.0, b"\x05"),
                       PackedTensor(float("nan"), b"\x05"), PackedTensor(1e100, b"\x05"),
                       PackedTensor(1.0, b"\xff")):
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                merge(self.base, [Adapter(adapter.parent, {"head": tensor})], self.schema, 0)


if __name__ == "__main__":
    unittest.main()
