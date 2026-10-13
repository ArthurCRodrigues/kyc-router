"""Small, deterministic geometry checks; no real identity data is needed."""

import copy
import unittest

import numpy as np

from kyc_router.geometry import crop_field, rectangles_from_template, scale_field_shape
from kyc_router.schema import REQUIRED_FIELDS


RAW_NAMES = ("number", "surname", "name", "birth_date", "expiry_date", "nationality", "gender")


def template_record():
    return {"regions": [
        {
            "shape_attributes": {"name": "rect", "x": 703, "y": 242, "width": 251, "height": 55},
            "region_attributes": {"field_name": raw},
        }
        for raw in RAW_NAMES
    ]}


class GeometryTests(unittest.TestCase):
    def test_notebook_rounding_order(self):
        shape = {"x": 703, "y": 242, "width": 251, "height": 55}
        scaled = scale_field_shape(shape, 2168, 1360, 1084, 680)
        self.assertEqual(scaled, {"x": 352, "y": 121, "width": 126, "height": 28})
        self.assertEqual(shape["x"], 703)

    def test_padding_clipping_and_copy(self):
        image = np.arange(30 * 50 * 3).astype(np.uint8).reshape(30, 50, 3)
        shape = {"x": 20, "y": 10, "width": 10, "height": 10}
        crop, bounds = crop_field(image, shape)
        self.assertEqual(bounds, {"x1": 8, "y1": 9, "x2": 42, "y2": 24})
        np.testing.assert_array_equal(crop, image[9:24, 8:42])
        crop[:] = 0
        self.assertTrue(image[9:24, 8:42].any(), "Crops must own their pixels.")
        crop, bounds = crop_field(image, {"x": 0, "y": 0, "width": 5, "height": 5})
        self.assertEqual(bounds, {"x1": 0, "y1": 0, "x2": 17, "y2": 7})
        self.assertEqual(crop.shape, (7, 17, 3))

    def test_empty_and_invalid_shapes(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        for shape in (
            {"x": 1, "y": 1, "width": 0, "height": 2},
            {"x": 1, "y": 1, "width": 2, "height": -1},
            {"x": 100, "y": 100, "width": 2, "height": 2},
            {"x": True, "y": 1, "width": 2, "height": 2},
            {"x": 1.5, "y": 1, "width": 2, "height": 2},
            {"x": 1, "y": 1, "width": 2},
        ):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                crop_field(image, shape)

    def test_template_geometry_does_not_require_transcriptions(self):
        record = template_record()
        rectangles = rectangles_from_template(record, 2168, 1360)
        self.assertEqual(tuple(rectangles), REQUIRED_FIELDS)
        self.assertEqual(rectangles["surname"], {"x": 352, "y": 121, "width": 126, "height": 28})
        with_labels = copy.deepcopy(record)
        for region in with_labels["regions"]:
            region["region_attributes"]["value"] = "A label that must never enter inference"
        self.assertEqual(rectangles_from_template(with_labels, 2168, 1360), rectangles)
        self.assertNotIn("value", str(rectangles))

    def test_template_requires_unique_rectangles(self):
        for defect in ("missing", "duplicate", "polygon"):
            record = template_record()
            if defect == "missing":
                record["regions"].pop()
            elif defect == "duplicate":
                record["regions"].append(copy.deepcopy(record["regions"][0]))
            else:
                record["regions"][0]["shape_attributes"]["name"] = "polygon"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                rectangles_from_template(record, 2168, 1360)


if __name__ == "__main__":
    unittest.main()
