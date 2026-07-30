import unittest

import cv2
import numpy as np

from detect_folder import _find_component_boxes, _shrink_line_polygons


class ComponentOrientationTests(unittest.TestCase):
    def test_horizontal_rows_are_split_on_y_axis(self) -> None:
        mask = np.zeros((80, 140), dtype=np.uint8)
        cv2.rectangle(mask, (20, 15), (110, 21), 255, -1)
        cv2.rectangle(mask, (20, 43), (110, 49), 255, -1)
        polygon = [[10, 8], [120, 8], [120, 58], [10, 58]]

        components = _find_component_boxes(mask, [polygon])
        items = _shrink_line_polygons(
            mask,
            [polygon],
            padding=0,
            component_boxes=components,
            method='test',
        )

        self.assertEqual(2, len(components['horizontal']))
        self.assertEqual(2, len(items))
        self.assertTrue(all(item['w'] > item['h'] for item in items))
        self.assertLess(items[0]['y'], items[1]['y'])

    def test_vertical_columns_are_split_on_x_axis(self) -> None:
        mask = np.zeros((140, 80), dtype=np.uint8)
        cv2.rectangle(mask, (15, 20), (21, 110), 255, -1)
        cv2.rectangle(mask, (43, 20), (49, 110), 255, -1)
        polygon = [[8, 10], [58, 10], [58, 120], [8, 120]]

        components = _find_component_boxes(mask, [polygon])
        items = _shrink_line_polygons(
            mask,
            [polygon],
            padding=0,
            component_boxes=components,
            method='test',
        )

        self.assertEqual(2, len(components['vertical']))
        self.assertEqual(2, len(items))
        self.assertTrue(all(item['h'] > item['w'] for item in items))
        self.assertLess(items[0]['x'], items[1]['x'])


if __name__ == '__main__':
    unittest.main()
