import unittest

from ctd_overlay_processor.font_size_calibration import (
    _round_positive,
    calibrate_ocr_output,
    character_ink_size,
    ocr_characters,
)
from ctd_overlay_processor.measure_view import char_box_label
from ctd_overlay_processor.mit48px_ocr import (
    DEFAULT_ALPHABET_PATH,
    DEFAULT_IMPLEMENTATION_PATH,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
)
from measure_ocr import _choose_variant, apply_calibrated_font_sizes


class FontSizeCalibrationTests(unittest.TestCase):
    def test_mit48_runtime_paths_are_project_local(self) -> None:
        self.assertEqual(PROJECT_ROOT / 'data' / 'models' / 'mit48pxctc_ocr.ckpt', DEFAULT_MODEL_PATH)
        for path in (DEFAULT_ALPHABET_PATH, DEFAULT_IMPLEMENTATION_PATH):
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.is_relative_to(PROJECT_ROOT), path)

    def test_font_size_uses_normal_half_up_rounding(self) -> None:
        self.assertEqual(30, _round_positive(30.4))
        self.assertEqual(31, _round_positive(30.5))

    def test_ocr_characters_drops_whitespace(self) -> None:
        self.assertEqual(['日', '本', '語'], ocr_characters('日 本\n語'))

    def test_calibration_requires_matching_character_count(self) -> None:
        output = {
            'pages': {
                '001.jpg': [{
                    'font_size': 30,
                    'ocr_characters': [{
                        'line_index': 0,
                        'character_index': 0,
                        'orientation': 'vertical',
                        'ocr_text': '',
                        'ocr_probability': 0,
                        'status': 'not_single_character',
                        'width': 20,
                        'height': 22,
                    }],
                }],
            },
        }
        ready = calibrate_ocr_output(output)
        self.assertEqual(0, ready)
        self.assertEqual('no_reliable_characters', output['pages']['001.jpg'][0]['font_fit']['status'])

    def test_calibration_returns_suggestion_for_three_supported_characters(self) -> None:
        expected_size = 30
        boxes = []
        for character in '日本語':
            ink_size = character_ink_size(character, expected_size)
            self.assertIsNotNone(ink_size)
            width, height = ink_size or (0, 0)
            boxes.append({'width': width, 'height': height})
        output = {
            'pages': {
                '001.jpg': [{
                    'font_size': 30,
                    'ocr_characters': [
                        {
                            'line_index': 0,
                            'character_index': index,
                            'orientation': 'vertical',
                            'ocr_text': char,
                            'ocr_probability': 0.99,
                            'status': 'accepted',
                            **box,
                        }
                        for index, (char, box) in enumerate(zip('日本語', boxes))
                    ],
                }],
            },
        }
        ready = calibrate_ocr_output(output)
        fit = output['pages']['001.jpg'][0]['font_fit']
        self.assertEqual(1, ready)
        self.assertEqual('ready', fit['status'])
        self.assertEqual(expected_size, fit['suggested_font_size'])
        self.assertIsInstance(fit['suggested_font_size'], int)

    def test_calibration_recovers_rendered_integer_font_size(self) -> None:
        expected_size = 40
        characters = '日本語'
        boxes = []
        for character in characters:
            ink_size = character_ink_size(character, expected_size)
            self.assertIsNotNone(ink_size)
            width, height = ink_size or (0, 0)
            boxes.append({'width': width, 'height': height})
        output = {
            'pages': {
                '001.jpg': [{
                    'font_size': 34,
                    'orientation': 'vertical',
                    'ocr_characters': [
                        {
                            'line_index': 0,
                            'character_index': index,
                            'ocr_text': character,
                            'ocr_probability': 0.99,
                            'status': 'accepted',
                            **box,
                        }
                        for index, (character, box) in enumerate(zip(characters, boxes))
                    ],
                }],
            },
        }
        ready = calibrate_ocr_output(output)
        fit = output['pages']['001.jpg'][0]['font_fit']
        self.assertEqual(1, ready)
        self.assertEqual('ready', fit['status'])
        self.assertEqual(expected_size, fit['suggested_font_size'])
        self.assertIsInstance(fit['suggested_font_size'], int)

    def test_character_variant_uses_best_single_character_only(self) -> None:
        selected = _choose_variant(
            [
                {'text': '日本', 'probability': 0.99, 'pad': 8},
                {'text': '日', 'probability': 0.91, 'pad': 4},
            ],
            0.3,
        )
        self.assertEqual('accepted', selected['status'])
        self.assertEqual('日', selected['ocr_text'])
        self.assertEqual(4, selected['selected_pad'])

    def test_character_variant_rejects_low_confidence_without_affecting_neighbors(self) -> None:
        selected = _choose_variant(
            [{'text': '語', 'probability': 0.2, 'pad': 4}],
            0.3,
        )
        self.assertEqual('low_confidence', selected['status'])
        self.assertEqual('語', selected['ocr_text'])

    def test_apply_calibrated_font_sizes_updates_measure_and_preserves_detected_size(self) -> None:
        measure = {'pages': {'001.jpg': [{'font_size': 27}]}}
        output = {
            'pages': {
                '001.jpg': [{
                    'measure_item_index': 0,
                    'font_fit': {'status': 'ready', 'suggested_font_size': 31},
                }],
            },
        }
        changed = apply_calibrated_font_sizes(measure, output, even_font_size=True)
        item = measure['pages']['001.jpg'][0]
        self.assertEqual(1, changed)
        self.assertEqual(27, item['font_size_detected'])
        self.assertEqual(32, item['font_size'])
        self.assertEqual('mit48_cached_font_ink_ratio', item['font_size_method'])
        self.assertEqual(32, output['pages']['001.jpg'][0]['font_fit']['applied_font_size'])

    def test_char_box_label_keeps_geometry_only(self) -> None:
        self.assertEqual('W22H32', char_box_label({'width': 22, 'height': 32}))
        self.assertEqual(
            'W22H32FS36',
            char_box_label({'width': 22, 'height': 32, 'calculated_font_size': 36}),
        )


if __name__ == '__main__':
    unittest.main()
