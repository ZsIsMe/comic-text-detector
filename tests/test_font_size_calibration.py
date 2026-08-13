import unittest

from PySide6.QtGui import QFont, QFontMetricsF, QGuiApplication

from ctd_overlay_processor.font_size_calibration import (
    calibrate_ocr_output,
    load_application_font,
    ocr_characters,
)
from ctd_overlay_processor.measure_view import char_box_label
from measure_ocr import _choose_variant


class FontSizeCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QGuiApplication.instance() or QGuiApplication([])
        _, cls.family = load_application_font()

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
        ready = calibrate_ocr_output(output, self.family)
        self.assertEqual(0, ready)
        self.assertEqual('no_reliable_characters', output['pages']['001.jpg'][0]['font_fit']['status'])

    def test_calibration_returns_suggestion_for_three_supported_characters(self) -> None:
        font = QFont(self.family)
        font.setPixelSize(30)
        metrics = QFontMetricsF(font)
        boxes = []
        for char in '日本語':
            rect = metrics.tightBoundingRect(char)
            boxes.append({'width': rect.width(), 'height': rect.height()})
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
        ready = calibrate_ocr_output(output, self.family)
        fit = output['pages']['001.jpg'][0]['font_fit']
        self.assertEqual(1, ready)
        self.assertEqual('ready', fit['status'])
        self.assertGreater(fit['suggested_font_size'], 0)

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

    def test_char_box_label_appends_calculated_font_size(self) -> None:
        self.assertEqual(
            'W22H32 F28',
            char_box_label({'width': 22, 'height': 32, 'calculated_font_size': 28}),
        )
        self.assertEqual('W22H32', char_box_label({'width': 22, 'height': 32}))


if __name__ == '__main__':
    unittest.main()
