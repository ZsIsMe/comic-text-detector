import unittest

import numpy as np
import torch

from ctd_overlay_processor.font_size_calibration import (
    _font_size_candidates,
    _round_positive,
    calibrate_ocr_output,
    character_ink_size,
    fit_character_pixel_size,
    ocr_characters,
)
from ctd_overlay_processor.measure_view import char_box_label
from ctd_overlay_processor.mit48px_ocr import (
    DEFAULT_ALPHABET_PATH,
    DEFAULT_IMPLEMENTATION_PATH,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    collapse_ctc_runs,
)
from ctd_overlay_processor.processor import load_char_boxes
from measure_ocr import (
    _character_region_from_token,
    _line_tasks_for_page,
    _projection_ink_runs,
    _select_ink_run,
    apply_calibrated_font_sizes,
)


class FontSizeCalibrationTests(unittest.TestCase):
    def test_mit48_runtime_paths_are_project_local(self) -> None:
        self.assertEqual(PROJECT_ROOT / 'data' / 'models' / 'mit48pxctc_ocr.ckpt', DEFAULT_MODEL_PATH)
        for path in (DEFAULT_ALPHABET_PATH, DEFAULT_IMPLEMENTATION_PATH):
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.is_relative_to(PROJECT_ROOT), path)

    def test_font_size_uses_normal_half_up_rounding(self) -> None:
        self.assertEqual(30, _round_positive(30.4))
        self.assertEqual(31, _round_positive(30.5))

    def test_candidate_grid_uses_default_24_and_step_2(self) -> None:
        self.assertEqual([28.0, 30.0, 32.0], _font_size_candidates(29.4, 31.2, 24.0, 2.0))

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
        self.assertAlmostEqual(expected_size, fit['suggested_font_size_float'], places=3)
        self.assertIsInstance(fit['suggested_font_size'], float)
        self.assertEqual(24.0, fit['default_font_size'])
        self.assertEqual(2.0, fit['font_size_step'])

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
        self.assertIsInstance(fit['suggested_font_size'], float)

    def test_ctc_collapse_keeps_independent_best_timestep(self) -> None:
        dictionary = ['<BLANK>', '日', '本', '語']
        log_probs = torch.full((8, len(dictionary)), -10.0)
        for timestep, character_id in enumerate([0, 1, 1, 0, 2, 2, 0, 3]):
            log_probs[timestep, character_id] = -0.01
        log_probs[2, 1] = -0.001
        tokens = collapse_ctc_runs(log_probs, dictionary, valid_timesteps=8)
        self.assertEqual(['日', '本', '語'], [token['text'] for token in tokens])
        self.assertEqual([2, 4, 7], [token['timestep'] for token in tokens])

    def test_missing_ctc_token_does_not_shift_later_position(self) -> None:
        dictionary = ['<BLANK>', '日', '本', '語']
        full = torch.full((9, len(dictionary)), -10.0)
        missing = torch.full((9, len(dictionary)), -10.0)
        for tensor, ids in (
            (full, [0, 1, 1, 0, 2, 2, 0, 3, 3]),
            (missing, [0, 1, 1, 0, 0, 0, 0, 3, 3]),
        ):
            for timestep, character_id in enumerate(ids):
                tensor[timestep, character_id] = -0.01
        full_tokens = collapse_ctc_runs(full, dictionary)
        missing_tokens = collapse_ctc_runs(missing, dictionary)
        self.assertEqual(7, full_tokens[-1]['timestep'])
        self.assertEqual(7, missing_tokens[-1]['timestep'])
        self.assertEqual(full_tokens[-1]['position_ratio'], missing_tokens[-1]['position_ratio'])

    def test_wrong_token_without_local_ink_does_not_change_neighbor_regions(self) -> None:
        mask = np.zeros((12, 60), dtype=np.uint8)
        mask[2:10, 10:16] = 255
        mask[2:10, 44:50] = 255
        prepared = {
            'mask': mask,
            'origin': (100, 200),
            'source_size': (60, 12),
            'canonical_size': (60, 12),
            'orientation': 'horizontal',
        }
        task = {
            'line_index': 0,
            'line': {'x': 100, 'y': 200, 'w': 60, 'h': 12, 'font_width_px': 8},
        }
        left = _character_region_from_token(
            {'text': '日', 'probability': 0.99, 'position_ratio': 0.22, 'timestep': 2},
            prepared, task, 0, 0.3,
        )
        wrong = _character_region_from_token(
            {'text': '錯', 'probability': 0.99, 'position_ratio': 0.5, 'timestep': 5},
            prepared, task, 1, 0.3,
        )
        right = _character_region_from_token(
            {'text': '語', 'probability': 0.99, 'position_ratio': 0.78, 'timestep': 8},
            prepared, task, 2, 0.3,
        )
        self.assertEqual([110, 202, 116, 210], left['bbox'])
        self.assertIsNone(wrong)
        self.assertEqual([144, 202, 150, 210], right['bbox'])

    def test_ctc_anchor_selects_complete_ink_run_instead_of_clipping_window(self) -> None:
        projection = np.zeros(50, dtype=np.int32)
        projection[8:40] = 1
        runs = _projection_ink_runs(projection, 31)
        self.assertEqual((8, 40), _select_ink_run(runs, 13.8, 31))

    def test_ctc_anchor_in_gap_prefers_forward_run_and_attached_stroke(self) -> None:
        runs = [(78, 104), (107, 110), (112, 132)]
        self.assertEqual((107, 132), _select_ink_run(runs, 111.0, 25))

    def test_new_line_tasks_ignore_legacy_hard_character_boxes(self) -> None:
        items = [{'source_block_index': 2, 'font_size': 30, 'orientation': 'vertical'}]
        line = {'x': 10, 'y': 20, 'w': 24, 'h': 100}
        debug = {
            'font_size': {
                '001.jpg': [{
                    'source_block_index': 2,
                    'lines': [line],
                    'char_boxes': [{'bbox': [1, 2, 3, 4]}],
                }],
            },
        }
        output_items, tasks = _line_tasks_for_page(items, debug, '001.jpg', None)
        self.assertEqual(1, len(output_items))
        self.assertEqual(1, len(tasks))
        self.assertEqual(line, tasks[0]['line'])
        self.assertNotIn('char_boxes', output_items[0])

    def test_font_size_uses_each_characters_width_and_height(self) -> None:
        width, height = character_ink_size('日', 40) or (0, 0)
        horizontal = fit_character_pixel_size('日', {'width': width, 'height': height}, 'horizontal')
        vertical = fit_character_pixel_size('日', {'width': width, 'height': height}, 'vertical')
        self.assertAlmostEqual(40, horizontal['estimated_pixel_size'], places=2)
        self.assertAlmostEqual(40, vertical['estimated_pixel_size'], places=2)
        self.assertEqual('width_height', horizontal['size_dimension'])
        self.assertEqual('width_height', vertical['size_dimension'])

    def test_character_is_rejected_when_width_and_height_sizes_disagree(self) -> None:
        width, height = character_ink_size('日', 40) or (0, 0)
        output = {
            'pages': {
                '001.jpg': [{
                    'font_size': 40,
                    'ocr_characters': [{
                        'line_index': 0,
                        'character_index': 0,
                        'ocr_text': '日',
                        'ocr_probability': 0.99,
                        'status': 'accepted',
                        'width': width * 2,
                        'height': height,
                    }],
                }],
            },
        }
        calibrate_ocr_output(output)
        result = output['pages']['001.jpg'][0]['font_fit']['character_results'][0]
        self.assertFalse(result['accepted'])
        self.assertEqual('width_height_size_disagree', result['reason'])

    def test_processor_loads_only_final_reliable_character_regions(self) -> None:
        measure_ocr = {
            'pages': {
                '001.jpg': [{
                    'source_block_index': 4,
                    'ocr_characters': [
                        {'line_index': 0, 'character_index': 0, 'status': 'accepted', 'bbox': [1, 2, 3, 4]},
                        {'line_index': 0, 'character_index': 1, 'status': 'accepted', 'bbox': [5, 6, 7, 8]},
                        {'line_index': 0, 'character_index': 2, 'status': 'low_confidence', 'bbox': [9, 10, 11, 12]},
                    ],
                    'font_fit': {
                        'character_results': [
                            {'line_index': 0, 'character_index': 0, 'accepted': True, 'pixel_size': 30, 'estimated_pixel_size': 30.2, 'error': 0.1},
                            {'line_index': 0, 'character_index': 1, 'accepted': False, 'pixel_size': 31},
                            {'line_index': 0, 'character_index': 2, 'accepted': True, 'pixel_size': 32},
                        ],
                    },
                }],
            },
        }
        boxes = load_char_boxes({'font_size': {'001.jpg': [{'char_boxes': [{'bbox': [0, 0, 1, 1]}]}]}}, '001.jpg', measure_ocr)
        self.assertEqual(1, len(boxes))
        self.assertEqual([1, 2, 3, 4], boxes[0]['bbox'])
        self.assertEqual(30, boxes[0]['calculated_font_size'])

    def test_apply_calibrated_font_sizes_updates_measure_and_preserves_detected_size(self) -> None:
        measure = {'pages': {'001.jpg': [{'font_size': 27}]}}
        output = {
            'pages': {
                '001.jpg': [{
                    'measure_item_index': 0,
                    'font_fit': {'status': 'ready', 'suggested_font_size': 31.0},
                }],
            },
        }
        changed = apply_calibrated_font_sizes(measure, output)
        item = measure['pages']['001.jpg'][0]
        self.assertEqual(1, changed)
        self.assertEqual(27, item['font_size_detected'])
        self.assertEqual(31.0, item['font_size'])
        self.assertEqual('mit48_cached_font_ink_candidate_grid', item['font_size_method'])
        self.assertEqual(31.0, output['pages']['001.jpg'][0]['font_fit']['applied_font_size'])

    def test_apply_calibrated_font_sizes_preserves_one_decimal(self) -> None:
        measure = {'pages': {'001.jpg': [{'font_size': 27}]}}
        output = {
            'pages': {
                '001.jpg': [{
                    'measure_item_index': 0,
                    'font_fit': {
                        'status': 'ready',
                        'suggested_font_size': 30.6,
                        'suggested_font_size_float': 30.6,
                    },
                }],
            },
        }
        apply_calibrated_font_sizes(measure, output)
        self.assertEqual(30.6, measure['pages']['001.jpg'][0]['font_size'])
        self.assertEqual(30.6, output['pages']['001.jpg'][0]['font_fit']['applied_font_size'])
        self.assertEqual(30.6, output['pages']['001.jpg'][0]['font_fit']['applied_font_size_source_float'])

    def test_char_box_label_keeps_geometry_only(self) -> None:
        self.assertEqual('W22H32', char_box_label({'width': 22, 'height': 32}))
        self.assertEqual(
            'W22H32FS36.0',
            char_box_label({'width': 22, 'height': 32, 'calculated_font_size': 36}),
        )
        self.assertEqual(
            'W22H32FS35.7',
            char_box_label({
                'width': 22,
                'height': 32,
                'calculated_font_size': 36,
                'estimated_font_size': 35.74,
            }),
        )


if __name__ == '__main__':
    unittest.main()
