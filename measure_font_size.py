#!/usr/bin/env python3
"""Font-size selection for measure.json.

Usage examples:
  python measure_font_size.py 6 30 18
  python measure_font_size.py --json 18 21 30
"""

from __future__ import annotations

import argparse
import json


RUBY_FILTER_RATIO = 0.55


def lower_median(values: list[float]) -> float:
    sorted_values = sorted(values)
    return sorted_values[(len(sorted_values) - 1) // 2]


def choose_font_size(
    widths: list[float],
    ruby_filter_ratio: float = RUBY_FILTER_RATIO,
) -> tuple[float, dict]:
    valid_widths = [float(width) for width in widths if float(width) > 0]
    if not valid_widths:
        raise ValueError('widths must contain at least one positive number')

    max_width = max(valid_widths)
    threshold = max_width * ruby_filter_ratio
    selected_widths = [width for width in valid_widths if width >= threshold]
    if not selected_widths:
        selected_widths = valid_widths

    font_size = lower_median(selected_widths)
    debug = {
        'method': 'ruby_filtered_lower_median',
        'ruby_filter_ratio': ruby_filter_ratio,
        'widths': valid_widths,
        'max_width': max_width,
        'threshold': threshold,
        'selected_widths': selected_widths,
        'font_size': font_size,
    }
    return font_size, debug


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Choose block font_size from line widths with ruby filtering.',
    )
    parser.add_argument('widths', nargs='+', type=float, help='Line widths, e.g. 6 30 18')
    parser.add_argument(
        '--ratio',
        type=float,
        default=RUBY_FILTER_RATIO,
        help=f'Ruby filtering ratio, default {RUBY_FILTER_RATIO}',
    )
    parser.add_argument('--json', action='store_true', help='Print full debug JSON.')
    args = parser.parse_args()

    font_size, debug = choose_font_size(args.widths, args.ratio)
    if args.json:
        print(json.dumps(debug, ensure_ascii=False, indent=2))
    else:
        selected = ', '.join(f'{width:g}' for width in debug['selected_widths'])
        print(f'font_size={font_size:g} selected=[{selected}] threshold={debug["threshold"]:g}')


if __name__ == '__main__':
    main()
