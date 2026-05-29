#!/usr/bin/env python3
"""Run PaddleOCR-VL manga OCR for boxes in measure.json."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'PaddleOCR-VL-For-Manga'
DEFAULT_PROMPT = 'OCR:'


def _load_json(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def _write_json(path: str | Path, data: dict) -> None:
    with open(path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_input_paths(
    path1: str,
    path2: str | None,
    measure_json: str | None,
) -> tuple[str, str]:
    if measure_json is not None:
        return measure_json, path1

    if path2 is None:
        image_dir = path1
        return osp.join(image_dir, 'ctd', 'measure.json'), image_dir

    # Backward-compatible form: measure_ocr.py ctd/measure.json image_dir
    return path1, path2


def _image_path_for_page(image_dir: str | Path, page_name: str) -> Path:
    image_dir = Path(image_dir)
    path = image_dir / page_name
    if path.is_file():
        return path

    stem = Path(page_name).stem
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
        candidate = image_dir / f'{stem}{ext}'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'找不到原圖：{page_name}')


def _crop_box(image: Image.Image, xyxy: list[Any], pad: int) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'無效 xyxy_pixel：{xyxy}')
    return image.crop((x1, y1, x2, y2)).convert('RGB')


def _clean_ocr_text(text: str) -> str:
    text = text.strip()
    for prefix in ('Assistant:', 'assistant:', 'OCR:', 'Text:', '文本：', '文字：'):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def _require_transformers() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            '缺少 OCR 推理依賴。請先安裝：\n'
            '  .venv/bin/pip install transformers safetensors accelerate einops\n'
            f'原始錯誤：{exc}'
        ) from exc
    return torch, AutoModelForCausalLM, AutoProcessor


def _chunks(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


class MangaOCR:
    def __init__(
        self,
        model_dir: str | Path,
        device: str,
        dtype: str,
        max_new_tokens: int,
        prompt: str,
    ) -> None:
        torch, AutoModelForCausalLM, AutoProcessor = _require_transformers()
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True,
        )

        torch_dtype = self._resolve_dtype(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch_dtype,
        )
        self.model.to(device)
        self.model.eval()
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

    def _resolve_dtype(self, dtype: str) -> Any:
        if dtype == 'auto':
            if self.device == 'cuda':
                return self.torch.float16
            return self.torch.float32
        if dtype == 'float16':
            return self.torch.float16
        if dtype == 'bfloat16':
            return self.torch.bfloat16
        if dtype == 'float32':
            return self.torch.float32
        raise ValueError(f'未知 dtype：{dtype}')

    def _prompt_text(self) -> str:
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': self.prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def recognize(self, image: Image.Image) -> str:
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images: list[Image.Image]) -> list[str]:
        prompt_text = self._prompt_text()
        prompts = [prompt_text] * len(images)
        inputs = self.processor(images=images, text=prompts, return_tensors='pt', padding=True)
        inputs = {
            key: value.to(self.device) if hasattr(value, 'to') else value
            for key, value in inputs.items()
        }
        input_len = inputs['input_ids'].shape[-1]
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated = outputs[:, input_len:]
        texts = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [_clean_ocr_text(text) for text in texts]


def _iter_pages(measure: dict, page_filter: str | None) -> list[tuple[str, list[dict]]]:
    pages = measure.get('pages', {})
    if page_filter is not None:
        return [(page_filter, pages.get(page_filter, []))]
    return list(pages.items())


def run(
    measure_path: str,
    image_dir: str,
    output_path: str | None,
    model_dir: str,
    device: str,
    dtype: str,
    pad: int,
    max_new_tokens: int,
    prompt: str,
    page: str | None,
    limit_pages: int | None,
    limit_items: int | None,
    batch_size: int,
    save_crops: str | None,
    dry_run: bool,
) -> str:
    measure = _load_json(measure_path)
    output_path = output_path or osp.join(osp.dirname(measure_path), 'measure_ocr.json')
    pages = _iter_pages(measure, page)
    if limit_pages is not None:
        pages = pages[:limit_pages]

    save_crops_path = Path(save_crops) if save_crops else None
    if save_crops_path is not None:
        save_crops_path.mkdir(parents=True, exist_ok=True)

    ocr = None if dry_run else MangaOCR(model_dir, device, dtype, max_new_tokens, prompt)
    output = {'pages': {}}

    start_time = time.perf_counter()
    total_pages = len(pages)
    for page_index, (page_name, items) in enumerate(pages, start=1):
        page_start = time.perf_counter()
        image_path = _image_path_for_page(image_dir, page_name)
        image = Image.open(image_path).convert('RGB')
        output_items = []
        if limit_items is not None:
            items = items[:limit_items]
        print(f'[{page_index}/{total_pages}] OCR {page_name}: {len(items)} boxes')

        indexed_items = list(enumerate(items))
        for batch in _chunks(indexed_items, max(1, batch_size)):
            batch_indices = [index for index, _ in batch]
            batch_items = [item for _, item in batch]
            batch_crops = [_crop_box(image, item['xyxy_pixel'], pad) for item in batch_items]

            if save_crops_path is not None:
                for index, crop in zip(batch_indices, batch_crops):
                    crop_name = f'{Path(page_name).stem}-{index:03d}.png'
                    crop.save(save_crops_path / crop_name)

            if dry_run:
                batch_texts = [''] * len(batch_items)
                batch_errors = [None] * len(batch_items)
            else:
                try:
                    batch_texts = ocr.recognize_batch(batch_crops) if ocr is not None else [''] * len(batch_items)
                    batch_errors = [None] * len(batch_items)
                except Exception as exc:  # Keep page OCR running if one batch fails.
                    batch_texts = [''] * len(batch_items)
                    batch_errors = [str(exc)] * len(batch_items)

            for item, text, error in zip(batch_items, batch_texts, batch_errors):
                out_item = dict(item)
                out_item['ocr_text'] = text
                if error:
                    out_item['ocr_error'] = error
                output_items.append(out_item)
        output['pages'][page_name] = output_items
        page_elapsed = time.perf_counter() - page_start
        print(f'[{page_index}/{total_pages}] 完成 {page_name} ({page_elapsed:.1f}s)')

    _write_json(output_path, output)
    print(f'總耗時：{time.perf_counter() - start_time:.1f}s')
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Add OCR text to measure.json boxes with PaddleOCR-VL-For-Manga.',
    )
    parser.add_argument(
        'path1',
        help='Image folder, or path to ctd/measure.json when using the old two-argument form',
    )
    parser.add_argument(
        'path2',
        nargs='?',
        default=None,
        help='Optional image folder for old form: measure_ocr.py ctd/measure.json image_dir',
    )
    parser.add_argument('--measure-json', default=None, help='Default: <image_dir>/ctd/measure.json')
    parser.add_argument('--output', default=None, help='Default: measure_ocr.json next to measure.json')
    parser.add_argument('--model', default=str(DEFAULT_MODEL_DIR), help='PaddleOCR-VL model directory')
    parser.add_argument('--device', default='mps', help='cpu, cuda, mps, ...')
    parser.add_argument(
        '--dtype',
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help='Model dtype',
    )
    parser.add_argument('--pad', type=int, default=2, help='Crop padding in pixels')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--page', default=None, help='Only process one page, e.g. 241.png')
    parser.add_argument('--limit-pages', type=int, default=None)
    parser.add_argument('--limit-items', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1, help='Number of crops per generate call')
    parser.add_argument('--save-crops', default=None, help='Optional folder for crop QA images')
    parser.add_argument('--dry-run', action='store_true', help='Only crop/write JSON; do not load OCR model')
    args = parser.parse_args()
    measure_path, image_dir = _resolve_input_paths(args.path1, args.path2, args.measure_json)

    output = run(
        measure_path=measure_path,
        image_dir=image_dir,
        output_path=args.output,
        model_dir=args.model,
        device=args.device,
        dtype=args.dtype,
        pad=args.pad,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        page=args.page,
        limit_pages=args.limit_pages,
        limit_items=args.limit_items,
        batch_size=args.batch_size,
        save_crops=args.save_crops,
        dry_run=args.dry_run,
    )
    print(f'輸出：{output}')


if __name__ == '__main__':
    main()
