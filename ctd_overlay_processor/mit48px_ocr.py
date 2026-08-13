"""Small adapter for BallonsTranslator's mit48px CTC OCR weights."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import einops
import numpy as np
import torch


DEFAULT_BALLONS_ROOT = Path('/Users/zhongsheng/Documents/comic_translate/BallonsTranslator')
DEFAULT_MODEL_PATH = DEFAULT_BALLONS_ROOT / 'data' / 'models' / 'mit48pxctc_ocr.ckpt'
DEFAULT_ALPHABET_PATH = DEFAULT_BALLONS_ROOT / 'data' / 'alphabet-all-v5.txt'
DEFAULT_IMPLEMENTATION_PATH = (
    DEFAULT_BALLONS_ROOT / 'ballontranslator' / 'modules' / 'ocr' / 'mit48px_ctc.py'
)


def resolved_device(requested: str) -> str:
    requested = str(requested or 'cpu').lower()
    if requested == 'mps':
        return 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    if requested == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cpu'


def _load_ocr_class(implementation_path: Path):
    if not implementation_path.is_file():
        raise FileNotFoundError(f'找不到 mit48px CTC 實作：{implementation_path}')

    # The source file imports TextBlock for annotations only. Supplying a tiny
    # module avoids importing the whole BallonsTranslator application and all of
    # its unrelated GUI/document dependencies.
    module_names = ('ballontranslator', 'ballontranslator.utils', 'ballontranslator.utils.textblock')
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        package = types.ModuleType('ballontranslator')
        package.__path__ = []
        utils = types.ModuleType('ballontranslator.utils')
        utils.__path__ = []
        textblock = types.ModuleType('ballontranslator.utils.textblock')
        textblock.TextBlock = type('TextBlock', (), {})
        sys.modules['ballontranslator'] = package
        sys.modules['ballontranslator.utils'] = utils
        sys.modules['ballontranslator.utils.textblock'] = textblock

        spec = importlib.util.spec_from_file_location('_ctd_mit48px_ctc', implementation_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'無法載入 mit48px CTC 實作：{implementation_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.OCR
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def prepare_character_crop(image_rgb: np.ndarray, bbox: list[Any], pad: int = 2) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'無效單字框：{bbox}')
    crop = image_rgb[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    target_h = 48
    target_w = max(4, int(round(target_h * crop_w / max(1, crop_h))))
    interpolation = cv2.INTER_CUBIC if target_h > crop_h else cv2.INTER_AREA
    return cv2.resize(crop, (target_w, target_h), interpolation=interpolation)


class Mit48pxCtcOcr:
    def __init__(
        self,
        device: str = 'cpu',
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        alphabet_path: str | Path = DEFAULT_ALPHABET_PATH,
        implementation_path: str | Path = DEFAULT_IMPLEMENTATION_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.alphabet_path = Path(alphabet_path)
        self.implementation_path = Path(implementation_path)
        for path in (self.model_path, self.alphabet_path):
            if not path.is_file():
                raise FileNotFoundError(f'找不到 mit48px CTC 檔案：{path}')

        self.device = resolved_device(device)
        dictionary = self.alphabet_path.read_text(encoding='utf-8').splitlines()
        ocr_class = _load_ocr_class(self.implementation_path)
        self.net = ocr_class(dictionary, 768)
        state = torch.load(self.model_path, map_location='cpu')
        weights = state['model'] if isinstance(state, dict) and 'model' in state else state
        for key in (
            'encoders.layers.0.pe.pe',
            'encoders.layers.1.pe.pe',
            'encoders.layers.2.pe.pe',
        ):
            weights.pop(key, None)
        self.net.load_state_dict(weights, strict=False)
        self.net.eval()
        if self.device != 'cpu':
            self.net = self.net.to(self.device)

    @torch.inference_mode()
    def recognize_batch(self, crops: list[np.ndarray], batch_size: int = 32) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(crops), max(1, batch_size)):
            batch = crops[start:start + max(1, batch_size)]
            widths = [int(crop.shape[1]) for crop in batch]
            max_width = (4 * (max(widths) + 7) // 4) + 128
            region = np.zeros((len(batch), 48, max_width, 3), dtype=np.uint8)
            for index, crop in enumerate(batch):
                region[index, :, :crop.shape[1], :] = crop
            images = (torch.from_numpy(region).float() - 127.5) / 127.5
            images = einops.rearrange(images, 'N H W C -> N C H W')
            if self.device != 'cpu':
                images = images.to(self.device)
            decoded = self.net.decode(images, widths, 0)
            for sequence in decoded:
                chars = []
                probabilities = []
                for character_id, log_probability, *_ in sequence:
                    character = self.net.dictionary[character_id]
                    if character == '<SP>':
                        character = ' '
                    chars.append(character)
                    probabilities.append(math.exp(float(log_probability)))
                probability = (
                    float(math.exp(sum(math.log(max(value, 1e-12)) for value in probabilities) / len(probabilities)))
                    if probabilities
                    else 0.0
                )
                results.append({
                    'text': ''.join(chars),
                    'probability': round(probability, 6),
                    'character_probabilities': [round(value, 6) for value in probabilities],
                })
        return results

