#!/usr/bin/env python3
"""
Gerar variantes de pré-processamento da imagem query e comparar com o candidato.

Funcionalidades:
- Aplica crops: bbox original, expand +10% e +20%, center crop e full image
- Para cada variante aplica: nenhum, equalize (PIL) e CLAHE (se OpenCV disponível)
- Calcula pHash e MSE contra a imagem candidata
- Opcional: tenta calcular embeddings usando `clip_utils.extract_clip_features` se --emb for passado
- Salva imagens e um relatório JSON/PNG em `--out-dir`

Uso exemplo:
 python tools/preprocess_variants.py --query C:\...\c_capturada.jpg --candidate C:\...\chevrolet.png --bbox 25,0,475,338 --out-dir .\debug\preproc

Dependências: pillow, imagehash, numpy. OpenCV é opcional (para CLAHE).
"""
from PIL import Image, ImageOps
import imagehash
import argparse
import os
import json
import numpy as np
from typing import Tuple

try:
    import cv2
    _has_cv2 = True
except Exception:
    _has_cv2 = False


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype('float32') - b.astype('float32')) ** 2))


def expand_bbox(bbox: Tuple[int,int,int,int], expand_ratio: float, w: int, h: int) -> Tuple[int,int,int,int]:
    left, top, right, bottom = bbox
    cw = right - left
    ch = bottom - top
    expand_w = int(cw * expand_ratio)
    expand_h = int(ch * expand_ratio)
    nl = max(0, left - expand_w)
    nt = max(0, top - expand_h)
    nr = min(w, right + expand_w)
    nb = min(h, bottom + expand_h)
    return (nl, nt, nr, nb)


def apply_clahe_pil(img: Image.Image) -> Image.Image:
    # Try to apply CLAHE using OpenCV if available; fallback to ImageOps.equalize
    if _has_cv2:
        arr = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(arr)
        return Image.fromarray(cl).convert('RGB')
    else:
        return ImageOps.equalize(img.convert('RGB'))


def preprocess_crop(img: Image.Image, bbox: Tuple[int,int,int,int]=None, expand_pct: float=0.0, mode: str='none') -> Image.Image:
    """
    Apply optional bbox expansion and preprocessing mode to a PIL image and return the processed crop (RGB).

    Args:
        img: source PIL Image (full image)
        bbox: optional (left,top,right,bottom). If None, operates on full image.
        expand_pct: fraction (0.0-1.0) to expand bbox before cropping
        mode: 'none' | 'equalize' | 'clahe'

    Returns:
        Processed PIL.Image in RGB mode.
    """
    im = img
    w, h = im.size
    if bbox is not None:
        try:
            left, top, right, bottom = bbox
            if expand_pct and expand_pct > 0.0:
                bbox = expand_bbox((left, top, right, bottom), expand_pct, w, h)
                left, top, right, bottom = bbox
            im = im.crop((left, top, right, bottom))
        except Exception:
            # fallback to full image
            im = img

    # ensure RGB
    im = im.convert('RGB')

    if mode is None:
        mode = 'none'

    mode = mode.lower()
    if mode == 'equalize':
        return ImageOps.equalize(im)
    if mode == 'clahe':
        return apply_clahe_pil(im)
    # default: none
    return im


def save_img(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def compute_phash(img: Image.Image):
    return str(imagehash.phash(img))


def try_extract_embedding(path: str):
    try:
        # import lazily to avoid hard dependency
        from clip_utils import extract_clip_features
        # extract_clip_features may expect an ort session; try calling with path only
        vec = extract_clip_features(path)
        return vec.tolist() if hasattr(vec, 'tolist') else list(vec)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--bbox', required=False, help='left,top,right,bottom')
    p.add_argument('--out-dir', default='./debug/preproc')
    p.add_argument('--thumb-size', type=int, default=160)
    p.add_argument('--emb', action='store_true', help='try to compute embeddings (optional)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    q = Image.open(args.query).convert('RGB')
    c = Image.open(args.candidate).convert('RGB')
    w, h = q.size

    if args.bbox:
        left, top, right, bottom = [int(x) for x in args.bbox.split(',')]
        base_bbox = (left, top, right, bottom)
    else:
        # default center crop 50%
        side = min(w, h)//2
        left = (w - side)//2
        top = (h - side)//2
        base_bbox = (left, top, left+side, top+side)

    variants = []
    # crops: base, +10%, +20%, center, full
    variants.append(('base', base_bbox))
    variants.append(('expand_10', expand_bbox(base_bbox, 0.10, w, h)))
    variants.append(('expand_20', expand_bbox(base_bbox, 0.20, w, h)))
    # center crop same side as base
    side = base_bbox[2] - base_bbox[0]
    leftc = max(0, (w - side)//2)
    topc = max(0, (h - side)//2)
    variants.append(('center', (leftc, topc, leftc+side, topc+side)))
    variants.append(('full', (0,0,w,h)))

    results = []

    # compute candidate thumb for MSE
    def make_thumb(img):
        im = img.copy()
        im.thumbnail((args.thumb_size, args.thumb_size), Image.LANCZOS)
        bg = Image.new('RGB', (args.thumb_size, args.thumb_size), (255,255,255))
        x = (args.thumb_size - im.width)//2
        y = (args.thumb_size - im.height)//2
        bg.paste(im, (x,y))
        return bg

    c_thumb = make_thumb(c)
    c_thumb_arr = np.asarray(c_thumb).astype('float32')

    for name, bbox in variants:
        left, top, right, bottom = bbox
        cropped = q.crop((left, top, right, bottom))
        # preprocessing modes
        modes = [('orig', cropped), ('equalize', ImageOps.equalize(cropped.convert('RGB')))]
        if _has_cv2:
            modes.append(('clahe', apply_clahe_pil(cropped)))

        for mode_name, img in modes:
            ph = compute_phash(img)
            thumb = make_thumb(img)
            mse_val = mse(np.asarray(thumb).astype('float32'), c_thumb_arr)
            out_name = f"{name}_{mode_name}.png"
            out_path = os.path.join(args.out_dir, out_name)
            save_img(thumb, out_path)

            emb = None
            emb_sim = None
            if args.emb:
                # try compute embeddings for both (may fail)
                try:
                    q_emb = try_extract_embedding(out_path)
                    c_emb = try_extract_embedding(args.candidate)
                    if q_emb is not None and c_emb is not None:
                        # cosine similarity
                        qa = np.array(q_emb)
                        ca = np.array(c_emb)
                        # normalize
                        qa = qa / (np.linalg.norm(qa)+1e-12)
                        ca = ca / (np.linalg.norm(ca)+1e-12)
                        emb_sim = float(np.dot(qa, ca))
                except Exception:
                    emb_sim = None

            results.append({
                'variant': name,
                'mode': mode_name,
                'bbox': bbox,
                'out_thumb': out_path,
                'phash': ph,
                'mse_thumb': mse_val,
                'emb_sim': emb_sim,
            })

    report = {
        'query': args.query,
        'candidate': args.candidate,
        'has_cv2': _has_cv2,
        'results': results,
    }

    with open(os.path.join(args.out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('Wrote report to', args.out_dir)


if __name__ == '__main__':
    main()
