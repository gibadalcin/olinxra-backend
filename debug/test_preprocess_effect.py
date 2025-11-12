#!/usr/bin/env python3
"""Teste rápido: compara métricas (pHash hamming, MSE, embedding sim) antes/depois do preprocess.

Uso:
 python debug/test_preprocess_effect.py --query <path> --candidate <path> --bbox 25,0,475,338 --mode clahe --expand 20
"""
import argparse
import os
from PIL import Image
import imagehash
import numpy as np
try:
    from tools.preprocess_variants import preprocess_crop
except Exception:
    # fallback: load by path
    import importlib.util
    spec = importlib.util.spec_from_file_location('preprocess_variants', os.path.join(os.path.dirname(__file__), '..', 'tools', 'preprocess_variants.py'))
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    preprocess_crop = pv.preprocess_crop

try:
    from clip_utils import initialize_onnx_session, extract_clip_features
except Exception:
    import importlib.util
    clip_path = os.path.join(os.path.dirname(__file__), '..', 'clip_utils.py')
    spec = importlib.util.spec_from_file_location('clip_utils', clip_path)
    cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cu)
    initialize_onnx_session = cu.initialize_onnx_session
    extract_clip_features = cu.extract_clip_features


def make_thumb(img, size=160):
    im = img.copy()
    im.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new('RGB', (size, size), (255,255,255))
    x = (size - im.width)//2
    y = (size - im.height)//2
    bg.paste(im, (x,y))
    return bg


def mse(a, b):
    a = np.asarray(a).astype('float32')
    b = np.asarray(b).astype('float32')
    return float(np.mean((a - b) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--bbox', required=False)
    p.add_argument('--mode', default='none', choices=['none','equalize','clahe'])
    p.add_argument('--expand', type=float, default=0.0, help='percent to expand bbox, e.g. 10 or 20')
    args = p.parse_args()

    q = Image.open(args.query).convert('RGB')
    c = Image.open(args.candidate).convert('RGB')

    w,h = q.size
    if args.bbox:
        left, top, right, bottom = [int(x) for x in args.bbox.split(',')]
        bbox = (left, top, right, bottom)
    else:
        side = min(w,h)//2
        left = (w-side)//2
        top = (h-side)//2
        bbox = (left, top, left+side, top+side)

    # original crop
    orig_crop = q.crop(bbox)
    # preprocessed
    proc = preprocess_crop(q, bbox=bbox, expand_pct=(args.expand/100.0), mode=args.mode)

    # thumbs
    t_orig = make_thumb(orig_crop)
    t_proc = make_thumb(proc)
    t_cand = make_thumb(c)

    ph_orig = str(imagehash.phash(orig_crop))
    ph_proc = str(imagehash.phash(proc))
    ph_cand = str(imagehash.phash(c))
    ham_orig = imagehash.hex_to_hash(ph_orig) - imagehash.hex_to_hash(ph_cand)
    ham_proc = imagehash.hex_to_hash(ph_proc) - imagehash.hex_to_hash(ph_cand)

    mse_orig = mse(t_orig, t_cand)
    mse_proc = mse(t_proc, t_cand)

    # embeddings
    ort_sess = initialize_onnx_session()
    # save temporary crops
    tmp_o = 'debug/tmp_orig.jpg'
    tmp_p = 'debug/tmp_proc.jpg'
    os.makedirs('debug', exist_ok=True)
    orig_crop.save(tmp_o, format='JPEG', quality=90)
    proc.save(tmp_p, format='JPEG', quality=90)

    vec_o = extract_clip_features(tmp_o, ort_sess)
    vec_p = extract_clip_features(tmp_p, ort_sess)
    vec_c = extract_clip_features(args.candidate, ort_sess)

    def cos(a,b):
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        a = a / (np.linalg.norm(a)+1e-12)
        b = b / (np.linalg.norm(b)+1e-12)
        return float(np.dot(a,b))

    sim_orig = cos(vec_o, vec_c)
    sim_proc = cos(vec_p, vec_c)

    print('RESULTS')
    print('candidate phash:', ph_cand)
    print('orig phash:', ph_orig, 'hamming->cand:', ham_orig, 'mse_thumb:', mse_orig, 'emb_sim:', sim_orig)
    print('proc  phash:', ph_proc, 'hamming->cand:', ham_proc, 'mse_thumb:', mse_proc, 'emb_sim:', sim_proc)

    # save side-by-side for quick visual
    side = Image.new('RGB', (t_orig.width*3 + 20, t_orig.height + 20), (255,255,255))
    side.paste(t_orig, (10,10))
    side.paste(t_proc, (10 + t_orig.width + 5, 10))
    side.paste(t_cand, (10 + t_orig.width*2 + 10, 10))
    side.save('debug/preproc_compare_sidebyside.png')
    print('Wrote debug/preproc_compare_sidebyside.png')


if __name__ == '__main__':
    main()
