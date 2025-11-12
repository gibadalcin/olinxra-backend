#!/usr/bin/env python3
"""
Mini ferramenta para gerar miniaturas e comparar visualmente duas imagens.

Uso:
  python tools/visual_compare_thumbnail.py --query path/to/query.jpg --candidate path/to/candidate.jpg --out out.png

Saída:
  - salva imagem lado-a-lado com métricas (pHash, hamming, MSE)
  - imprime métricas no stdout

Dependências: Pillow, imagehash, numpy
"""
from PIL import Image, ImageDraw, ImageFont
import imagehash
import argparse
import numpy as np
import os


def mse(a: np.ndarray, b: np.ndarray) -> float:
    # mean squared error between two images (assumes same shape)
    err = np.mean((a.astype('float32') - b.astype('float32')) ** 2)
    return float(err)


def make_thumb(img: Image.Image, size: int = 128) -> Image.Image:
    # preserve aspect ratio, fit into square and pad with white background
    img = img.convert('RGBA')
    img.thumbnail((size, size), Image.LANCZOS)
    thumb = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    thumb.paste(img, (x, y), img)
    return thumb.convert('RGB')


def draw_metrics(side_by_side: Image.Image, metrics_text: str) -> Image.Image:
    draw = ImageDraw.Draw(side_by_side)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # place text at the bottom area
    w, h = side_by_side.size
    padding = 6
    # semi-transparent rectangle as background for text
    text_h = 14 * (metrics_text.count('\n') + 1) + padding * 2
    overlay = Image.new('RGBA', (w, text_h), (255, 255, 255, 220))
    side_by_side.paste(overlay, (0, h - text_h))
    draw = ImageDraw.Draw(side_by_side)
    draw.text((padding, h - text_h + padding), metrics_text, fill='black', font=font)
    return side_by_side


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True, help='path to query image')
    p.add_argument('--candidate', required=True, help='path to candidate image')
    p.add_argument('--bbox', help='optional bbox for query crop: left,top,right,bottom', default=None)
    p.add_argument('--thumb-size', type=int, default=128, help='thumbnail size in px (square)')
    p.add_argument('--out', default='thumb_compare.png', help='output image path')
    p.add_argument('--message', default=None, help='optional recognition message to overlay')
    args = p.parse_args()

    if not os.path.exists(args.query):
        raise FileNotFoundError(args.query)
    if not os.path.exists(args.candidate):
        raise FileNotFoundError(args.candidate)

    q = Image.open(args.query).convert('RGB')
    c = Image.open(args.candidate).convert('RGB')

    if args.bbox:
        try:
            left, top, right, bottom = [int(x) for x in args.bbox.split(',')]
            q = q.crop((left, top, right, bottom))
        except Exception as e:
            print('warning: invalid bbox, ignoring:', e)

    thumb_q = make_thumb(q, size=args.thumb_size)
    thumb_c = make_thumb(c, size=args.thumb_size)

    # compute pHash and hamming
    ph_q = imagehash.phash(q)
    ph_c = imagehash.phash(c)
    hamming = ph_q - ph_c

    # compute MSE on the thumbnails (RGB)
    a = np.asarray(thumb_q).astype('float32')
    b = np.asarray(thumb_c).astype('float32')
    the_mse = mse(a, b)

    # prepare side-by-side image
    spacing = 10
    w = thumb_q.width + thumb_c.width + spacing
    h = max(thumb_q.height, thumb_c.height) + 60
    out_img = Image.new('RGB', (w, h), (255, 255, 255))
    out_img.paste(thumb_q, (0, 0))
    out_img.paste(thumb_c, (thumb_q.width + spacing, 0))

    metrics_text = f"query: {os.path.basename(args.query)}\ncandidate: {os.path.basename(args.candidate)}\nphash_q: {str(ph_q)}\nphash_c: {str(ph_c)}\nhamming: {hamming}\nMSE (thumb): {the_mse:.2f}\nthumb_size: {args.thumb_size}px"
    if args.message:
        metrics_text = f"{metrics_text}\n\nRECOG: {args.message}"

    out_img = draw_metrics(out_img, metrics_text)
    out_img.save(args.out)

    print('Saved comparison image to', args.out)
    print('Metrics:')
    print(metrics_text)


if __name__ == '__main__':
    main()
