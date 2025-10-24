import os
import re
import tempfile
import base64
import hashlib
import asyncio
import logging
from datetime import datetime

import httpx
from fastapi import HTTPException
from PIL import Image as PILImage

from gcs_utils import upload_image_to_gcs, get_bucket
from glb_generator import generate_plane_glb

logger = logging.getLogger(__name__)


def _generate_signed_url_conteudo(gs_url, filename=None):
    """Generate signed URL for given gs:// path or filename using GCS client.
    This mirrors the behavior in main. If filename is omitted, try extracting from gs_url.
    """
    tipo_bucket = "conteudo"
    if gs_url.startswith("gs://olinxra-logos/"):
        tipo_bucket = "logos"
    elif gs_url.startswith("gs://olinxra-conteudo/"):
        tipo_bucket = "conteudo"
    elif filename and "conteudo" in filename:
        tipo_bucket = "conteudo"
    else:
        tipo_bucket = "logos"

    if not filename:
        if gs_url.startswith("gs://olinxra-conteudo/"):
            filename = gs_url[len("gs://olinxra-conteudo/"):]
        elif gs_url.startswith("gs://olinxra-logos/"):
            filename = gs_url[len("gs://olinxra-logos/"):]
        else:
            filename = gs_url.split("/")[-1]

    try:
        bucket = get_bucket(tipo_bucket)
        url = bucket.blob(filename).generate_signed_url(version="v4", expiration=3600, method="GET")
        logger.debug(f"Signed URL gerada para bucket={tipo_bucket} filename={filename}")
        return url
    except Exception as e:
        logger.error(f"Erro ao gerar signed URL (orchestrator) para {filename}: {e}")
        return ""


async def generate_glb_internal(image_url: str, owner_uid: str = 'anonymous', provided_filename: str = None, params: dict = None, contentId: str = None, db=None):
    """
    Orchestrator function: download/normalize image, generate .glb with generate_plane_glb,
    upload to GCS, optionally persist in modelos_ra (if db provided). Returns same dict shape
    as previous helper: { 'glb_signed_url', 'gs_url', 'cached', 'doc_id' }
    """
    params = params or {}
    try:
        plane_height = float(params.get('plane_height', params.get('planeHeight', 1.0)))
    except Exception:
        plane_height = 1.0
    try:
        base_height = float(params.get('height', params.get('height', 0.0)))
    except Exception:
        base_height = 0.0
    flip_u = bool(params.get('flip_u', params.get('flipU', True)))
    flip_v = bool(params.get('flip_v', params.get('flipV', True)))

    temp_image = None
    temp_glb = None
    doc_id = None
    try:
        ALLOWED_DOMAINS = [d.strip().lower() for d in os.getenv('GLB_ALLOWED_DOMAINS', '').split(',') if d.strip()]
        MAX_IMAGE_BYTES = int(os.getenv('GLB_MAX_IMAGE_BYTES', '5000000'))
        MAX_IMAGE_DIM = int(os.getenv('GLB_MAX_DIM', '2048'))

        from urllib.parse import urlparse
        parsed = None
        if image_url.startswith('data:'):
            m = re.match(r'data:(image/[^;]+);base64,(.+)', image_url, re.I)
            if not m:
                raise HTTPException(status_code=400, detail='Invalid data URL for image')
            mime = m.group(1).lower()
            b64 = m.group(2)
            if not mime.startswith('image'):
                raise HTTPException(status_code=400, detail='Data URL is not an image')
            try:
                img_bytes = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=400, detail='Failed to decode base64 image')
            src_hash = hashlib.sha256(img_bytes).hexdigest()
            if mime in ('image/jpeg', 'image/jpg'):
                ext = '.jpg'
            elif mime == 'image/png':
                ext = '.png'
            else:
                ext = '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
                temp_image = tf.name
                tf.write(img_bytes)
        else:
            parsed = urlparse(image_url)
            if parsed.scheme not in ('https',):
                raise HTTPException(status_code=400, detail='image_url must use https scheme')
            hostname = (parsed.hostname or '').lower()
            if ALLOWED_DOMAINS and hostname not in ALLOWED_DOMAINS:
                raise HTTPException(status_code=400, detail='image_url host not allowed')

            src_hash = hashlib.sha256(image_url.encode('utf-8')).hexdigest()

            async with httpx.AsyncClient(timeout=30.0) as client_http:
                async with client_http.stream('GET', image_url, follow_redirects=True, timeout=30.0) as resp:
                    if resp.status_code != 200:
                        raise HTTPException(status_code=400, detail='Failed to download image')
                    content_type = resp.headers.get('content-type', '')
                    if not content_type.startswith('image'):
                        raise HTTPException(status_code=400, detail='URL does not point to an image')
                    content_length = resp.headers.get('content-length')
                    if content_length and int(content_length) > MAX_IMAGE_BYTES:
                        raise HTTPException(status_code=413, detail='Image too large')

                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(parsed.path)[1] or '.jpg') as tf:
                        temp_image = tf.name
                        total = 0
                        async for chunk in resp.aiter_bytes():
                            total += len(chunk)
                            if total > MAX_IMAGE_BYTES:
                                raise HTTPException(status_code=413, detail='Image too large')
                            tf.write(chunk)

        def _resize_if_needed(src_path, max_dim):
            img = PILImage.open(src_path)
            w, h = img.size
            if max(w, h) > max_dim:
                ratio = max_dim / float(max(w, h))
                new_size = (int(w * ratio), int(h * ratio))
                img = img.convert('RGB')
                img = img.resize(new_size, PILImage.LANCZOS)
                dst = src_path + '.resized.jpg'
                img.save(dst, format='JPEG', quality=90)
                return dst
            return src_path

        processed_image = await asyncio.to_thread(_resize_if_needed, temp_image, MAX_IMAGE_DIM)

        params_string = f"{base_height}|{plane_height}|{flip_u}|{flip_v}"
        sha = hashlib.sha256((src_hash + '|' + params_string).encode('utf-8')).hexdigest()[:16]
        if provided_filename:
            safe_base = os.path.basename(provided_filename)
            filename = f"{owner_uid}/ra/{safe_base}"
        else:
            filename = f"{owner_uid}/ra/generated_{sha}.glb"

        bucket = get_bucket('conteudo')
        blob = bucket.blob(filename)
        exists = await asyncio.to_thread(blob.exists)
        if exists:
            gcs_path = f'gs://{bucket.name}/{filename}'
            signed = _generate_signed_url_conteudo(gcs_path, filename)
            return {'glb_signed_url': signed, 'gs_url': gcs_path, 'cached': True}

        with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as tg:
            temp_glb = tg.name

        await asyncio.to_thread(generate_plane_glb, processed_image, temp_glb, base_height, plane_height, flip_u, flip_v)

        gcs_path = await asyncio.to_thread(upload_image_to_gcs, temp_glb, filename, 'conteudo')
        signed = _generate_signed_url_conteudo(gcs_path, filename)

        if db is not None:
            try:
                ra_coll = db.get_collection('modelos_ra')
                doc = {
                    'owner_uid': owner_uid,
                    'gs_path': gcs_path,
                    'filename': os.path.basename(filename),
                    'source': {'type': 'data' if image_url.startswith('data:') else 'url', 'value': src_hash if image_url.startswith('data:') else image_url},
                    'params': {'height': base_height, 'plane_height': plane_height, 'flip_u': flip_u, 'flip_v': flip_v},
                    'public': False,
                    'contentId': contentId,
                    'created_at': datetime.utcnow()
                }
                res = await ra_coll.insert_one(doc)
                doc_id = str(res.inserted_id)
            except Exception:
                doc_id = None

        return {'glb_signed_url': signed, 'gs_url': gcs_path, 'cached': False, 'doc_id': doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('Erro gerando GLB (orchestrator): %s', e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if temp_image and os.path.exists(temp_image):
                os.remove(temp_image)
        except Exception:
            pass
        try:
            if temp_glb and os.path.exists(temp_glb):
                os.remove(temp_glb)
        except Exception:
            pass
