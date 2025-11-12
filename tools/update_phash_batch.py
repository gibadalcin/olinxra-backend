"""
Script para calcular e persistir pHash em documentos da coleção `logos` que ainda
não possuem o campo `phash`.

Uso:
  python tools/update_phash_batch.py --limit 100 --dry-run

Requisitos: definir variáveis de ambiente MONGO_URI e (opcional) MONGO_DB_NAME.
"""
import os
import asyncio
import logging
import argparse
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

# Garantir que o diretório raiz do projeto esteja no sys.path para que
# importações de módulos locais (ex: gcs_utils.py) funcionem mesmo quando
# o script é executado a partir da pasta `tools/`.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motor.motor_asyncio import AsyncIOMotorClient
import httpx
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

async def download_bytes_from_gs(cand_url):
    # cand_url expected like gs://bucket/path
    from gcs_utils import get_bucket
    without_prefix = cand_url[len('gs://'):]
    bucket_name, _, path = without_prefix.partition('/')
    bucket = get_bucket('logos')
    blob = bucket.blob(path)
    return await asyncio.to_thread(blob.download_as_bytes)

async def download_bytes(cand):
    # cand can be a url or filename
    if isinstance(cand, str):
        if cand.startswith('gs://'):
            try:
                return await download_bytes_from_gs(cand)
            except Exception:
                logging.exception('Erro ao baixar gs:// URL')
                return None
        if cand.startswith('http://') or cand.startswith('https://'):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(cand)
                    r.raise_for_status()
                    return r.content
            except Exception:
                logging.exception('Erro ao baixar URL http(s)')
                return None
    # otherwise assume it's a filename in bucket
    try:
        from gcs_utils import get_bucket
        bucket = get_bucket('logos')
        blob = bucket.blob(cand)
        return await asyncio.to_thread(blob.download_as_bytes)
    except Exception:
        logging.exception('Erro ao baixar por filename do bucket')
        return None

async def process_one(doc, logos_collection, dry_run=False):
    _id = doc.get('_id')
    nome = doc.get('nome')
    logging.info(f'Processando _id={_id} nome={nome}')

    # prefer existing url fields
    cand_url = doc.get('url') or doc.get('gs_url') or doc.get('gcs_url')
    filename = doc.get('filename')

    data = None
    if cand_url:
        data = await download_bytes(cand_url)
    if data is None and filename:
        data = await download_bytes(filename)
    if data is None:
        logging.warning(f'Não foi possível obter bytes para _id={_id} (pular)')
        return False

    try:
        img = PILImage.open(BytesIO(data)).convert('RGB')
    except Exception:
        logging.exception('Erro ao abrir imagem como PIL')
        return False

    try:
        import imagehash
        ph = imagehash.phash(img)
        ph_hex = str(ph)
    except Exception:
        logging.exception('Erro ao calcular pHash')
        return False

    logging.info(f'Computed pHash={ph_hex} for _id={_id} nome={nome}')

    if dry_run:
        return True

    try:
        # Log atual (se existir) para auditoria
        existing = await logos_collection.find_one({'_id': _id}, {'phash': 1})
        existing_ph = existing.get('phash') if existing else None
        logging.info(f'Existing phash for _id={_id}: {existing_ph}')

        result = await logos_collection.update_one({'_id': _id}, {'$set': {'phash': ph_hex}})
        # Resultado do update para confirmar
        try:
            matched = getattr(result, 'matched_count', None)
            modified = getattr(result, 'modified_count', None)
            logging.info(f'Update result for _id={_id}: matched={matched} modified={modified}')
        except Exception:
            logging.info(f'Update completed for _id={_id} (no detailed result available)')

        return True
    except Exception:
        logging.exception('Erro ao atualizar documento no Mongo')
        return False

async def main(args):
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        raise RuntimeError('MONGO_URI não definido')
    db_name = os.getenv('MONGO_DB_NAME', 'olinxra')

    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    logos = db['logos']

    query = {'$or': [{'phash': {'$exists': False}}, {'phash': None}, {'phash': ''}]}
    cursor = logos.find(query)
    count = 0
    updated = 0

    async for doc in cursor:
        if args.limit and count >= args.limit:
            break
        ok = await process_one(doc, logos, dry_run=args.dry_run)
        count += 1
        if ok and not args.dry_run:
            updated += 1
        await asyncio.sleep(args.delay)

    logging.info(f'Done. Processed={count} updated={updated} (dry_run={args.dry_run})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000, help='Número máximo de documentos a processar')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay entre documentos (evita burst sobre GCS/Mongo)')
    # Segurança: por padrão o script roda em modo dry-run. Passe --apply para persistir no DB.
    parser.add_argument('--dry-run', action='store_true', help='Somente computa pHash sem salvar no DB (deprecated, continua suportado)')
    parser.add_argument('--apply', action='store_true', help='Persistir pHash no MongoDB (necessário para gravação)')
    args = parser.parse_args()

    # Normalizar flags: --apply ativa escrita; caso o usuário passe explicitamente
    # --dry-run e --apply ao mesmo tempo, abortamos para evitar ambiguidade.
    if args.dry_run and args.apply:
        parser.error('Não use --dry-run e --apply juntos')
    # Segurança: se --apply não for passado, mantemos dry_run=True por padrão.
    if not args.apply:
        args.dry_run = True

    try:
        asyncio.run(main(args))
    except Exception:
        logging.exception('Execução abortada')
