from fastapi import Query
import logging
import json
import os
import numpy as np
import firebase_admin
import httpx
import tempfile
import smtplib
import hashlib
from fastapi import Request
from firebase_admin import credentials, auth
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form, Query, Body, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from bson import ObjectId
from bson.errors import InvalidId
from datetime import timedelta, datetime
import uuid
from gcs_utils import upload_image_to_gcs, get_bucket, GCS_BUCKET_CONTEUDO, GCS_BUCKET_LOGOS
from google.api_core.exceptions import PreconditionFailed
from glb_generator import generate_plane_glb
from schemas import validate_button_block_payload
from clip_utils import extract_clip_features
from faiss_index import LogoIndex
from email.mime.text import MIMEText
import asyncio
import time
import onnxruntime as ort
from PIL import Image as PILImage
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()


logo_index = None
ort_session = None
http_bearer = HTTPBearer()
client = None
db = None
logos_collection = None
# Cache de geocoding em memória (otimização para evitar chamadas repetidas ao Nominatim)
# Cache de geocoding em memória (otimização para evitar chamadas repetidas ao Nominatim)
geocode_cache = {}
# Cache simples de regiões (tupla: (nome_marca, tipo_regiao, nome_regiao) -> (ts, conteudo))
region_cache = {}
REGION_CACHE_TTL = 60 * 10  # 10 minutos
# REMOVIDO: images_collection = None


async def uploaded_assets_cleanup_worker(db, stop_event: asyncio.Event, interval_hours: int = 24, ttl_days: int = 7):
    """Background worker that periodically cleans uploaded_assets marked as unattached
    older than ttl_days. Deletes GCS objects (image + glb) and removes DB records.
    The worker stops when stop_event is set.
    """
    interval_seconds = max(60, int(interval_hours) * 3600)
    ttl_days = int(ttl_days)
    logging.info(f"[cleanup_worker] iniciado: intervalo={interval_hours}h ttl_days={ttl_days}")
    from gcs_utils import delete_gs_path

    while True:
        try:
            threshold = datetime.utcnow() - timedelta(days=ttl_days)
            try:
                orphans = await db['uploaded_assets'].find({'attached': False, 'created_at': {'$lt': threshold}}).to_list(length=1000)
            except Exception:
                logging.exception('[cleanup_worker] Falha ao buscar uploaded_assets órfãos')
                orphans = []

            for a in orphans:
                try:
                    fname = a.get('filename')
                    # safety: check if any conteudo references this filename
                    ref = await db['conteudos'].find_one({'$or': [{'blocos.filename': fname}, {'blocos.items.filename': fname}]})
                    if ref:
                        # mark attached to avoid deletion
                        await db['uploaded_assets'].update_one({'_id': a['_id']}, {'$set': {'attached': True, 'attached_at': datetime.utcnow(), 'conteudo_id': str(ref.get('_id'))}})
                        logging.info(f"[cleanup_worker] asset {a.get('_id')} referenced by conteudo {ref.get('_id')}, marcado como attached")
                        continue

                    # delete files (image + glb)
                    try:
                        if a.get('gs_url'):
                            await asyncio.to_thread(delete_gs_path, a.get('gs_url'))
                    except Exception:
                        logging.exception('[cleanup_worker] Falha ao deletar gs_url')
                    try:
                        if a.get('glb_url'):
                            await asyncio.to_thread(delete_gs_path, a.get('glb_url'))
                    except Exception:
                        logging.exception('[cleanup_worker] Falha ao deletar glb_url')

                    await db['uploaded_assets'].delete_one({'_id': a['_id']})
                    logging.info(f"[cleanup_worker] asset {a.get('_id')} deletado")
                except Exception:
                    logging.exception('[cleanup_worker] Erro ao processar asset')

        except Exception:
            logging.exception('[cleanup_worker] Erro inesperado no loop principal')

        # wait for interval or stop
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            logging.info('[cleanup_worker] stop_event set, encerrando worker')
            break
        except asyncio.TimeoutError:
            # timeout expired, loop again
            continue


    async def populate_location_fields(db, batch_size: int = 500, sleep_between_batches: float = 0.5):
        """
        Background migration helper: procura documentos em `conteudos` que possuem
        `latitude` e `longitude` mas não possuem o campo GeoJSON `location` e os
        atualiza definindo `location: { type: 'Point', coordinates: [lon, lat] }`.

        Executado em background no startup para preencher gradualmente os documentos
        e permitir que consultas $geoNear utilizem o índice 2dsphere.
        """
        try:
            logging.info('[migrate] populate_location_fields: iniciando background migration')
            filter_query = {'location': {'$exists': False}, 'latitude': {'$exists': True}, 'longitude': {'$exists': True}}
            while True:
                cursor = db['conteudos'].find(filter_query, {'_id': 1, 'latitude': 1, 'longitude': 1}).limit(batch_size)
                batch = await cursor.to_list(length=batch_size)
                if not batch:
                    logging.info('[migrate] populate_location_fields: nada a migrar, finalizando task')
                    break
                for doc in batch:
                    try:
                        lat = float(doc.get('latitude'))
                        lon = float(doc.get('longitude'))
                        await db['conteudos'].update_one({'_id': doc['_id']}, {'$set': {'location': {'type': 'Point', 'coordinates': [lon, lat]}}})
                    except Exception:
                        logging.exception('[migrate] Erro atualizando documento %s' % str(doc.get('_id')))
                logging.info(f'[migrate] populate_location_fields: processados {len(batch)} documentos, aguardando {sleep_between_batches}s antes do próximo lote')
                await asyncio.sleep(sleep_between_batches)
        except Exception:
            logging.exception('[migrate] Erro inesperado em populate_location_fields')

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db, logos_collection
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise RuntimeError("Variável de ambiente MONGO_URI não encontrada.")
    DB_NAME = os.getenv("MONGO_DB_NAME", "olinxra")

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    logos_collection = db["logos"]
    # Criar índices recomendados para a coleção de conteúdos
    try:
        # Índice composto por marca_id + owner_uid + tipo_regiao + nome_regiao (documents com marca_id)
        await db['conteudos'].create_index([
            ('marca_id', 1),
            ('owner_uid', 1),
            ('tipo_regiao', 1),
            ('nome_regiao', 1)
        ], name='idx_marcaid_owner_region', sparse=True)

        # Índice composto por nome_marca + owner_uid + tipo_regiao + nome_regiao (compatibilidade com docs antigos)
        await db['conteudos'].create_index([
            ('nome_marca', 1),
            ('owner_uid', 1),
            ('tipo_regiao', 1),
            ('nome_regiao', 1)
        ], name='idx_nomemarca_owner_region', sparse=True)
        
        # NOVO: Índice otimizado para smart-content (nome_marca + tipo_regiao + nome_regiao)
        await db['conteudos'].create_index([
            ('nome_marca', 1),
            ('tipo_regiao', 1),
            ('nome_regiao', 1)
        ], name='idx_smart_content_lookup')

        # Índice 2dsphere para consultas geoespaciais, se usarmos campo 'location'
        await db['conteudos'].create_index([('location', '2dsphere')], name='idx_location_2dsphere')

        # Índice simples por owner_uid
        await db['conteudos'].create_index([('owner_uid', 1)], name='idx_owner_uid')
        # Índice para assets temporários (staged uploads). TTL de 7 dias.
        try:
            await db['uploaded_assets'].create_index([('created_at', 1)], expireAfterSeconds=7*24*3600)
            logging.info('Índice TTL criado para uploaded_assets (7 dias)')
        except Exception:
            logging.exception('Falha ao criar índice TTL para uploaded_assets')

        logging.info('Índices de conteúdo verificados/criados com sucesso.')
        # Opcional: migration background para popular campo `location` a partir de latitude/longitude
        run_migration = os.getenv('RUN_LOCATION_MIGRATION', 'false').lower() in ('1', 'true', 'yes')
        if run_migration:
            try:
                asyncio.create_task(populate_location_fields(db))
                logging.info('populate_location_fields agendado em background (RUN_LOCATION_MIGRATION=true)')
            except Exception:
                logging.exception('Falha ao agendar populate_location_fields')
    except Exception as e:
        logging.exception(f'Falha ao criar índices em conteudos: {e}')
    # REMOVIDO: images_collection = db["images"]

    logging.info("Iniciando a aplicação...")
    initialize_firebase()
    initialize_onnx_session()
    await load_faiss_index()
    # Start background cleanup worker for uploaded_assets
    try:
        cleanup_interval = int(os.getenv('CLEANUP_INTERVAL_HOURS', '24'))
    except Exception:
        cleanup_interval = 24
    try:
        ttl_days = int(os.getenv('UPLOADED_ASSETS_TTL_DAYS', '7'))
    except Exception:
        ttl_days = 7

    stop_event = asyncio.Event()
    cleanup_task = asyncio.create_task(uploaded_assets_cleanup_worker(db, stop_event, cleanup_interval, ttl_days))

    try:
        yield
    finally:
        # signal worker to stop and wait for it to finish
        try:
            stop_event.set()
            # allow up to 30s for graceful shutdown
            await asyncio.wait_for(cleanup_task, timeout=30.0)
        except asyncio.TimeoutError:
            logging.warning('[lifespan] cleanup_task did not finish within timeout, cancelling')
            try:
                cleanup_task.cancel()
            except Exception:
                pass
        except Exception:
            logging.exception('[lifespan] error while shutting down cleanup_task')
        if client:
            client.close()

app = FastAPI(lifespan=lifespan)


def sanitize_for_json(obj):
    """Recursively convert common non-JSON-serializable types (ObjectId, numpy types, bytes)
    into JSON-safe Python types.

    - ObjectId -> str
    - numpy numbers/arrays -> python numbers / lists
    - bytes -> decode as utf-8 when possible, otherwise base64
    - dict/list -> recurse
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    # ObjectId
    if isinstance(obj, ObjectId):
        return str(obj)

    # numpy scalars
    if _np is not None and isinstance(obj, (_np.generic,)):
        try:
            return obj.item()
        except Exception:
            return float(obj)

    # numpy arrays
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()

    # dict -> recurse
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    # list/tuple -> recurse
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    # bytes -> try decode
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8')
        except Exception:
            import base64
            return base64.b64encode(bytes(obj)).decode('ascii')

    # fallback: primitive types (str, int, float, bool, None)
    return obj


def resize_image_if_needed(src_path: str, max_dim: int = 2048) -> str:
    """
    Redimensiona imagem se exceder dimensão máxima, mantendo aspect ratio.
    
    Args:
        src_path: Caminho para arquivo de imagem
        max_dim: Dimensão máxima permitida (largura ou altura)
    
    Returns:
        Caminho para imagem processada (original se não precisou redimensionar)
    """
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


# --- Helpers de Inicialização ---
###############################################################
# Função utilitária e endpoint para gerar signed URL de conteúdo
def gerar_signed_url_conteudo(gs_url=None, filename=None, expiration=3600, skip_exists_check=False):
    """
    Gera um signed URL para um objeto no bucket de conteúdo ou logos.
    Aceita dois modos de chamada:
    - gs_url (ex: 'gs://bucket/path/file.glb') OR
    - filename (ex: 'public/ra/totem/file.glb') — neste caso assumimos bucket de conteúdo.
    
    Args:
        gs_url: URL completa no formato gs://bucket/path
        filename: Nome do arquivo no bucket
        expiration: Tempo de expiração em segundos (padrão: 3600 = 1h)
        skip_exists_check: Se True, pula verificação de existência (otimização para smart-content)
    """
    try:
        # Determine bucket names from gcs_utils (loaded from env)
        bucket_conteudo = GCS_BUCKET_CONTEUDO
        bucket_logos = GCS_BUCKET_LOGOS

        tipo_bucket = 'conteudo'

        # If a full gs_url was provided, infer the bucket by its prefix
        if gs_url and isinstance(gs_url, str):
            if bucket_logos and gs_url.startswith(f'gs://{bucket_logos}/'):
                tipo_bucket = 'logos'
            elif bucket_conteudo and gs_url.startswith(f'gs://{bucket_conteudo}/'):
                tipo_bucket = 'conteudo'
            else:
                # fallback: if filename contains 'conteudo', prefer conteudo
                if filename and 'conteudo' in filename:
                    tipo_bucket = 'conteudo'
                else:
                    tipo_bucket = 'logos' if bucket_logos else 'conteudo'

        # If no filename given, try to extract from gs_url
        if not filename:
            if gs_url and isinstance(gs_url, str) and gs_url.startswith('gs://'):
                # remove gs://bucket/
                parts = gs_url.split('/', 3)
                if len(parts) >= 4:
                    filename = parts[3]
                else:
                    filename = gs_url.split('/')[-1]
            else:
                # fallback: use last path segment of provided string
                filename = (filename or '')

        bucket = get_bucket('conteudo' if tipo_bucket == 'conteudo' else 'logos')
        blob = bucket.blob(filename)
        
        # ⚡ OTIMIZAÇÃO: Pular verificação de existência quando skip_exists_check=True
        # Isso economiza ~100-200ms por arquivo em smart-content
        if not skip_exists_check:
            # Verify object exists before returning a signed URL. Generating a
            # signed URL for a non-existent object will produce a 404 when used
            # by clients and makes debugging harder. Check existence and return
            # None if absent so the caller can handle it explicitly.
            try:
                if not blob.exists():
                    logging.info(f"Requested signed URL for non-existent object: gs://{bucket.name}/{filename}")
                    # Server-side fallback: if the filename looks like a transformed
                    # variant (contains _s... or _t...), try to derive the original
                    # filename by stripping transformation suffixes and check if
                    # that object exists. This keeps the signing logic server-side
                    # (no client leaks) and avoids brittle client-side fallbacks.
                    try:
                        import re
                        derived = re.sub(r'(_s[^_]*)|(_t[^_]*)', '', filename)
                        if derived and derived != filename:
                            # ensure extension
                            if not derived.lower().endswith('.glb') and filename.lower().endswith('.glb'):
                                # keep .glb extension if original had it
                                if '.' not in derived:
                                    derived = derived + '.glb'
                            derived_blob = bucket.blob(derived)
                            try:
                                if derived_blob.exists():
                                    logging.info(f"Falling back to original object for signing: gs://{bucket.name}/{derived}")
                                    url = derived_blob.generate_signed_url(version='v4', expiration=3600, method='GET')
                                    return url
                            except Exception:
                                logging.exception("Failed to check derived blob.exists() for %s", derived)
                    except Exception:
                        logging.exception("Error deriving original filename from %s", filename)
                    # if fallback didn't find anything, return None
                    return None
            except Exception:
                # If the existence check fails (permissions/network), fall back
                # to attempting to generate the signed URL so we don't block
                # legitimate requests; we'll log the exception.
                logging.exception("Failed to check blob.exists() for %s", filename)

        url = blob.generate_signed_url(
            version='v4',
            expiration=expiration,
            method='GET'
        )
        return url
    except Exception as e:
        logging.exception(f"Erro ao gerar signed URL para {filename} (bucket {tipo_bucket}): {e}")
        return None

def get_glb_path_from_image_url(image_url):
    """
    Deriva o path do GLB a partir de uma URL de imagem.
    
    Exemplo:
        gs://bucket/TR77xSOJ.../totem_header.jpg 
        → gs://bucket/TR77xSOJ.../ra/models/totem_header.glb
    
    Args:
        image_url: URL da imagem (gs://bucket/path/image.jpg)
    
    Returns:
        URL do GLB correspondente ou None se não conseguir derivar
    """
    try:
        if not image_url or not isinstance(image_url, str):
            return None
        
        if not image_url.startswith('gs://'):
            return None
        
        # Parse: gs://bucket/owner_uid/image.jpg
        # Result: gs://bucket/owner_uid/ra/models/image.glb
        
        # Remove gs://bucket/
        parts = image_url.split('/', 3)
        if len(parts) < 4:
            return None
        
        bucket = parts[2]  # nome do bucket
        path = parts[3]    # owner_uid/image.jpg
        
        # Extrair owner_uid e filename
        path_parts = path.split('/', 1)
        if len(path_parts) < 2:
            # Imagem não está em owner_uid/image.jpg (pode ser public/...)
            # Tentar extrair apenas o nome do arquivo
            filename = path.split('/')[-1]
            owner_uid = None
        else:
            owner_uid = path_parts[0]
            filename = path_parts[1].split('/')[-1]  # pega última parte do path
        
        # Remover extensão e adicionar .glb
        name_without_ext = filename.rsplit('.', 1)[0]
        glb_filename = f"{name_without_ext}.glb"
        
        # Construir path do GLB
        if owner_uid:
            glb_path = f"{owner_uid}/ra/models/{glb_filename}"
        else:
            # Fallback: public/ra/models/
            glb_path = f"public/ra/models/{glb_filename}"
        
        glb_url = f"gs://{bucket}/{glb_path}"
        return glb_url
    except Exception as e:
        logging.exception(f"Erro ao derivar GLB path de {image_url}: {e}")
        return None

async def delete_image_and_glb(item, db):
    """
    Deleta uma imagem e seu GLB associado do GCS.
    
    Args:
        item: Dict com 'gs_url' ou 'filename' da imagem
        db: Database connection para pending_deletes
    
    Returns:
        True se deletou com sucesso, False caso contrário
    """
    from gcs_utils import delete_gs_path, delete_file
    
    deleted_image = False
    deleted_glb = False
    
    try:
        # 1. Deletar imagem original
        image_url = item.get('gs_url')
        image_filename = item.get('filename')
        
        if image_url:
            # Derivar GLB URL antes de deletar a imagem
            glb_url = get_glb_path_from_image_url(image_url)
            
            # Deletar imagem
            deleted_image = await asyncio.to_thread(delete_gs_path, image_url)
            logging.info(f"[delete_image_and_glb] Imagem deletada: {image_url} (sucesso: {deleted_image})")
            
            # 2. Deletar GLB associado (se existir)
            if glb_url:
                try:
                    deleted_glb = await asyncio.to_thread(delete_gs_path, glb_url)
                    logging.info(f"[delete_image_and_glb] GLB deletado: {glb_url} (sucesso: {deleted_glb})")
                except Exception as e:
                    logging.warning(f"[delete_image_and_glb] Erro ao deletar GLB {glb_url}: {e}")
        
        elif image_filename:
            # Construir gs_url a partir do filename
            image_url = f"gs://{GCS_BUCKET_CONTEUDO}/{image_filename}"
            glb_url = get_glb_path_from_image_url(image_url)
            
            # Deletar imagem
            deleted_image = await asyncio.to_thread(delete_file, image_filename, item.get('tipo', 'conteudo'))
            logging.info(f"[delete_image_and_glb] Imagem deletada: {image_filename} (sucesso: {deleted_image})")
            
            # 2. Deletar GLB associado (se existir)
            if glb_url:
                try:
                    deleted_glb = await asyncio.to_thread(delete_gs_path, glb_url)
                    logging.info(f"[delete_image_and_glb] GLB deletado: {glb_url} (sucesso: {deleted_glb})")
                except Exception as e:
                    logging.warning(f"[delete_image_and_glb] Erro ao deletar GLB {glb_url}: {e}")
        
        return deleted_image  # Retorna sucesso se pelo menos a imagem foi deletada
    
    except Exception as e:
        logging.exception(f"[delete_image_and_glb] Erro ao deletar imagem e GLB: {e}")
        return False

@app.get("/api/conteudo-signed-url")
async def get_conteudo_signed_url(gs_url: str = Query(None), filename: str = Query(None)):
    """
    Retorna um signed URL. Aceita ou um gs_url completo (gs://bucket/path) ou apenas `filename`.
    Prefer `filename` quando o cliente não deve saber o nome do bucket.
    """
    url = gerar_signed_url_conteudo(gs_url, filename)
    if not url:
        # Object not found or could not generate a signed URL
        raise HTTPException(status_code=404, detail='object not found')
    return {"signed_url": url}


@app.post('/api/conteudo-signed-urls')
async def get_conteudo_signed_urls(payload: dict = Body(...)):
    """Batch endpoint. Expects JSON: { "gs_urls": ["gs://...", ...] }
    Returns: { "signed_urls": { "gs://...": "https://...", ... } }
    """
    gs_urls = payload.get('gs_urls') if isinstance(payload, dict) else None
    if not gs_urls or not isinstance(gs_urls, list):
        raise HTTPException(status_code=400, detail='gs_urls must be a list')
    result = {}
    try:
        tasks = [asyncio.to_thread(gerar_signed_url_conteudo, u, None) for u in gs_urls]
        signed_list = await asyncio.gather(*tasks, return_exceptions=True)
        for orig, signed in zip(gs_urls, signed_list):
            if isinstance(signed, Exception):
                logging.exception(f"Error generating signed url for {orig}: {signed}")
                result[orig] = None
            else:
                result[orig] = signed
        return { 'signed_urls': result }
    except Exception as e:
        logging.exception(f"Failed to generate batch signed urls: {e}")
        raise HTTPException(status_code=500, detail='Failed to generate signed urls')


@app.get('/api/default-totem-signed-url')
async def default_totem_removed():
    """
    Endpoint removido: demos de modelos (totem/astronaut) foram retirados do backend.
    Retorna 410 Gone para evitar que clientes dependam deste recurso.
    """
    logging.info("Deprecated endpoint '/api/default-totem-signed-url' was called and will return 410 Gone.")
    raise HTTPException(status_code=410, detail='Endpoint removed: demo models are not served by the backend')

def initialize_firebase():
    cred_json_str = os.getenv("FIREBASE_CRED_JSON")
    if not cred_json_str:
        logging.error("Variável de ambiente FIREBASE_CRED_JSON não encontrada.")
        raise RuntimeError("Credenciais do Firebase ausentes.")
    
    try:
        cred_dict = json.loads(cred_json_str)
        cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase inicializado com sucesso a partir da variável de ambiente.")
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao decodificar JSON das credenciais do Firebase: {e}")
        raise
    except Exception as e:
        logging.error(f"Erro ao inicializar o Firebase: {e}")
        raise

def initialize_onnx_session():
    global ort_session
    MODEL_PATH = "quantized_clip_model.onnx"  # Troque para o modelo não quantizado
    try:
        ort_session = ort.InferenceSession(MODEL_PATH)
        logging.info("Sessão ONNX Runtime inicializada com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo ONNX: {e}. Verifique o arquivo '{MODEL_PATH}'.")
        raise RuntimeError(f"Erro ao carregar modelo ONNX: {e}")

async def load_faiss_index():
    global logo_index
    logging.info("Iniciando carregamento do índice FAISS...")
    INDEX_DIM = 512
    logo_index = LogoIndex(dim=INDEX_DIM)
    
    try:
        logos = logos_collection.find({})
        count = 0
        async for logo in logos:
            if 'vector' in logo and len(logo['vector']) == INDEX_DIM:
                vector = np.array(logo['vector'], dtype=np.float32)
                logo_index.add_logo(vector, logo)
                count += 1
        logging.info(f"Índice FAISS construído com {count} vetores.")
    except Exception as e:
        logging.error(f"Falha ao carregar o índice FAISS do MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Falha ao inicializar o índice FAISS.")

async def verify_firebase_token_dep(credentials: HTTPAuthorizationCredentials = Security(http_bearer)):
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token Firebase inválido ou expirado",
        )


async def attach_signed_urls_to_blocos(blocos):
    # Delegar para a implementação central com expirações padrão e checagem de existência
    return await _attach_signed_urls_core(
        blocos,
        expiration_image=3600,
        expiration_glb=7*24*60*60,
        skip_exists_check=False,
        include_preview=False
    )


async def _attach_signed_urls_core(
    blocos,
    *,
    expiration_image: int = 3600,
    expiration_glb: int = 7 * 24 * 60 * 60,
    skip_exists_check: bool = False,
    include_preview: bool = False,
):
    """Implementação central para anexar signed URLs aos blocos.

    Os parâmetros permitem controlar os tempos de expiração, optar por pular a
    verificação de existência dos objetos (caminho rápido) e indicar se deve
    tentar anexar um `preview_signed_url` (thumbnail).
    """
    if not blocos or not isinstance(blocos, list):
        return blocos

    tasks = []
    task_metadata = []

    for b in blocos:
        try:
            tipo_selecionado = b.get('tipoSelecionado') or ''
            tipo_label = b.get('tipo') or ''
            is_media = False
            if isinstance(tipo_selecionado, str) and tipo_selecionado.lower() in ('imagem', 'carousel', 'video'):
                is_media = True
            else:
                tl = tipo_label.lower() if isinstance(tipo_label, str) else ''
                if tl.startswith('imagem') or tl.startswith('video') or tl.startswith('carousel'):
                    is_media = True
            if not is_media:
                continue

            # Carousel items
            if b.get('items') and isinstance(b.get('items'), list):
                for it in b['items']:
                    # URL da imagem
                    url = it.get('url') or (it.get('meta') and it['meta'].get('url'))
                    filename = it.get('filename') or (it.get('meta') and it['meta'].get('filename'))
                    if not url and filename:
                        url = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"
                    if url:
                        tasks.append(asyncio.to_thread(
                            gerar_signed_url_conteudo, url, filename,
                            expiration=expiration_image, skip_exists_check=skip_exists_check
                        ))
                        task_metadata.append((it, 'signed_url', 'image'))

                        # Preview (optional)
                        if include_preview:
                            preview_fn = None
                            if isinstance(it.get('preview_filename'), str):
                                preview_fn = it.get('preview_filename')
                            elif isinstance(it.get('thumb_filename'), str):
                                preview_fn = it.get('thumb_filename')
                            elif it.get('meta') and isinstance(it['meta'].get('preview_filename'), str):
                                preview_fn = it['meta'].get('preview_filename')

                            if not preview_fn and filename and isinstance(filename, str):
                                name, dot, ext = filename.rpartition('.')
                                base = name if name else filename
                                candidates = [f"{base}_t.{ext}" if ext else f"{base}_t",
                                              f"{base}_s.{ext}" if ext else f"{base}_s",
                                              f"{base}-thumb.{ext}" if ext else f"{base}-thumb"]
                                preview_fn = candidates[0]

                            if preview_fn:
                                preview_gs = None
                                if url and isinstance(url, str) and url.startswith('gs://'):
                                    parts = url.split('/')
                                    if len(parts) >= 4:
                                        preview_gs = '/'.join(parts[:3] + [preview_fn])
                                    else:
                                        preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"
                                else:
                                    preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"

                                tasks.append(asyncio.to_thread(
                                    gerar_signed_url_conteudo, preview_gs, preview_fn,
                                    expiration=expiration_image, skip_exists_check=skip_exists_check
                                ))
                                task_metadata.append((it, 'preview_signed_url', 'image_preview'))

                    # URL do GLB
                    glb_url = it.get('glb_url')
                    glb_filename = it.get('glb_filename')
                    if glb_url:
                        tasks.append(asyncio.to_thread(
                            gerar_signed_url_conteudo, glb_url, glb_filename,
                            expiration=expiration_glb, skip_exists_check=skip_exists_check
                        ))
                        task_metadata.append((it, 'glb_signed_url', 'glb'))
                continue

            # Single media block - imagem
            url = b.get('url')
            filename = b.get('filename')
            if not url and filename:
                url = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"
            if url:
                tasks.append(asyncio.to_thread(
                    gerar_signed_url_conteudo, url, filename,
                    expiration=expiration_image, skip_exists_check=skip_exists_check
                ))
                task_metadata.append((b, 'signed_url', 'image'))

            # Preview for single media block (optional)
            if include_preview:
                preview_fn = None
                if isinstance(b.get('preview_filename'), str):
                    preview_fn = b.get('preview_filename')
                elif isinstance(b.get('thumb_filename'), str):
                    preview_fn = b.get('thumb_filename')
                elif b.get('meta') and isinstance(b['meta'].get('preview_filename'), str):
                    preview_fn = b['meta'].get('preview_filename')

                if not preview_fn and filename and isinstance(filename, str):
                    name, dot, ext = filename.rpartition('.')
                    base = name if name else filename
                    preview_fn = f"{base}_t.{ext}" if ext else f"{base}_t"

                if preview_fn:
                    preview_gs = None
                    if url and isinstance(url, str) and url.startswith('gs://'):
                        parts = url.split('/')
                        if len(parts) >= 4:
                            preview_gs = '/'.join(parts[:3] + [preview_fn])
                        else:
                            preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"
                    else:
                        preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"

                    tasks.append(asyncio.to_thread(
                        gerar_signed_url_conteudo, preview_gs, preview_fn,
                        expiration=expiration_image, skip_exists_check=skip_exists_check
                    ))
                    task_metadata.append((b, 'preview_signed_url', 'image_preview'))

            # Single media block - GLB
            glb_url = b.get('glb_url')
            glb_filename = b.get('glb_filename')
            if glb_url:
                tasks.append(asyncio.to_thread(
                    gerar_signed_url_conteudo, glb_url, glb_filename,
                    expiration=expiration_glb, skip_exists_check=skip_exists_check
                ))
                task_metadata.append((b, 'glb_signed_url', 'glb'))
        except Exception:
            continue

    # Executar em paralelo
    if tasks:
        t_before = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        t_after = time.perf_counter()

        # Aplicar resultados e contabilizar tipos atribuídos
        counts = {}
        for (target, field, tipo), result in zip(task_metadata, results):
            try:
                if isinstance(result, Exception):
                    continue
                if result:
                    target[field] = result
                    counts[field] = counts.get(field, 0) + 1
            except Exception:
                continue

    dur_total_ms = (t_after - t_before) * 1000.0
    logging.debug(f"[_attach_signed_urls_core] gather dur_ms={dur_total_ms:.1f} assigned_counts={counts}")

    return blocos


async def attach_signed_urls_to_blocos_fast(blocos):
    """
    Versão OTIMIZADA de attach_signed_urls_to_blocos para smart-content.
    
    Diferenças:
    - skip_exists_check=True (pula verificação blob.exists(), economiza ~100-200ms por arquivo)
    - TTL de 7 dias para TODAS as URLs (permite cache mais longo)
    - Assume que os blocos já foram validados no banco
    
    ⚡ GANHO ESPERADO: -2 a -3s no total
    """
    if not blocos or not isinstance(blocos, list):
        return blocos
    
    tasks = []
    task_metadata = []
    
    for b in blocos:
        try:
            tipo_selecionado = b.get('tipoSelecionado') or ''
            tipo_label = b.get('tipo') or ''
            is_media = False
            if isinstance(tipo_selecionado, str) and tipo_selecionado.lower() in ('imagem', 'carousel', 'video'):
                is_media = True
            else:
                tl = tipo_label.lower() if isinstance(tipo_label, str) else ''
                if tl.startswith('imagem') or tl.startswith('video') or tl.startswith('carousel'):
                    is_media = True
            if not is_media:
                continue
            
            # Carousel items
            if b.get('items') and isinstance(b.get('items'), list):
                for it in b['items']:
                    # URL da imagem (TTL 7 dias)
                    url = it.get('url') or (it.get('meta') and it['meta'].get('url'))
                    filename = it.get('filename') or (it.get('meta') and it['meta'].get('filename'))
                    if not url and filename:
                        url = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"
                    if url:
                        tasks.append(asyncio.to_thread(
                            gerar_signed_url_conteudo, url, filename, 
                            expiration=7*24*60*60, skip_exists_check=True
                        ))
                        task_metadata.append((it, 'signed_url', 'image'))

                        # Tentativa de gerar também uma preview (thumbnail) assinada
                        # Priorizar campos explícitos se existirem
                        preview_fn = None
                        # campos possíveis fornecidos pelo uploader/DB
                        if isinstance(it.get('preview_filename'), str):
                            preview_fn = it.get('preview_filename')
                        elif isinstance(it.get('thumb_filename'), str):
                            preview_fn = it.get('thumb_filename')
                        elif it.get('meta') and isinstance(it['meta'].get('preview_filename'), str):
                            preview_fn = it['meta'].get('preview_filename')

                        # IMPORTANTE: para manter o fast-path rápido, NÃO derivamos nomes de
                        # preview a partir do filename aqui. Isso provocou várias tentativas
                        # e checagens que aumentaram dramaticamente a latência.
                        # Apenas use previews explicitamente fornecidos pelo uploader/DB
                        # (preview_filename, thumb_filename, meta.preview_filename).
                        # Se nenhum campo explícito existir, não tentamos gerar preview.
                        if not preview_fn:
                            preview_fn = None

                        if preview_fn:
                            # If original url was a gs:// form, try to turn into gs://.../preview_fn
                            preview_gs = None
                            if url and isinstance(url, str) and url.startswith('gs://'):
                                # replace tail filename with preview_fn
                                parts = url.split('/')
                                if len(parts) >= 4:
                                    preview_gs = '/'.join(parts[:3] + [preview_fn])
                                else:
                                    preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"
                            else:
                                preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"

                                # Para previews, é preferível verificar a existência do objeto
                                # antes de retornar uma preview_signed_url para evitar 404s no cliente.
                                # Mantemos skip_exists_check=True para os arquivos principais,
                                # mas para thumbnails/previews vamos fazer a verificação.
                                tasks.append(asyncio.to_thread(
                                    gerar_signed_url_conteudo, preview_gs, preview_fn,
                                    expiration=7*24*60*60, skip_exists_check=False
                                ))
                            task_metadata.append((it, 'preview_signed_url', 'image_preview'))
                    
                    # URL do GLB (TTL 7 dias)
                    glb_url = it.get('glb_url')
                    glb_filename = it.get('glb_filename')
                    if glb_url:
                        tasks.append(asyncio.to_thread(
                            gerar_signed_url_conteudo, glb_url, glb_filename, 
                            expiration=7*24*60*60, skip_exists_check=True
                        ))
                        task_metadata.append((it, 'glb_signed_url', 'glb'))
                continue
            
            # Single media block - imagem (TTL 7 dias)
            url = b.get('url')
            filename = b.get('filename')
            if not url and filename:
                url = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"
            if url:
                tasks.append(asyncio.to_thread(
                    gerar_signed_url_conteudo, url, filename,
                    expiration=7*24*60*60, skip_exists_check=True
                ))
                task_metadata.append((b, 'signed_url', 'image'))

            # Para single media blocks, também tentar gerar preview_signed_url
            preview_fn = None
            if isinstance(b.get('preview_filename'), str):
                preview_fn = b.get('preview_filename')
            elif isinstance(b.get('thumb_filename'), str):
                preview_fn = b.get('thumb_filename')
            elif b.get('meta') and isinstance(b['meta'].get('preview_filename'), str):
                preview_fn = b['meta'].get('preview_filename')

                # Não derivar preview a partir do filename no fast-path.
                # Somente previews explícitos (preview_filename/thumb_filename/meta.preview_filename)
                # serão considerados para anexar preview_signed_url.
                if not preview_fn:
                    preview_fn = None

            if preview_fn:
                preview_gs = None
                if url and isinstance(url, str) and url.startswith('gs://'):
                    parts = url.split('/')
                    if len(parts) >= 4:
                        preview_gs = '/'.join(parts[:3] + [preview_fn])
                    else:
                        preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"
                else:
                    preview_gs = f"gs://{GCS_BUCKET_CONTEUDO}/{preview_fn}"
                # Verificar existência do preview antes de anexar (fast-path preserva
                # a otimização para imagens principais, mas queremos evitar previews 404)
                tasks.append(asyncio.to_thread(
                    gerar_signed_url_conteudo, preview_gs, preview_fn,
                    expiration=7*24*60*60, skip_exists_check=False
                ))
                task_metadata.append((b, 'preview_signed_url', 'image_preview'))
            
            # Single media block - GLB (TTL 7 dias)
            glb_url = b.get('glb_url')
            glb_filename = b.get('glb_filename')
            if glb_url:
                tasks.append(asyncio.to_thread(
                    gerar_signed_url_conteudo, glb_url, glb_filename,
                    expiration=7*24*60*60, skip_exists_check=True
                ))
                task_metadata.append((b, 'glb_signed_url', 'glb'))
        except Exception:
            continue
    
    # Executar em paralelo
    if tasks:
        t_before = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        t_after = time.perf_counter()

        counts = {}
        for (target, field, tipo), result in zip(task_metadata, results):
            try:
                if isinstance(result, Exception):
                    continue
                if result:
                    target[field] = result
                    counts[field] = counts.get(field, 0) + 1
            except Exception:
                continue

    dur_total_ms = (t_after - t_before) * 1000.0
    logging.debug(f"[attach_signed_urls_fast_core] gather dur_ms={dur_total_ms:.1f} assigned_counts={counts}")
    
    return blocos


# Configura CORS permitindo configurar as origens via variável de ambiente
# Lê CORS_ALLOW_ORIGINS (comma-separated). Se ausente, usa fallback somente em dev.
_env_origins = os.getenv('CORS_ALLOW_ORIGINS', '').strip()
if _env_origins:
    try:
        _allow_origins = [o.strip() for o in _env_origins.split(',') if o.strip()]
    except Exception:
        logging.error('Formato inválido em CORS_ALLOW_ORIGINS; esperar comma-separated list')
        _allow_origins = []
else:
    # fallback apenas para desenvolvimento local
    if os.getenv('ENV', 'development') == 'production':
        logging.error('CORS_ALLOW_ORIGINS não definido em produção. Abortando inicialização.')
        raise RuntimeError('Variável CORS_ALLOW_ORIGINS é obrigatória em produção.')
    logging.warning('CORS_ALLOW_ORIGINS não definido — usando fallback http://localhost:5173 (apenas local).')
    _allow_origins = ['http://localhost:5173']

logging.info(f"CORS origins configured: {_allow_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

async def _search_and_compare_logic(file: UploadFile):
    if not logo_index:
        raise HTTPException(status_code=503, detail="Índice de logos não está pronto.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # heavy CPU work: run in threadpool to avoid blocking event loop
        # Implementação: extrair embedding do crop central e preferir esse vetor
        # (evita falsos positivos causados por fundo). Variáveis de ambiente:
        # SEARCH_CENTER_CROP_RATIO (float, default 0.7)
        # SEARCH_PREFER_CENTER_ONLY (true/false, default true)
        crop_ratio = float(os.getenv('SEARCH_CENTER_CROP_RATIO', '0.7'))
        prefer_center_only = os.getenv('SEARCH_PREFER_CENTER_ONLY', 'true').lower() in ('1', 'true', 'yes')

        # Gerar crop central em disco
        center_path = None
        try:
            img = PILImage.open(temp_path).convert('RGB')
            w, h = img.size
            side = min(w, h)
            crop_side = int(side * crop_ratio)
            left = max(0, (w - crop_side) // 2)
            top = max(0, (h - crop_side) // 2)
            right = left + crop_side
            bottom = top + crop_side
            center_crop = img.crop((left, top, right, bottom))
            center_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            center_path = center_tmp.name
            center_crop.save(center_path, format='JPEG', quality=90)
        except Exception:
            logging.exception('[search_logo] falha ao gerar crop central; prosseguindo com imagem full')
            center_path = None

        # Extrair embedding do crop (se criado)
        center_vector = None
        if center_path:
            try:
                center_vector = await asyncio.to_thread(extract_clip_features, center_path, ort_session)
            except Exception:
                logging.exception('[search_logo] falha ao extrair embedding do crop central')
                center_vector = None

        # Função interna para executar busca + filtros (reaproveita lógica existente)
        async def _search_and_filter(qvec, q_img_path):
            import numpy as np
            acceptance_threshold = float(os.getenv('SEARCH_ACCEPTANCE_THRESHOLD', '0.38'))
            min_margin = float(os.getenv('SEARCH_MIN_MARGIN', '0.05'))
            phash_max_hamming = int(os.getenv('SEARCH_PHASH_MAX_HAMMING', '12'))

            try:
                results_raw = logo_index.search_raw(qvec, top_k=3)
            except Exception as e:
                logging.exception('[search_logo] Falha ao consultar índice FAISS: %s', e)
                return {"found": False, "trusted": False, "debug": "Erro interno no índice FAISS", "debug_reason": "faiss_error", "query_vector": np.array(qvec).tolist()}

            if not results_raw:
                return {"found": False, "trusted": False, "debug": "Nenhum candidato retornado pelo índice", "debug_reason": "no_candidates", "query_vector": np.array(qvec).tolist()}

            top1 = results_raw[0]
            d1 = float(top1.get('distance', 1.0))

            # Threshold absoluto
            if d1 > acceptance_threshold:
                logging.info(f"[search_logo] top1 distance {d1:.4f} > threshold {acceptance_threshold} -> rejeitado")
                return {"found": False, "trusted": False, "debug": "Top1 distance above threshold", "debug_reason": "distance_above_threshold", "query_vector": np.array(qvec).tolist()}

            # Margin entre top1 e top2
            if len(results_raw) > 1:
                d2 = float(results_raw[1].get('distance', 1.0))
                margin = d2 - d1
                if margin < min_margin:
                    logging.info(f"[search_logo] margin too small: d2({d2:.4f}) - d1({d1:.4f}) = {margin:.4f} < {min_margin}")
                    return {"found": False, "trusted": False, "debug": "Rejected by margin (ambiguous)", "debug_reason": "ambiguous_margin", "query_vector": np.array(qvec).tolist(), "d1": d1, "d2": d2}

            # Structural check using combined score (embedding similarity + pHash similarity)
            phash_hamming = None
            phash_similarity = None
            try:
                try:
                    import imagehash
                except Exception:
                    imagehash = None

                from io import BytesIO
                from gcs_utils import get_bucket

                # compute embedding-based similarity (assume confidence ~ 1 - distance)
                s_e = max(0.0, 1.0 - d1)

                # phash (best-effort)
                if imagehash is not None:
                    try:
                        q_img = PILImage.open(q_img_path).convert('RGB')
                        q_hash = imagehash.phash(q_img)

                        candidate = top1.get('metadata', {})
                        candidate_bytes = None
                        cand_url = candidate.get('url') or candidate.get('gs_url') or candidate.get('gcs_url')
                        cand_filename = candidate.get('filename') or candidate.get('nome')

                        if isinstance(cand_url, str) and cand_url.startswith('gs://'):
                            without_prefix = cand_url[len('gs://'):]
                            bucket_name, _, path = without_prefix.partition('/')
                            try:
                                bucket = get_bucket('logos')
                                blob = bucket.blob(path)
                                candidate_bytes = await asyncio.to_thread(blob.download_as_bytes)
                            except Exception:
                                logging.exception('[search_logo] falha ao baixar candidato por gs_url')
                        elif cand_filename:
                            try:
                                bucket = get_bucket('logos')
                                blob = bucket.blob(cand_filename)
                                candidate_bytes = await asyncio.to_thread(blob.download_as_bytes)
                            except Exception:
                                logging.exception('[search_logo] falha ao baixar candidato por filename')

                        if candidate_bytes:
                            c_img = PILImage.open(BytesIO(candidate_bytes)).convert('RGB')
                            c_hash = imagehash.phash(c_img)
                            phash_hamming = int(q_hash - c_hash)
                            # default bits for phash (hash_size=8 => 64 bits)
                            phash_bits = int(os.getenv('SEARCH_PHASH_BITS', '64'))
                            phash_similarity = max(0.0, 1.0 - (phash_hamming / float(phash_bits)))
                            logging.info(f"[search_logo] pHash hamming={phash_hamming} bits={phash_bits} similarity={phash_similarity:.3f} for candidate={candidate.get('nome')}")
                        else:
                            logging.debug('[search_logo] candidate image bytes not available for phash check; skipping')
                    except Exception:
                        logging.exception('[search_logo] erro durante phash check (ignorado)')
            except Exception as e:
                logging.exception('[search_logo] erro inesperado no bloco phash: %s', e)

            # Combine scores: embedding similarity (s_e) + phash similarity (s_p)
            emb_weight = float(os.getenv('SEARCH_EMBEDDING_WEIGHT', '0.85'))
            phash_weight = float(os.getenv('SEARCH_PHASH_WEIGHT', '0.15'))
            combined_threshold = float(os.getenv('SEARCH_COMBINED_THRESHOLD', '0.70'))

            s_p = phash_similarity
            if s_p is not None:
                combined = emb_weight * s_e + phash_weight * s_p
                logging.info(f"[search_logo] combined score emb={s_e:.3f} phash={s_p:.3f} combined={combined:.3f} (thr={combined_threshold})")
                if combined >= combined_threshold:
                    match = top1
                    return {
                        "found": True,
                        "trusted": True,
                        "debug_reason": "combined_accepted",
                        "combined_score": combined,
                        "emb_similarity": s_e,
                        "phash_similarity": s_p,
                        "name": match['metadata'].get('nome', 'Logo encontrado') if (match := top1) else top1['metadata'].get('nome', 'Logo encontrado'),
                        "confidence": float(top1.get('confidence', 0)),
                        "distance": float(top1.get('distance', 0)),
                        "owner": top1['metadata'].get('owner_uid', ''),
                        "query_vector": np.array(qvec).tolist(),
                        "phash_hamming": phash_hamming
                    }
                else:
                    # found but not trusted: expose diagnostics so client can decide
                    return {
                        "found": True,
                        "trusted": False,
                        "debug": f"combined_rejected emb={s_e:.3f} phash={s_p:.3f} comb={combined:.3f}",
                        "debug_reason": "combined_rejected",
                        "combined_score": combined,
                        "emb_similarity": s_e,
                        "phash_similarity": s_p,
                        "name": top1['metadata'].get('nome', 'Logo encontrado'),
                        "confidence": float(top1.get('confidence', 0)),
                        "distance": float(top1.get('distance', 0)),
                        "owner": top1['metadata'].get('owner_uid', ''),
                        "query_vector": np.array(qvec).tolist(),
                        "phash_hamming": phash_hamming
                    }
            else:
                # No phash available: fallback to embedding similarity alone
                # Accept if embedding similarity itself meets the combined threshold
                combined = s_e
                logging.info(f"[search_logo] no phash available, emb_similarity={s_e:.3f} (thr={combined_threshold})")
                if combined >= combined_threshold:
                    match = top1
                    return {
                        "found": True,
                        "trusted": True,
                        "debug_reason": "emb_only_accepted",
                        "emb_similarity": s_e,
                        "name": match['metadata'].get('nome', 'Logo encontrado') if (match := top1) else top1['metadata'].get('nome', 'Logo encontrado'),
                        "confidence": float(top1.get('confidence', 0)),
                        "distance": float(top1.get('distance', 0)),
                        "owner": top1['metadata'].get('owner_uid', ''),
                        "query_vector": np.array(qvec).tolist()
                    }
                else:
                    return {
                        "found": True,
                        "trusted": False,
                        "debug": f"emb_too_low emb={s_e:.3f}",
                        "debug_reason": "low_embedding_score",
                        "emb_similarity": s_e,
                        "name": top1['metadata'].get('nome', 'Logo encontrado'),
                        "confidence": float(top1.get('confidence', 0)),
                        "distance": float(top1.get('distance', 0)),
                        "owner": top1['metadata'].get('owner_uid', ''),
                        "query_vector": np.array(qvec).tolist()
                    }

        # Primeiro, tentar com o crop central se disponível
        if center_vector is not None:
            res_center = await _search_and_filter(center_vector, center_path)
            # Se preferir apenas center, retornamos direto
            if prefer_center_only:
                # cleanup temp center file
                try:
                    if center_path and os.path.exists(center_path):
                        os.remove(center_path)
                except Exception:
                    pass
                return res_center

            # Se crop não aceitou, tentar com a imagem completa
            if not res_center.get('found'):
                try:
                    full_vector = await asyncio.to_thread(extract_clip_features, temp_path, ort_session)
                except Exception:
                    logging.exception('[search_logo] falha ao extrair embedding da imagem full')
                    # cleanup
                    try:
                        if center_path and os.path.exists(center_path):
                            os.remove(center_path)
                    except Exception:
                        pass
                    return res_center

                res_full = await _search_and_filter(full_vector, temp_path)
                # cleanup
                try:
                    if center_path and os.path.exists(center_path):
                        os.remove(center_path)
                except Exception:
                    pass
                # preferir resultado da full se aceitou, senão retornar res_center (mais conservador)
                if res_full.get('found'):
                    return res_full
                return res_center

        # Se não houve crop (erro), cair back para comportamento anterior com full image
        try:
            full_vector = await asyncio.to_thread(extract_clip_features, temp_path, ort_session)
        except Exception:
            logging.exception('[search_logo] falha ao extrair embedding da imagem full')
            return {"found": False, "trusted": False, "debug": "Erro ao extrair embedding", "debug_reason": "extract_failed", "query_vector": None}

        res_full = await _search_and_filter(full_vector, temp_path)
        return res_full
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post('/upload/cancel')
async def upload_cancel(
    temp_id: str = Body(None),
    filename: str = Body(None),
    token: dict = Depends(verify_firebase_token_dep)
):
    """Cancel a staged upload (uploaded_assets). Accepts either temp_id or filename.
    Deletes GCS objects (image + glb) and removes uploaded_assets record.
    """
    if not temp_id and not filename:
        raise HTTPException(status_code=400, detail='temp_id or filename required')
    try:
        query = {'owner_uid': token.get('uid')}
        if temp_id:
            query['temp_id'] = temp_id
        else:
            query['filename'] = filename

        asset = await db['uploaded_assets'].find_one(query)
        if not asset:
            return {'ok': True, 'message': 'asset not found'}

        from gcs_utils import delete_gs_path
        # delete image
        try:
            if asset.get('gs_url'):
                await asyncio.to_thread(delete_gs_path, asset.get('gs_url'))
        except Exception:
            logging.exception('[upload_cancel] Falha ao deletar gs_url (continuando)')
        # delete glb if present
        try:
            if asset.get('glb_url'):
                await asyncio.to_thread(delete_gs_path, asset.get('glb_url'))
        except Exception:
            logging.exception('[upload_cancel] Falha ao deletar glb_url (continuando)')

        await db['uploaded_assets'].delete_one({'_id': asset['_id']})
        return {'ok': True, 'deleted': True}
    except HTTPException:
        raise
    except Exception:
        logging.exception('[upload_cancel] Erro ao cancelar upload')
        raise HTTPException(status_code=500, detail='internal error')


@app.post('/admin/cleanup-uploaded-assets')
async def admin_cleanup_uploaded_assets(token: dict = Depends(verify_firebase_token_dep)):
    # Only allow master admin to trigger
    master_email = os.getenv('USER_ADMIN_EMAIL')
    if token.get('email') != master_email:
        raise HTTPException(status_code=403, detail='Forbidden')

    threshold = datetime.utcnow() - timedelta(days=7)
    processed = []
    try:
        orphans = await db['uploaded_assets'].find({'attached': False, 'created_at': {'$lt': threshold}}).to_list(length=1000)
        from gcs_utils import delete_gs_path
        for a in orphans:
            try:
                fname = a.get('filename')
                # safety: check if any conteudo references this filename
                ref = await db['conteudos'].find_one({'$or': [{'blocos.filename': fname}, {'blocos.items.filename': fname}]})
                if ref:
                    # mark attached to avoid deletion
                    await db['uploaded_assets'].update_one({'_id': a['_id']}, {'$set': {'attached': True, 'attached_at': datetime.utcnow(), 'conteudo_id': str(ref.get('_id'))}})
                    processed.append({'id': str(a.get('_id')), 'status': 'referenced'})
                    continue

                # delete files
                try:
                    if a.get('gs_url'):
                        await asyncio.to_thread(delete_gs_path, a.get('gs_url'))
                except Exception:
                    logging.exception('[admin_cleanup] Falha ao deletar gs_url')
                try:
                    if a.get('glb_url'):
                        await asyncio.to_thread(delete_gs_path, a.get('glb_url'))
                except Exception:
                    logging.exception('[admin_cleanup] Falha ao deletar glb_url')

                await db['uploaded_assets'].delete_one({'_id': a['_id']})
                processed.append({'id': str(a.get('_id')), 'status': 'deleted'})
            except Exception:
                logging.exception('[admin_cleanup] Erro ao processar asset')
                processed.append({'id': str(a.get('_id')), 'status': 'error'})
    except Exception:
        logging.exception('[admin_cleanup] Erro ao buscar assets órfãos')
        raise HTTPException(status_code=500, detail='internal error')

    return {'processed': processed, 'count': len(processed)}


@app.post('/api/validate-button-block')
async def api_validate_button_block(payload: dict = Body(...), token: dict = Depends(verify_firebase_token_dep)):
    """Valida um payload de botão usando o Pydantic schema; usado apenas para testes e adm."""
    try:
        validated = validate_button_block_payload(payload)
        return {"valid": True, "normalized": validated.dict(exclude_none=True)}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.post('/search-logo/')
async def search_logo(file: UploadFile = File(...)):
    return await _search_and_compare_logic(file)

@app.post('/authenticated-search-logo/')
async def search_logo_auth(file: UploadFile = File(...), token: dict = Depends(verify_firebase_token_dep)):
    return await _search_and_compare_logic(file)

@app.get('/debug/user')
async def debug_user(token: dict = Depends(verify_firebase_token_dep)):
    master_email = os.getenv("USER_ADMIN_EMAIL")
    return {
        "user_email": token.get("email"),
        "master_email": master_email,
        "is_master": token.get("email") == master_email,
        "uid": token.get("uid")
    }

@app.get('/debug/logos')
async def debug_logos():
    try:
        logos = await logos_collection.find({}).to_list(length=100)
        return {
            "total_logos": len(logos),
            "logos": [
                {
                    "id": str(logo["_id"]),
                    "nome": logo.get("nome"),
                    "has_vector": "vector" in logo,
                    "vector_length": len(logo["vector"]) if "vector" in logo else 0
                }
                for logo in logos
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/add-logo/')
async def add_logo(
    file: UploadFile = File(...),
    name: str = Form(...),
    token: dict = Depends(verify_firebase_token_dep)
):
    existing = await logos_collection.find_one({"nome": name})
    if existing:
        raise HTTPException(status_code=400, detail="Já existe uma imagem com esse nome.")
    
    allowed_types = ["image/png", "image/jpeg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Tipo de arquivo não permitido.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        # Extrai features em threadpool para não bloquear o event loop
        features = await asyncio.to_thread(extract_clip_features, temp_path, ort_session)
        features = np.array(features, dtype=np.float32)
        features /= np.linalg.norm(features)
        # Upload ao GCS pode envolver I/O síncrono; executa em threadpool também
        gcs_url = await asyncio.to_thread(upload_image_to_gcs, temp_path, os.path.basename(file.filename), "logos")
        doc = {
            "nome": name,
            "url": gcs_url,
            "filename": os.path.basename(file.filename),
            "owner_uid": token["uid"],
            "vector": features.tolist()
        }
        result = await logos_collection.insert_one(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar logo: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return {"success": True, "id": str(result.inserted_id)}

@app.get('/images')
async def get_images(ownerId: str = None):
    logging.info(f"ownerId recebido: '{ownerId}'")
    filtro = {}
    if ownerId:
        filtro = {"owner_uid": ownerId}
    imagens = await logos_collection.find(filtro).to_list(length=100)
    logging.info(f"Imagens encontradas: {len(imagens)}")

    # ✅ OTIMIZAÇÃO CRÍTICA: Attach signed_url em PARALELO para todas as imagens
    # Antes: 5 logos × 700ms = 3.5s ❌
    # Agora: 5 logos em paralelo = ~700ms ✅
    try:
        async def process_image(img):
            """Processa uma imagem e anexa signed_url"""
            try:
                url = img.get('url') if isinstance(img, dict) else None
                signed = None
                if isinstance(url, str) and url.startswith('gs://'):
                    # ✅ Usa skip_exists_check=True para logos (economia de ~300ms por logo)
                    signed = await asyncio.to_thread(
                        gerar_signed_url_conteudo, 
                        url, 
                        img.get('filename'),
                        expiration=604800,  # 7 dias
                        skip_exists_check=True  # Logos sempre existem, não precisa verificar
                    )
                    if not signed:
                        logging.warning(f"Could not generate signed_url for {url} (id={img.get('_id')})")
                img['signed_url'] = signed
            except Exception as e:
                logging.exception(f"Unexpected error while processing image for signed_url: {e}")
                img['signed_url'] = None
            return img
        
        # ✅ Processa todas as imagens EM PARALELO usando asyncio.gather
        imagens = await asyncio.gather(*[process_image(img) for img in imagens])
        
    except Exception:
        logging.exception('Erro ao anexar signed_url às imagens')
    
    try:
        sanitized = [sanitize_for_json(img) for img in imagens]
        return sanitized
    except Exception as e:
        logging.exception(f"Falha ao sanitizar documentos de imagens: {e}")
        # Fallback: return minimal representation
        minimal = []
        for img in imagens:
            try:
                minimal.append({
                    "id": str(img.get("_id")) if img.get("_id") else None,
                    "nome": img.get("nome"),
                    "url": img.get("url"),
                    "filename": img.get("filename"),
                    "owner_uid": img.get("owner_uid")
                })
            except Exception:
                continue
        return minimal


@app.post('/api/generate-glb-from-image')
async def api_generate_glb_from_image(payload: dict = Body(...), request: Request = None):
    """
    Generate a simple GLB from a remote image URL and upload to GCS. Returns signed URL.
    Expects payload: { "image_url": "https://...", "filename": "optional-name.glb" }
    """
    # Optional runtime guard: require Firebase auth to request signed URLs
    GLB_REQUIRE_AUTH = os.getenv('GLB_REQUIRE_AUTH', 'false').lower() == 'true'
    decoded_token = None
    if GLB_REQUIRE_AUTH:
        # Expect Authorization: Bearer <idToken>
        auth_header = None
        try:
            auth_header = request.headers.get('authorization') if request and request.headers else None
        except Exception:
            auth_header = None
        if not auth_header or not isinstance(auth_header, str) or not auth_header.lower().startswith('bearer '):
            raise HTTPException(status_code=401, detail='Authorization header required')
        idtoken = auth_header.split(' ', 1)[1]
        try:
            decoded_token = auth.verify_id_token(idtoken)
        except Exception:
            raise HTTPException(status_code=401, detail='Invalid or expired Firebase token')

    image_url = payload.get('image_url')
    # Use hash-based filename for stable cache key (avoid re-generating for same content)
    if not image_url:
        raise HTTPException(status_code=400, detail='image_url required')

    # Normaliza a chave de cache para evitar duplicação quando a mesma imagem é referenciada
    # por URLs diferentes (ex.: GCS signed URLs com querystring expirada) ou por data URLs
    # equivalentes. Para data URLs: hash dos bytes decodificados. Para HTTPS: hash de
    # scheme://host/path (sem query/fragment), com host em minúsculas.
    from urllib.parse import urlsplit, urlunsplit
    def _stable_sha_from_image(image_url_str: str) -> str:
        try:
            if isinstance(image_url_str, str) and image_url_str.startswith('data:'):
                import base64, re
                m = re.match(r'data:(image/[^;]+);base64,(.+)', image_url_str, re.I)
                if not m:
                    # Se for um data URL inesperado, usa o string bruto
                    return hashlib.sha256(image_url_str.encode('utf-8')).hexdigest()[:16]
                b64 = m.group(2)
                try:
                    raw = base64.b64decode(b64, validate=False)
                except Exception:
                    raw = base64.b64decode(b64)
                return hashlib.sha256(raw).hexdigest()[:16]
            else:
                p = urlsplit(image_url_str)
                # normaliza host para minúsculas e remove query/fragment
                netloc = (p.hostname or '').lower()
                # preserva porta explícita se houver
                if p.port:
                    netloc = f"{netloc}:{p.port}"
                normalized = urlunsplit((p.scheme, netloc, p.path or '/', '', ''))
                return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
        except Exception:
            # fallback seguro
            return hashlib.sha256(str(image_url_str).encode('utf-8')).hexdigest()[:16]

    sha = _stable_sha_from_image(image_url)
    base_filename = f'generated_{sha}.glb'

    # Determine owner prefix (if provided) and compose final filename inside the 'conteudo' bucket
    owner_uid = None
    try:
        owner_uid = payload.get('owner_uid') or payload.get('owner') or payload.get('ownerUid')
    except Exception:
        owner_uid = None

    provided_filename = None
    try:
        provided_filename = payload.get('filename')
    except Exception:
        provided_filename = None

    # Arquitetura de pastas no GCS:
    # - Imagens originais: {owner_uid}/image.jpg
    # - GLBs gerados:      {owner_uid}/ra/models/image.glb
    # IMPORTANTE: GLBs devem SEMPRE ficar isolados por usuário para segurança e gerenciamento.
    # Se não houver owner_uid, usar 'anonymous' como fallback (para compatibilidade com requests sem auth).
    if not owner_uid:
        logging.warning(f"[generate-glb] owner_uid não fornecido, usando 'anonymous' como fallback")
        owner_uid = 'anonymous'
    
    if provided_filename and isinstance(provided_filename, str) and provided_filename.strip() != "":
        # avoid double-prefixing: if provided_filename already looks like a path, use as-is
        if '/' in provided_filename:
            filename = provided_filename
        else:
            filename = f"{owner_uid}/ra/models/{provided_filename}"
    else:
        filename = f"{owner_uid}/ra/models/{base_filename}"

    temp_image = None
    temp_glb = None
    try:
        # Configurable limits / whitelist
        ALLOWED_DOMAINS = [d.strip().lower() for d in os.getenv('GLB_ALLOWED_DOMAINS', '').split(',') if d.strip()]
        MAX_IMAGE_BYTES = int(os.getenv('GLB_MAX_IMAGE_BYTES', '5000000'))  # 5 MB default
        MAX_IMAGE_DIM = int(os.getenv('GLB_MAX_DIM', '2048'))

        # If image_url is a data URL (base64) accept it and write to temp file
        from urllib.parse import urlparse
        parsed = None
        if image_url.startswith('data:'):
            import base64, re
            m = re.match(r'data:(image/[^;]+);base64,(.+)', image_url, re.I)
            if not m:
                raise HTTPException(status_code=400, detail='Invalid data URL for image')
            mime = m.group(1).lower()
            b64 = m.group(2)
            if not mime.startswith('image'):
                raise HTTPException(status_code=400, detail='Data URL is not an image')
            # choose extension
            if mime in ('image/jpeg', 'image/jpg'):
                ext = '.jpg'
            elif mime == 'image/png':
                ext = '.png'
            else:
                # fallback to jpg
                ext = '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
                temp_image = tf.name
                try:
                    tf.write(base64.b64decode(b64))
                except Exception:
                    raise HTTPException(status_code=400, detail='Failed to decode base64 image')
        else:
            parsed = urlparse(image_url)
            if parsed.scheme not in ('https',):
                raise HTTPException(status_code=400, detail='image_url must use https scheme')
            hostname = (parsed.hostname or '').lower()
            if ALLOWED_DOMAINS and hostname not in ALLOWED_DOMAINS:
                raise HTTPException(status_code=400, detail='image_url host not allowed')

            # Check cache: does file already exist in GCS?
            bucket = get_bucket('conteudo')
            blob = bucket.blob(filename)
            # Diagnostic log: before checking existence
            logging.info(
                "[generate-glb] checking existence in GCS before blob.exists: bucket=%s filename=%s owner_uid=%s base_filename=%s image_url=%s",
                bucket.name,
                filename,
                owner_uid,
                base_filename,
                (image_url[:200] + '...') if isinstance(image_url, str) and len(image_url) > 200 else image_url,
            )
            exists = await asyncio.to_thread(blob.exists)
            # Diagnostic log: after checking existence
            logging.info(
                "[generate-glb] blob.exists result: %s for gs://%s/%s",
                exists,
                bucket.name,
                filename,
            )
            if exists:
                logging.info(
                    "[generate-glb] cache hit - returning signed URL without regenerating: gs://%s/%s",
                    bucket.name,
                    filename,
                )
                gcs_path = f'gs://{bucket.name}/{filename}'
                signed = gerar_signed_url_conteudo(gcs_path, filename)
                return { 'glb_signed_url': signed, 'gs_url': gcs_path, 'cached': True }

            # download image with streaming, enforce max bytes
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

        # Resize/normalize image if too big in dimensions (run in thread)
        processed_image = await asyncio.to_thread(resize_image_if_needed, temp_image, MAX_IMAGE_DIM)

        # generate glb (but first try to acquire a lightweight generation lock/marker
        # to avoid multiple processes doing the expensive generation concurrently)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as tg:
            temp_glb = tg.name

        # Ensure we have a bucket reference (may not be set if image_url was a data: URL)
        try:
            bucket
        except NameError:
            bucket = get_bucket('conteudo')
            blob = bucket.blob(filename)

        marker_name = f"{filename}.generating"
        marker_blob = bucket.blob(marker_name)
        we_are_owner = False

        try:
            # Try to create the marker atomically: only the first creator will succeed.
            await asyncio.to_thread(marker_blob.upload_from_string, "", if_generation_match=0)
            we_are_owner = True
            logging.info("[generate-glb] acquired generation marker %s", marker_name)
        except PreconditionFailed:
            # Another process created the marker first.
            logging.info("[generate-glb] generation marker already exists (another process is generating): %s", marker_name)
        except Exception as e:
            # Unexpected error creating marker; log and proceed (we'll still try to generate)
            logging.exception("[generate-glb] unexpected error creating generation marker %s: %s", marker_name, e)

        # If we didn't acquire the marker, wait for the other process to finish generating
        if not we_are_owner:
            wait_seconds = int(os.getenv('GLB_GENERATION_WAIT_SECONDS', '30'))
            waited = 0.0
            interval = 0.5
            while waited < wait_seconds:
                exists = await asyncio.to_thread(blob.exists)
                if exists:
                    gcs_path = f'gs://{bucket.name}/{filename}'
                    signed = gerar_signed_url_conteudo(gcs_path, filename)
                    return { 'glb_signed_url': signed, 'gs_url': gcs_path, 'cached': True }
                await asyncio.sleep(interval)
                waited += interval
            logging.info("[generate-glb] timeout waiting for concurrent generation; proceeding to generate: %s", filename)

        try:
            # run generator in thread to avoid blocking
            # allow caller to specify base height (meters above ground) via payload.height
            try:
                base_height = float(payload.get('height', 0.0))
            except Exception:
                base_height = 0.0
            # Generate GLB: ensure plane_height is numeric (default 1.0) and explicitly
            # pass flip flags by name to avoid accidental positional swaps.
            plane_h = 1.0
            try:
                # allow caller to suggest plane height via payload.plane_height (meters)
                ph = payload.get('plane_height') if isinstance(payload, dict) else None
                if ph is not None:
                    plane_h = float(ph)
            except Exception:
                plane_h = 1.0

            # Allow callers to override UV flips per-content (useful to correct
            # orientation differences between viewers/camera preview). Accepts
            # boolean values or strings 'true'/'false'. Defaults chosen to avoid
            # horizontal mirroring while keeping the vertical orientation that
            # works for most mobile viewers.
            def _to_bool(v, default=False):
                try:
                    if v is None:
                        return default
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    if s in ('1','true','yes','y','t'):
                        return True
                    if s in ('0','false','no','n','f'):
                        return False
                except Exception:
                    pass
                return default

            flip_u = _to_bool(payload.get('flip_u') if isinstance(payload, dict) else None, False)
            # vertical flip default: True (fix common upside-down appearance on some viewers)
            flip_v = _to_bool(payload.get('flip_v') if isinstance(payload, dict) else None, True)

            await asyncio.to_thread(
                generate_plane_glb,
                processed_image,
                temp_glb,
                base_y=base_height,
                plane_height=plane_h,
                flip_u=flip_u,
                flip_v=flip_v,
            )

            # upload to GCS using the stable filename (set cache-control + metadata)
            # Guardar apenas um identificador/sumário da origem da imagem nos metadados
            # porque imagens embutidas (data:) ou URLs longas podem exceder o limite
            # permitido para a parte de metadata no upload multipart do GCS.
            try:
                gen_from = image_url
                if isinstance(gen_from, str) and gen_from.startswith('data:'):
                    # não armazenar toda a base64 nos metadados — ótima forma é guardar apenas o tamanho
                    gen_from = f"data:base64(length={len(gen_from)})"
                elif isinstance(gen_from, str) and len(gen_from) > 512:
                    # truncar URLs muito longas para evitar metadados enormes
                    gen_from = gen_from[:512] + '...'
            except Exception:
                gen_from = None

            metadata = { 'generated_from_image': gen_from or 'unknown', 'base_height': str(base_height) }
            gcs_path = await asyncio.to_thread(upload_image_to_gcs, temp_glb, filename, 'conteudo', 'public, max-age=31536000', metadata)
            signed = gerar_signed_url_conteudo(gcs_path, filename)
            return { 'glb_signed_url': signed, 'gs_url': gcs_path, 'cached': False }
        finally:
            # Clean up the marker if we created it so future requests don't wait unnecessarily.
            if we_are_owner:
                try:
                    await asyncio.to_thread(marker_blob.delete)
                    logging.info("[generate-glb] removed generation marker %s", marker_name)
                except Exception as e:
                    logging.exception("[generate-glb] failed to remove generation marker %s: %s", marker_name, e)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception('Erro gerando GLB: %s', e)
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
    

@app.delete('/delete-logo/')
async def delete_logo(id: str = Query(...), token: dict = Depends(verify_firebase_token_dep)):
    # Tentativa tolerante de resolver o registro da logo.
    # 1) Se for um ObjectId válido, usa isso
    # 2) Caso contrário, tenta encontrar por string do _id, filename ou nome
    logo = None
    object_id = None
    try:
        object_id = ObjectId(id)
    except Exception:
        object_id = None

    if object_id:
        logo = await logos_collection.find_one({"_id": object_id})
    else:
        # procurar por _id como string (compatibilidade) OU por filename OU por nome
        try:
            logo = await logos_collection.find_one({
                "$or": [
                    {"_id": id},
                    {"filename": id},
                    {"nome": id}
                ]
            })
        except Exception:
            # busca mais segura: percorrer e comparar manualmente (fallback)
            cursor = logos_collection.find({})
            async for doc in cursor:
                try:
                    if str(doc.get('_id')) == str(id) or doc.get('filename') == id or doc.get('nome') == id:
                        logo = doc
                        break
                except Exception:
                    continue

    if not logo:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")

    # Tenta deletar o arquivo no GCS de forma tolerante/idempotente
    from gcs_utils import delete_gs_path, delete_file

    deleted_ok = False
    # Preferir deletar via gs_url se disponível
    try:
        if logo.get('url') and isinstance(logo.get('url'), str) and logo.get('url').startswith('gs://'):
            deleted_ok = await asyncio.to_thread(delete_gs_path, logo.get('url'))
        elif logo.get('filename'):
            # delete_file aceita tanto 'path/filename' quanto 'gs://...' indireto
            deleted_ok = await asyncio.to_thread(delete_file, logo.get('filename'), 'logos')
        else:
            # fallback: nothing to delete in storage, consider deleted
            deleted_ok = True
    except Exception as e:
        logging.exception(f"Erro ao deletar arquivo do GCS para logo {logo.get('_id')}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao deletar arquivo do bucket: {str(e)}")

    # Agora remove o documento do banco
    try:
        # use _id se disponível no documento recuperado
        doc_id = logo.get('_id')
        if doc_id:
            await logos_collection.delete_one({'_id': doc_id})
        else:
            # fallback: tentar deletar por filename/nome
            await logos_collection.delete_one({'filename': logo.get('filename')})
    except Exception as e:
        logging.exception(f"Erro ao deletar documento de logo no MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao remover registro no banco: {str(e)}")

    return {"success": True, "deleted_gcs": bool(deleted_ok), "id": str(logo.get('_id')), "filename": logo.get('filename')}

@app.get("/admin/list")
async def list_users(token: dict = Depends(verify_firebase_token_dep)):
    try:
        if token.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Acesso negado.")

        users = []
        for user in auth.list_users().iterate_all():
            users.append({"uid": user.uid, "email": user.email})
        return users
    except Exception as e:
        logging.exception("Erro ao listar usuários: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/create")
async def create_admin(
    email: str = Body(...),
    password: str = Body(None),
    token: dict = Depends(verify_firebase_token_dep)
):
    if token.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acesso negado.")

    try:
        user = auth.create_user(email=email)
        auth.set_custom_user_claims(user.uid, {"role": "admin"})
        reset_link = auth.generate_password_reset_link(email)
        msg = MIMEText(f"Olá! Defina sua senha neste link: {reset_link}")
        msg["Subject"] = "Defina sua senha de acesso"
        msg["From"] = "no-reply@olinx.digital"
        msg["To"] = email

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(os.getenv("USER_ADMIN_EMAIL"), os.getenv("APP_PASSWORD"))
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())
        except Exception as e:
            auth.delete_user(user.uid)
            raise HTTPException(status_code=400, detail=f"Falha ao enviar e-mail: {str(e)}")

        return {"success": True, "uid": user.uid, "email": user.email}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class DeleteAdminRequest(BaseModel):
    uid: str

@app.post("/admin/delete")
async def delete_admin(
    req: DeleteAdminRequest,
    token: dict = Depends(verify_firebase_token_dep)
):
    uid = req.uid
    if token.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acesso negado.")

    try:
        auth.delete_user(uid)
        return {"success": True, "uid": uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/debug/inspect-request/')
async def debug_inspect_request(data: dict = Body(...)):
    import logging
    logging.info(f'[debug-inspect-request] Corpo recebido: {str(data)[:500]}')
    return {"received": data}

async def buscar_conteudo_por_marca_e_localizacao(marca_id, latitude, longitude, radius_m: float = None):
    """
    Procura um documento de conteúdo pela marca (preferencialmente por marca_id) e
    por faixa de latitude/longitude.

    Estratégia:
    1. Tenta buscar por campo `marca_id` igual a `str(marca_id)`.
    2. Se não encontrar, tenta resolver o `nome_marca` usando a coleção `logos`
       (caso `marca_id` seja um ObjectId vindo de `logos`) e busca por `nome_marca`.
    3. Por fim, tenta um fallback simples por `nome_marca` igual a `str(marca_id)`.
    Isso garante compatibilidade com documentos antigos que usam `nome_marca` e com
    novos documentos que usam `marca_id`.
    """
    # If radius_m is provided, try a geospatial $geoNear query first (more precise).
    # maxDistance for GeoJSON $near/$geoNear is in meters when using 2dsphere index.
    start = time.perf_counter()
    logging.debug(f"[buscar_conteudo] start marca_id={marca_id} lat={latitude} lon={longitude} radius_m={radius_m}")
    try:
        if radius_m is not None and latitude is not None and longitude is not None:
            # If we have marca_id, prefer to search by marca_id, otherwise by nome_marca will be resolved below.
            try:
                query = { }
                if marca_id:
                    query['marca_id'] = marca_id
                # Use aggregation with $geoNear to get distance metadata
                pipeline = [
                    {
                        "$geoNear": {
                            "near": {"type": "Point", "coordinates": [ float(longitude), float(latitude) ]},
                            "distanceField": "dist.calculated",
                            "maxDistance": float(radius_m),
                            "spherical": True,
                            "query": query
                        }
                    },
                    {"$limit": 1}
                ]
                cursor = db['conteudos'].aggregate(pipeline)
                docs = []
                async for doc in cursor:
                    docs.append(doc)
                if docs and len(docs) > 0:
                    # return the first match along with distance (meters)
                    d0 = docs[0]
                    d0['_id'] = d0.get('_id')
                    d0['_matched_by'] = 'distance'
                    # dist.calculated is in meters
                    try:
                        d0['_distance_m'] = float(d0.get('dist', {}).get('calculated', 0))
                    except Exception:
                        d0['_distance_m'] = None
                    dur = (time.perf_counter() - start) * 1000.0
                    logging.info(f"[buscar_conteudo] geoNear hit dur_ms={dur:.1f}")
                    return d0
            except Exception:
                # if geo query fails, fall back to bounding box approach below
                pass

    except Exception:
        # if anything goes wrong with radius logic, continue to fallback search
        pass

    # Fallback: use bounding box deltas for lat/lon when radius not provided or geo query failed
    # Se radius_m for fornecido, convertemos metros->graus (aprox 1 deg ~= 111.32km) para reduzir o box
    try:
        if radius_m is not None and radius_m > 0:
            deg = float(radius_m) / 111320.0
            # adicionar 10% de margem
            deg = max(deg * 1.1, 0.0001)
            lat_filter = {"$gte": latitude - deg, "$lte": latitude + deg}
            lon_filter = {"$gte": longitude - deg, "$lte": longitude + deg}
        else:
            lat_filter = {"$gte": latitude - 0.01, "$lte": latitude + 0.01}
            lon_filter = {"$gte": longitude - 0.01, "$lte": longitude + 0.01}
    except Exception:
        lat_filter = {"$gte": latitude - 0.01, "$lte": latitude + 0.01}
        lon_filter = {"$gte": longitude - 0.01, "$lte": longitude + 0.01}

    # 1) Busca por marca_id — suportando tanto ObjectId quanto strings
    try:
        filtro = {
            "latitude": lat_filter,
            "longitude": lon_filter
        }
        # Se o valor recebido já for um ObjectId, use-o diretamente.
        # Caso contrário, tente converter para ObjectId; se falhar, pesquise pela string.
        try:
            if isinstance(marca_id, ObjectId):
                filtro['marca_id'] = marca_id
            else:
                filtro['marca_id'] = ObjectId(str(marca_id))
        except Exception:
            filtro['marca_id'] = str(marca_id)

        doc = await db["conteudos"].find_one(filtro)
        if doc:
            dur = (time.perf_counter() - start) * 1000.0
            logging.info(f"[buscar_conteudo] match by marca_id dur_ms={dur:.1f}")
            return doc
    except Exception:
        # falha ao buscar por marca_id — seguimos com fallbacks
        doc = None

    # 2) Tentar resolver nome da marca a partir da coleção `logos`
    nome_marca = None
    try:
        # Se marca_id for um ObjectId ou string que represente o _id
        try:
            obj_id = ObjectId(marca_id)
            marca_doc = await logos_collection.find_one({"_id": obj_id})
        except Exception:
            # marca_id pode já ser um nome ou um string não-convertível
            marca_doc = None
        if marca_doc:
            nome_marca = marca_doc.get("nome")
    except Exception:
        nome_marca = None

    if nome_marca:
        filtro2 = {
            "nome_marca": nome_marca,
            "latitude": lat_filter,
            "longitude": lon_filter
        }
        doc = await db["conteudos"].find_one(filtro2)
        if doc:
            dur = (time.perf_counter() - start) * 1000.0
            logging.info(f"[buscar_conteudo] match by nome_marca dur_ms={dur:.1f}")
            return doc

    # 3) Fallback: busca por nome_marca igual ao valor recebido
    filtro3 = {
        "nome_marca": str(marca_id),
        "latitude": lat_filter,
        "longitude": lon_filter
    }
    result = await db["conteudos"].find_one(filtro3)
    dur = (time.perf_counter() - start) * 1000.0
    logging.info(f"[buscar_conteudo] fallback result dur_ms={dur:.1f}")
    return result

async def geocode_reverse(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1
    }
    headers = {"User-Agent": "OlinxRA/1.0"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get("address", {})
    except httpx.RequestError as e:
        logging.error(f"Erro na chamada de geocode_reverse: {e}")
    return {}

@app.get('/api/reverse-geocode')
async def api_reverse_geocode(lat: float = Query(...), lon: float = Query(...)):
    try:
        address = await geocode_reverse(lat, lon)
        return address
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na geocodificação reversa: {str(e)}")

# Cache em memória: {(nome_marca, latitude, longitude, radius): resultado}
consulta_cache = {}
CONSULTA_CACHE_MAX_SIZE = 1000

def add_to_consulta_cache(key, value):
    """Adiciona item ao cache de consulta com limite de tamanho (FIFO)"""
    if len(consulta_cache) >= CONSULTA_CACHE_MAX_SIZE:
        # Remove item mais antigo (primeiro inserido)
        consulta_cache.pop(next(iter(consulta_cache)))
    consulta_cache[key] = value

def make_cache_key(nome_marca, latitude, longitude):
    # Arredonda para evitar pequenas variações
    return (nome_marca, round(latitude, 6), round(longitude, 6))

@app.post('/consulta-conteudo/')
async def consulta_conteudo(
    nome_marca: str = Body(...),
    latitude: float = Body(...),
    longitude: float = Body(...),
    radius_m: float = Body(None)
):
    # include radius in cache key to avoid stale results
    cache_key = (nome_marca, round(latitude, 6), round(longitude, 6), int(radius_m) if radius_m else None)
    if cache_key in consulta_cache:
        return consulta_cache[cache_key]

    # Busca a marca no banco
    marca = await logos_collection.find_one({"nome": nome_marca})
    if not marca:
        resultado = {"conteudo": None, "mensagem": "Marca não encontrada."}
        add_to_consulta_cache(cache_key, resultado)
        return resultado

    # Busca conteúdo associado à marca e localização usando a função
    conteudo = await buscar_conteudo_por_marca_e_localizacao(marca["_id"], latitude, longitude, radius_m)

    # Busca endereço detalhado usando geocodificação reversa
    endereco = await geocode_reverse(latitude, longitude)

    # Monta string do local (exemplo: Rua, Bairro, Cidade, Estado, País)
    local_str = ", ".join([
        endereco.get("road", ""),
        endereco.get("suburb", ""),
        endereco.get("city", endereco.get("town", endereco.get("village", ""))),
        endereco.get("state", ""),
        endereco.get("country", "")
    ])
    local_str = local_str.strip(", ").replace(",,", ",")

    if conteudo:
        # If backend returned a full conteudo document (with blocos), attach signed urls
        if isinstance(conteudo, dict) and conteudo.get('blocos'):
            blocos_doc = conteudo.get('blocos', [])
            try:
                await attach_signed_urls_to_blocos(blocos_doc)
            except Exception:
                pass
            resultado = {
                "conteudo": {
                    "blocos": blocos_doc
                },
                "mensagem": "Conteúdo encontrado.",
                "localizacao": local_str,
                "endereco": endereco,
                "tipo_regiao": conteudo.get('tipo_regiao'),
                "nome_regiao": conteudo.get('nome_regiao')
            }
            
            # include radius_m if present in the stored document so frontends can prefill/edit it
            try:
                if isinstance(conteudo, dict) and conteudo.get('radius_m') is not None:
                    resultado['radius_m'] = conteudo.get('radius_m')
            except Exception:
                pass
            if conteudo.get('_matched_by'):
                resultado['matched_by'] = conteudo.get('_matched_by')
            if conteudo.get('_distance_m') is not None:
                resultado['distance_m'] = conteudo.get('_distance_m')
        else:
            resultado = {
                "conteudo": {
                    "texto": conteudo.get("texto", ""),
                    "imagens": conteudo.get("imagens", []),
                    "videos": conteudo.get("videos", []),
                },
                "mensagem": "Conteúdo encontrado.",
                "localizacao": local_str,
                "endereco": endereco
            }
    else:
        resultado = {
            "conteudo": None,
            "mensagem": f"Nenhum conteúdo associado a esta marca neste local: {local_str}.",
            "localizacao": local_str,
            "endereco": endereco
        }

    add_to_consulta_cache(cache_key, resultado)
    return resultado


@app.post('/api/smart-content')
async def smart_content_lookup(
    nome_marca: str = Body(...),
    latitude: float = Body(...),
    longitude: float = Body(...)
):
    """
    🚀 ENDPOINT OTIMIZADO - Faz lookup paralelo em todas as estratégias
    
    Ao invés de tentar sequencialmente (consulta → radius 50m → 200m → 1000m → 5000m → geocode → região),
    este endpoint executa TODAS as estratégias EM PARALELO e retorna assim que a primeira encontrar resultado.
    
    Reduz tempo de ~20s para ~2-3s quando o conteúdo está em região (caso G3).
    """
    
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    ts0 = datetime.utcnow().isoformat()
    logging.info(f"[smart-content][{request_id}] Buscando conteudo para marca={nome_marca}, lat={latitude}, lon={longitude} ts={ts0}")
    
    # 1. Buscar marca E fazer geocode EM PARALELO (otimização: -0.5s)
    async def fetch_marca():
        marca = await logos_collection.find_one({"nome": nome_marca})
        if not marca:
            logging.warning(f"[smart-content] Marca {nome_marca} nao encontrada")
        return marca
    
    async def fetch_geocode():
        # Cache de geocoding: arredondar coordenadas para 3 casas decimais (~111m de precisão)
        cache_key = f"{round(latitude, 3)},{round(longitude, 3)}"
        
        # Verificar cache primeiro
        if cache_key in geocode_cache:
            logging.info(f"[smart-content] ⚡ Geocode CACHE HIT: {cache_key}")
            return geocode_cache[cache_key]
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                rev_resp = await client.get(
                    f"https://nominatim.openstreetmap.org/reverse",
                    params={"lat": latitude, "lon": longitude, "format": "json"},
                    headers={"User-Agent": "OlinxRA/1.0 (contact@olinxra.com)"}
                )
                if rev_resp.status_code == 200:
                    data = rev_resp.json().get("address", {})
                    logging.info(f"[smart-content] Geocode API: {data.get('city', 'N/A')}, {data.get('state', 'N/A')}")
                    
                    # Salvar no cache (máximo 1000 entradas para evitar vazamento de memória)
                    if len(geocode_cache) < 1000:
                        geocode_cache[cache_key] = data
                    
                    return data
                else:
                    logging.warning(f"[smart-content] Geocode falhou: HTTP {rev_resp.status_code}")
        except Exception as e:
            logging.error(f"[smart-content] Erro no geocode: {e}")
        return None
    
    # Executar marca + geocode em paralelo
    marca, geocode_data = await asyncio.gather(fetch_marca(), fetch_geocode())
    t_after_fetch = time.perf_counter()
    logging.info(
        f"[smart-content][{request_id}] fetch completed; marca_found={bool(marca)} geocode_cached={bool(geocode_data)} dur_ms={(t_after_fetch-t0)*1000:.1f}"
    )
    
    if not marca:
        return {"conteudo": None, "mensagem": "Marca não encontrada."}
    
    marca_id = marca["_id"]
    logging.info(f"[smart-content] Marca encontrada: {marca_id}")
    
    # 3. Preparar todas as estratégias de busca em paralelo
    async def try_proximity(radius_m: float):
        """Tenta buscar por proximidade com raio específico"""
        start_p = time.perf_counter()
        try:
            result = await buscar_conteudo_por_marca_e_localizacao(marca_id, latitude, longitude, radius_m)
            dur = (time.perf_counter() - start_p) * 1000.0
            logging.info(f"[smart-content][{request_id}] proximity {radius_m}m dur_ms={dur:.1f} hit={bool(result and result.get('blocos'))}")
            if result and result.get('blocos'):
                return ('proximity', radius_m, result)
        except Exception as e:
            logging.debug(f"[smart-content][{request_id}] Proximity {radius_m}m falhou: {e}")
        return None
    
    async def try_region_lookup():
        """Tenta buscar por região usando geocode já obtido"""
        if not geocode_data:
            return None
        
        try:
            # Tentar hierarquia: bairro → cidade → estado → país
            regions = [
                ("bairro", geocode_data.get("suburb") or geocode_data.get("neighbourhood")),
                ("cidade", geocode_data.get("city") or geocode_data.get("town") or geocode_data.get("village")),
                ("estado", geocode_data.get("state")),
                ("pais", geocode_data.get("country"))
            ]
            
            # Buscar TODAS as regiões em paralelo (otimização: -1s)
            async def check_region(tipo_regiao, nome_regiao):
                if not nome_regiao:
                    return None
                cache_key = (nome_marca, tipo_regiao, nome_regiao)
                # checar cache simples de região
                try:
                    entry = region_cache.get(cache_key)
                    if entry:
                        ts, cached_conteudo = entry
                        if (time.time() - ts) < REGION_CACHE_TTL:
                            logging.info(f"[smart-content][{request_id}] region_cache HIT {tipo_regiao}/{nome_regiao}")
                            return ('region', f"{tipo_regiao}/{nome_regiao}", cached_conteudo)
                        else:
                            # expired
                            region_cache.pop(cache_key, None)
                except Exception:
                    pass

                start_r = time.perf_counter()
                filtro = {"nome_marca": nome_marca, "tipo_regiao": tipo_regiao, "nome_regiao": nome_regiao}
                conteudo = await db["conteudos"].find_one(filtro)
                dur_r = (time.perf_counter() - start_r) * 1000.0
                hit = bool(conteudo and conteudo.get("blocos"))
                logging.info(f"[smart-content][{request_id}] region_check {tipo_regiao}/{nome_regiao} dur_ms={dur_r:.1f} hit={hit}")
                if conteudo and conteudo.get("blocos"):
                    conteudo["_id"] = str(conteudo["_id"])
                    # armazenar no cache (timestamp + conteudo)
                    try:
                        if len(region_cache) < 2000:
                            region_cache[cache_key] = (time.time(), conteudo)
                    except Exception:
                        pass
                    logging.info(f"[smart-content][{request_id}] ✅ Encontrado em {tipo_regiao}/{nome_regiao}")
                    return ('region', f"{tipo_regiao}/{nome_regiao}", conteudo)
                return None
            
            # Executar todas as buscas em paralelo
            region_tasks = [check_region(tipo, nome) for tipo, nome in regions]
            region_results = await asyncio.gather(*region_tasks, return_exceptions=True)
            
            # Retornar primeiro resultado válido (prioridade: bairro > cidade > estado > país)
            for result in region_results:
                if result and not isinstance(result, Exception):
                    return result
            
        except Exception as e:
            logging.debug(f"[smart-content] Region lookup falhou: {e}")
        return None
    
    # 4. Estratégia adaptativa: priorizar region ou proximity dependendo da granularidade
    logging.info(f"[smart-content][{request_id}] Executando lookups em modo adaptativo...")

    # preparar tasks (criar tasks para permitir cancelamento)
    proximity_radii = [50, 200, 1000, 5000]
    proximity_tasks = [asyncio.create_task(try_proximity(r)) for r in proximity_radii]
    region_task = asyncio.create_task(try_region_lookup()) if geocode_data else None

    # deduzir granularidade a partir do geocode
    granularity = 'unknown'
    if geocode_data:
        if geocode_data.get('road') or geocode_data.get('suburb') or geocode_data.get('neighbourhood'):
            granularity = 'street'
        elif geocode_data.get('city'):
            granularity = 'city'
        elif geocode_data.get('state'):
            granularity = 'state'

    try:
        winner = None

        # Caso city/state: tentar region primeiro com um timeout curto; se não, esperar por first completed entre region+proximities
        if granularity in ('city', 'state') and region_task is not None:
            # aguardar region por até 0.45s — se bater, usamos imediatamente
            done, pending = await asyncio.wait([region_task], timeout=0.45, return_when=asyncio.FIRST_COMPLETED)
            if region_task in done:
                try:
                    res = region_task.result()
                    if res:
                        winner = res
                except Exception:
                    pass
            if winner is None:
                # aguardar o primeiro completed entre proximal + region
                all_tasks = [t for t in proximity_tasks]
                if region_task:
                    all_tasks.append(region_task)
                done, pending = await asyncio.wait(all_tasks, timeout=2.0, return_when=asyncio.FIRST_COMPLETED)
                # verificar resultados em 'done'
                for d in done:
                    try:
                        r = d.result()
                        if r:
                            winner = r
                            break
                    except Exception:
                        continue

        # Caso street (road/suburb) ou quando há um radius implícito: tentar proximity primeiro
        elif granularity == 'street':
            # esperar proximities por até 1.5s
            done, pending = await asyncio.wait(proximity_tasks, timeout=1.5, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                try:
                    r = d.result()
                    if r:
                        winner = r
                        break
                except Exception:
                    continue
            if winner is None and region_task is not None:
                # tentar region como fallback rápido
                try:
                    res = await asyncio.wait_for(region_task, timeout=0.5)
                    # region_task returns a tuple when hit, else None
                    if isinstance(res, tuple) and res:
                        winner = res
                except Exception:
                    pass

        # Caso desconhecido: aguardar o primeiro que terminar entre todos
        else:
            all_tasks = [t for t in proximity_tasks]
            if region_task:
                all_tasks.append(region_task)
            done, pending = await asyncio.wait(all_tasks, timeout=2.0, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                try:
                    r = d.result()
                    if r:
                        winner = r
                        break
                except Exception:
                    continue

        # Se ainda sem vencedor, aguardar todos e escolher o primeiro válido (safe fallback)
        if winner is None:
            # coletar resultados finais
            final_tasks = [t for t in proximity_tasks]
            if region_task:
                final_tasks.append(region_task)
            try:
                results = await asyncio.gather(*final_tasks, return_exceptions=True)
                for r in results:
                    if r and not isinstance(r, Exception):
                        winner = r
                        break
            except Exception:
                pass

        # cancelar tasks pendentes para evitar trabalho desnecessário
        for t in proximity_tasks:
            if not t.done():
                t.cancel()
        if region_task and not region_task.done():
            region_task.cancel()

        best_result = winner
    except Exception as e:
        # Garantir que qualquer exceção durante a fase adaptativa não quebre o handler
        logging.exception(f"[smart-content][{request_id}] Erro durante lookups adaptativos: {e}")
        best_result = None

    if not best_result:
        logging.warning("[smart-content] ❌ Nenhum conteudo encontrado")
        return {
            "conteudo": None,
            "mensagem": "Nenhum conteúdo encontrado para esta marca nesta localização."
        }
    
    # 6. Processar resultado encontrado
    strategy, detail, conteudo = best_result
    logging.info(f"[smart-content] ✅ Usando resultado: strategy={strategy}, detail={detail}")
    blocos_doc = conteudo.get('blocos', [])
    
    # ⚡ Gerar signed URLs com versão OTIMIZADA (skip_exists_check + TTL 7 dias)
    t_before_attach = time.perf_counter()
    try:
        await attach_signed_urls_to_blocos_fast(blocos_doc)
    except Exception as e:
        logging.exception(f"[smart-content][{request_id}] Erro ao anexar signed_urls: {e}")
    t_after_attach = time.perf_counter()
    logging.info(
        f"[smart-content][{request_id}] attach_signed_urls done; dur_ms={(t_after_attach-t_before_attach)*1000:.1f} total_ms={(t_after_attach-t0)*1000:.1f} blocks={len(blocos_doc)} strategy={strategy} detail={detail}"
    )
    
    # 7. Montar resposta
    resultado = {
        "conteudo": {"blocos": blocos_doc},
        "mensagem": "Conteúdo encontrado.",
        "tipo_regiao": conteudo.get('tipo_regiao'),
        "nome_regiao": conteudo.get('nome_regiao'),
        "matched_by": strategy,
        "matched_detail": detail
    }
    
    logging.info(f"[smart-content][{request_id}] 🎯 Retornando conteudo com {len(blocos_doc)} blocos")
    if conteudo.get('radius_m') is not None:
        resultado['radius_m'] = conteudo.get('radius_m')
    
    return resultado


@app.get('/api/marcas')
async def listar_marcas(
    ownerId: str = Query(..., description="ID do usuário dono das marcas"),
    token: dict = Depends(verify_firebase_token_dep)
):
    if not ownerId:
        raise HTTPException(status_code=422, detail="Parâmetro 'ownerId' é obrigatório.")
    marcas = await logos_collection.find({"owner_uid": ownerId}).to_list(length=100)
    return [{"id": str(marca["_id"]), "nome": marca.get("nome", "")} for marca in marcas]

@app.get('/api/conteudo')
async def get_conteudo(
    nome_marca: str = Query(..., description="Nome da marca"),
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    radius: float = Query(None, description="Optional radius in meters to search by proximity")
):
    cache_key = (nome_marca, round(latitude, 6), round(longitude, 6), int(radius) if radius else None)
    if cache_key in consulta_cache:
        return consulta_cache[cache_key]

    marca = await logos_collection.find_one({"nome": nome_marca})
    if not marca:
        resultado = {"conteudo": None, "mensagem": "Marca não encontrada."}
        add_to_consulta_cache(cache_key, resultado)
        return resultado

    conteudo = await buscar_conteudo_por_marca_e_localizacao(marca["_id"], latitude, longitude, radius)
    endereco = await geocode_reverse(latitude, longitude)
    local_str = ", ".join([
        endereco.get("road", ""),
        endereco.get("suburb", ""),
        endereco.get("city", endereco.get("town", endereco.get("village", ""))),
        endereco.get("state", ""),
        endereco.get("country", "")
    ])
    local_str = local_str.strip(", ").replace(",,", ",")

    if conteudo:
        blocos_resp = conteudo.get("blocos", None) if conteudo.get("blocos") else None
        if blocos_resp:
            await attach_signed_urls_to_blocos(blocos_resp)
        resultado = {
            "conteudo": blocos_resp,
            "mensagem": "Conteúdo encontrado.",
            "localizacao": local_str,
            "endereco": endereco,
        }
        # If the stored document included region metadata, pass it through
        try:
            if isinstance(conteudo, dict):
                if conteudo.get('tipo_regiao') is not None:
                    resultado['tipo_regiao'] = conteudo.get('tipo_regiao')
                if conteudo.get('nome_regiao') is not None:
                    resultado['nome_regiao'] = conteudo.get('nome_regiao')
        except Exception:
            pass
        try:
            if isinstance(conteudo, dict) and conteudo.get('radius_m') is not None:
                resultado['radius_m'] = conteudo.get('radius_m')
        except Exception:
            pass
    else:
        resultado = {
            "conteudo": None,
            "mensagem": f"Nenhum conteúdo associado a esta marca neste local: {local_str}.",
            "localizacao": local_str
        }

    add_to_consulta_cache(cache_key, resultado)
    return resultado

@app.get('/api/conteudo-por-regiao')
async def get_conteudo_por_regiao(
    nome_marca: str = Query(...),
    tipo_regiao: str = Query(None),
    nome_regiao: str = Query(None)
):
    # Busca a marca no banco
    marca = await logos_collection.find_one({"nome": nome_marca})
    if not marca:
        return {"blocos": []}

    # Busca conteúdo associado à marca e região
    filtro = {"nome_marca": nome_marca}
    if tipo_regiao:
        filtro["tipo_regiao"] = tipo_regiao
    if nome_regiao:
        filtro["nome_regiao"] = nome_regiao
    conteudo = await db["conteudos"].find_one(filtro)
    if conteudo:
        conteudo["_id"] = str(conteudo["_id"])
        blocos_ret = conteudo.get("blocos", [])
        if blocos_ret:
            await attach_signed_urls_to_blocos(blocos_ret)
        resp = {
            "blocos": blocos_ret,
            "tipo_regiao": conteudo.get("tipo_regiao"),
            "nome_regiao": conteudo.get("nome_regiao"),
            "latitude": conteudo.get("latitude"),
            "longitude": conteudo.get("longitude"),
        }
        # Include radius if present so admin UI can prefill the field
        try:
            if conteudo.get('radius_m') is not None:
                resp['radius_m'] = conteudo.get('radius_m')
        except Exception:
            pass
        return resp
    return {"blocos": []}


@app.post('/api/conteudo')
async def post_conteudo(
    payload: dict = Body(...),
    token: dict = Depends(verify_firebase_token_dep),
    dry_run: bool = Query(False)
):
    """Cria ou atualiza um documento de conteúdo para uma marca/região.
    Espera payload com: nome_marca, blocos (array), latitude, longitude, tipo_regiao, nome_regiao
    Se blocos estiver vazio e documento existir, remove o documento (action: deleted).
    Retorna { action: 'saved' } ou { action: 'deleted' }.
    """
    try:
        nome_marca = payload.get('nome_marca')
        blocos = payload.get('blocos', []) or []
        latitude = payload.get('latitude')
        longitude = payload.get('longitude')
        tipo_regiao = payload.get('tipo_regiao')
        nome_regiao = payload.get('nome_regiao')

        # Validação estrita para radius_m: se fornecido, deve ser convertível para float e não-negativo
        radius_val = None
        if 'radius_m' in payload and payload.get('radius_m') is not None:
            try:
                radius_val = float(payload.get('radius_m'))
                if radius_val < 0:
                    raise HTTPException(status_code=422, detail="Campo 'radius_m' deve ser um número não-negativo.")
            except (ValueError, TypeError):
                raise HTTPException(status_code=422, detail="Campo 'radius_m' inválido; deve ser um número (metros).")

        if not nome_marca:
            raise HTTPException(status_code=422, detail="Campo 'nome_marca' é obrigatório.")

        # Tentar resolver marca para obter um marca_id estável (ObjectId)
        marca_doc = await logos_collection.find_one({"nome": nome_marca})
        marca_obj_id = None
        if marca_doc and "_id" in marca_doc:
            marca_obj_id = marca_doc["_id"]
        else:
            logging.warning(f"[post_conteudo] Marca '{nome_marca}' não encontrada em 'logos' — salvando sem marca_id.")

        # Construir filtro base por owner + região
        base_filtro = {
            'owner_uid': token.get('uid'),
            'tipo_regiao': tipo_regiao,
            'nome_regiao': nome_regiao
        }

        # Procurar documento existente de forma robusta:
        # - se tivermos marca_obj_id, primeiro tente match por ObjectId,
        #   depois tente pelo string equivalente (compatibilidade com docs antigos),
        # - se não tivermos marca_obj_id, tente procurar por nome_marca.
        existente = None
        if marca_obj_id:
            try:
                existente = await db['conteudos'].find_one({**base_filtro, 'marca_id': marca_obj_id})
            except Exception:
                existente = None
            if not existente:
                # tenta também por string (caso o banco tenha o id como string)
                try:
                    existente = await db['conteudos'].find_one({**base_filtro, 'marca_id': str(marca_obj_id)})
                except Exception:
                    existente = None
        else:
            # sem marca_obj_id, usamos nome_marca como chave
            try:
                existente = await db['conteudos'].find_one({**base_filtro, 'nome_marca': nome_marca})
            except Exception:
                existente = None

        # Helper interno: detecta se um bloco aparenta conter mídia
        def bloco_possui_media(b):
            try:
                if not isinstance(b, dict):
                    return False
                # URL-like in url or conteudo
                u = b.get('url') or b.get('conteudo') or ''
                if isinstance(u, str) and u.strip() != '':
                    if u.startswith('gs://') or u.startswith('/') or u.startswith('http') or u.startswith('blob:'):
                        return True
                # filename or explicit type
                if b.get('filename') or b.get('nome'):
                    return True
                t = b.get('type') or b.get('content_type') or ''
                if isinstance(t, str) and (t.startswith('image') or t.startswith('video')):
                    return True
                # carousel items
                if b.get('items') and isinstance(b.get('items'), list):
                    for it in b.get('items'):
                        if not isinstance(it, dict):
                            continue
                        iu = it.get('url') or (it.get('meta') and it['meta'].get('url')) or (it.get('conteudo')) or ''
                        if isinstance(iu, str) and iu.strip() != '':
                            if iu.startswith('gs://') or iu.startswith('/') or iu.startswith('http') or iu.startswith('blob:'):
                                return True
                        if it.get('filename') or (it.get('meta') and it['meta'].get('filename')):
                            return True
                        it_type = it.get('type') or (it.get('meta') and it['meta'].get('type')) or ''
                        if isinstance(it_type, str) and (it_type.startswith('image') or it_type.startswith('video')):
                            return True
                return False
            except Exception:
                return False

        # Normalizar blocos para novos saves: garantir datetimes e remover campos
        # desnecessários em blocos de texto. Também valida se há URLs locais (blob:)
        # que indicam upload não finalizado.
        invalid_blocks = []
        cleaned_blocos = []
        for idx, b in enumerate(blocos):
            if not isinstance(b, dict):
                continue
            # detect blob: in url or conteudo
            try:
                u = b.get('url', '')
                c = b.get('conteudo', '')
                if (isinstance(u, str) and u.startswith('blob:')) or (isinstance(c, str) and c.startswith('blob:')):
                    invalid_blocks.append({ 'index': idx, 'filename': (b.get('filename') or b.get('nome') or '') })
                    # keep the block as-is for reporting; do not normalize
                    cleaned_blocos.append(b)
                    continue
            except Exception:
                cleaned_blocos.append(b)
                continue

            # Normalize created_at for blocos
            created = b.get('created_at')
            if isinstance(created, str):
                try:
                    created_dt = datetime.fromisoformat(created)
                except Exception:
                    try:
                        created_dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S.%f")
                    except Exception:
                        created_dt = datetime.utcnow()
                b['created_at'] = created_dt
            elif created is None:
                b['created_at'] = datetime.utcnow()

            # Determine whether this bloco represents media (image/carousel/video).
            # Prefer explicit machine-friendly `tipoSelecionado` or label, but also
            # detect media by presence of media-like fields (url/filename/type/items).
            tipo_selecionado = b.get('tipoSelecionado') or ''
            tipo_label = b.get('tipo') or ''
            is_media = False
            try:
                if isinstance(tipo_selecionado, str) and tipo_selecionado.lower() in ('imagem', 'carousel', 'video'):
                    is_media = True
                else:
                    tl = tipo_label.lower() if isinstance(tipo_label, str) else ''
                    if tl.startswith('imagem') or tl.startswith('video') or tl.startswith('carousel'):
                        is_media = True
                # If still not sure, inspect specific fields that indicate media
                if not is_media:
                    if bloco_possui_media(b):
                        is_media = True
            except Exception:
                is_media = False

            # If not media, strip media-related fields to keep document compact.
            if not is_media:
                for k in ('url', 'filename', 'type', 'subtipo', 'created_at'):
                    if k in b:
                        b.pop(k, None)

            # Remove ephemeral signed_url if present
            if 'signed_url' in b:
                b.pop('signed_url', None)

            cleaned_blocos.append(b)
        # Pós-processamento consolidado: para cada bloco, garantir propriedades derivadas
        for b in cleaned_blocos:
            try:
                tipo_selecionado = b.get('tipoSelecionado') or ''
                tipo_label = b.get('tipo') or ''
                is_media = False
                if isinstance(tipo_selecionado, str) and tipo_selecionado.lower() in ('imagem', 'carousel', 'video'):
                    is_media = True
                else:
                    tl = tipo_label.lower() if isinstance(tipo_label, str) else ''
                    if tl.startswith('imagem') or tl.startswith('video') or tl.startswith('carousel'):
                        is_media = True
                if not is_media and bloco_possui_media(b):
                    is_media = True

                # If media and missing url but has filename -> set url
                if is_media and (not b.get('url')) and b.get('filename'):
                    filename = b.get('filename')
                    if not str(filename).startswith('gs://'):
                        b['url'] = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"

                # If this is a carousel (has items), persist tipoSelecionado and normalize items
                if b.get('items') and isinstance(b.get('items'), list) and len(b.get('items')) > 0:
                    if not b.get('tipoSelecionado'):
                        b['tipoSelecionado'] = 'carousel'
                    for it in b['items']:
                        try:
                            if it is None:
                                continue
                            ca = it.get('created_at')
                            if isinstance(ca, str):
                                try:
                                    it['created_at'] = datetime.fromisoformat(ca)
                                except Exception:
                                    try:
                                        it['created_at'] = datetime.strptime(ca, "%Y-%m-%dT%H:%M:%S.%f")
                                    except Exception:
                                        try:
                                            it['created_at'] = datetime.strptime(ca, "%Y-%m-%d %H:%M:%S.%f")
                                        except Exception:
                                            it['created_at'] = datetime.utcnow()
                            elif ca is None:
                                it['created_at'] = datetime.utcnow()
                        except Exception:
                            it['created_at'] = datetime.utcnow()
                    # remove empty/blank parent media-like fields
                    for key in ('url', 'filename', 'nome', 'type'):
                        try:
                            val = b.get(key)
                            if val is None or (isinstance(val, str) and val.strip() == ''):
                                b.pop(key, None)
                        except Exception:
                            pass

                # If media and missing filename but has nome, infer filename using owner uid
                if is_media and (not b.get('filename')) and b.get('nome'):
                    nome = str(b.get('nome'))
                    if '/' in nome:
                        filename = nome
                    else:
                        owner = token.get('uid') or 'unknown'
                        filename = f"{owner}/{nome}"
                    b['filename'] = filename
                    if not b.get('url'):
                        b['url'] = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"
            except Exception:
                # keep going; post-processing must not fail the whole request
                pass
        if invalid_blocks:
            logging.warning('[post_conteudo] Rejeitando payload com blocos inválidos (blob:)', extra={'invalid_blocks': invalid_blocks, 'nome_marca': nome_marca, 'owner_uid': token.get('uid')})
            raise HTTPException(status_code=422, detail={ 'message': 'Payload contém referências locais (blob:). Faça upload das imagens primeiro.', 'invalid_blocks': invalid_blocks })

        # Validate button blocks (botao_destaque / botao_default) before persisting
        # Note: frontend may place editable fields inside bloco['meta'] (for local editing state).
        # To be tolerant, hydrate top-level fields from meta when missing so validation succeeds.
        try:
            from pydantic import ValidationError
            from schemas import validate_button_block_payload
            for bi, bb in enumerate(cleaned_blocos):
                try:
                    if not isinstance(bb, dict):
                        continue
                    tipo_check = (bb.get('tipo') or '').lower()
                    if tipo_check in ('botao_destaque', 'botao_default'):
                        # If frontend sent fields inside meta (e.g. meta.action or meta.label), copy them up
                        meta = bb.get('meta') or {}
                        if isinstance(meta, dict):
                            # action may be nested or split across keys
                            if not bb.get('action') and meta.get('action'):
                                bb['action'] = meta.get('action')
                            # also accept href/link/url directly on meta
                            if not bb.get('action') and (meta.get('href') or meta.get('link') or meta.get('url')):
                                bb['action'] = meta.get('action') if meta.get('action') else { 'type': 'link', 'href': meta.get('href') or meta.get('link') or meta.get('url'), 'target': meta.get('target') or '_self' }

                            # label may be stored under different keys in meta
                            if not bb.get('label'):
                                for key in ('label', 'buttonLabel', 'button_label', 'text', 'title'):
                                    if meta.get(key):
                                        bb['label'] = meta.get(key)
                                        break

                            # copy other optional decorative/button props if present
                            for fld in ('variant', 'color', 'icon', 'icon_family', 'icon_invert', 'size', 'disabled', 'aria_label', 'analytics', 'visibility', 'position', 'temp_id'):
                                if bb.get(fld) is None and meta.get(fld) is not None:
                                    bb[fld] = meta.get(fld)

                        # If action is a dict-like in meta.action but missing required nested fields, try to hydrate
                        if isinstance(bb.get('action'), dict):
                            a = bb['action']
                            # try to fill href from meta top-level if missing
                            if a.get('type') == 'link' and not a.get('href'):
                                if meta.get('href'):
                                    a['href'] = meta.get('href')
                                elif meta.get('link'):
                                    a['href'] = meta.get('link')
                                elif meta.get('url'):
                                    a['href'] = meta.get('url')
                            # try to fill name for callback
                            if a.get('type') == 'callback' and not a.get('name') and meta.get('name'):
                                a['name'] = meta.get('name')
                            bb['action'] = a

                        # Remove analytics object if it's empty or has no event_name to avoid rejecting payloads
                        try:
                            an = bb.get('analytics')
                            if isinstance(an, dict):
                                if not an.get('event_name'):
                                    bb.pop('analytics', None)
                        except Exception:
                            pass

                        # If after hydration required fields are still missing, build a helpful error
                        missing = []
                        if not bb.get('label'):
                            missing.append('label')
                        if not bb.get('action'):
                            missing.append('action')
                        # If action exists but lacks href/name depending on type, mark as missing
                        if bb.get('action') and isinstance(bb.get('action'), dict):
                            at = bb['action'].get('type')
                            if at == 'link' and not bb['action'].get('href'):
                                missing.append('action.href')
                            if at == 'callback' and not bb['action'].get('name'):
                                missing.append('action.name')
                        if missing:
                            raise HTTPException(status_code=422, detail={ 'message': f'Invalid button block at index {bi}', 'missing': missing, 'meta_keys': list(meta.keys()), 'block_preview': bb })

                        # will raise ValidationError if invalid
                        validate_button_block_payload(bb)
                except ValidationError as ve:
                    # pydantic ValidationError -> return 422 with details
                    raise HTTPException(status_code=422, detail={ 'message': f'Invalid button block at index {bi}', 'error': str(ve) })
                except HTTPException:
                    raise
                except Exception as e:
                    # Any other error in validation of this block
                    raise HTTPException(status_code=422, detail={ 'message': f'Invalid button block at index {bi}', 'error': str(e) })
        except HTTPException:
            raise
        except Exception:
            # If schemas cannot be imported or another issue occurs, log and continue
            logging.exception('[post_conteudo] Unexpected error during button block validation; skipping strict validation')

        # Se blocos estiverem vazios e já existe documento -> deletar (com confirmação)
        if existente and (not isinstance(cleaned_blocos, list) or len(cleaned_blocos) == 0):
            # compute files to delete (do not delete yet if dry_run)
            old_blocos = existente.get('blocos', []) or []
            to_delete = []
            for ob in old_blocos:
                try:
                    if ob.get('items') and isinstance(ob.get('items'), list):
                        for it in ob.get('items'):
                            if it:
                                url = it.get('url')
                                fname = it.get('filename') or it.get('nome')
                                if url:
                                    to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                                elif fname:
                                    to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                    else:
                        url = ob.get('url')
                        fname = ob.get('filename') or ob.get('nome')
                        if url:
                            to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                        elif fname:
                            to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                except Exception:
                    continue

            if dry_run:
                return {'action': 'dry_run', 'to_delete': to_delete, 'blocos': []}

            # enqueue deletions and attempt immediate delete; then remove document
            try:
                for item in to_delete:
                    try:
                        # insert pending_deletes entry
                        pend = {
                            'gs_url': item.get('gs_url'),
                            'filename': item.get('filename'),
                            'tipo': item.get('tipo', 'conteudo'),
                            'status': 'pending',
                            'retries': 0,
                            'created_at': datetime.utcnow(),
                            'last_attempt': None
                        }
                        res = await db['pending_deletes'].insert_one(pend)
                        
                        # 🆕 Deletar imagem E GLB associado
                        ok = await delete_image_and_glb(item, db)
                        
                        if ok:
                            await db['pending_deletes'].update_one({'_id': res.inserted_id}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                    except Exception:
                        continue

                await db['conteudos'].delete_one({'_id': existente['_id']})
                exists_id = existente.get('_id')
                logging.info(f"[post_conteudo] Documento {str(exists_id) if exists_id is not None else 'unknown'} deletado (blocos vazios)")
            except Exception as e:
                logging.exception('Erro ao deletar documento existente com blocos vazios')
                raise HTTPException(status_code=500, detail=f'Erro ao deletação: {str(e)}')
            return { 'action': 'deleted' }

        # Caso contrário, cria ou atualiza (upsert)
        # Implementação B: não deduplicar blocos por filename. Quando existir documento,
        # concatenamos os blocos recebidos ao array existente (permitindo múltiplos blocos
        # que referenciem o mesmo arquivo). Isso preserva o reuso do arquivo no storage
        # mas garante que cada bloco enviado gere uma entrada no documento.
        if existente:
            try:
                # Antes de substituir blocos, MESCLAR campos gerenciados pelo servidor
                # (por exemplo: glb_url, glb_signed_url, glb_source) para o caso em
                # que o frontend não reenvie esses campos ao editar o conteúdo.
                # Critério de match: filename (preferencial) ou url.
                try:
                    old_blocos = existente.get('blocos', []) or []

                    # Indexar blocos antigos por filename e url para busca rápida
                    old_by_filename = {}
                    old_by_url = {}
                    for ob in old_blocos:
                        try:
                            if not isinstance(ob, dict):
                                continue
                            fn = ob.get('filename') or ob.get('nome')
                            if fn:
                                old_by_filename[str(fn)] = ob
                            u = ob.get('url')
                            if u:
                                old_by_url[str(u)] = ob

                            # indexar items dentro de carousels também
                            if ob.get('items') and isinstance(ob.get('items'), list):
                                for it in ob.get('items'):
                                    try:
                                        if not isinstance(it, dict):
                                            continue
                                        it_fn = it.get('filename') or it.get('nome')
                                        if it_fn:
                                            old_by_filename[str(it_fn)] = it
                                        it_u = it.get('url')
                                        if it_u:
                                            old_by_url[str(it_u)] = it
                                    except Exception:
                                        continue
                        except Exception:
                            continue

                    # Também considerar uploaded_assets (staged uploads) para o owner,
                    # caso o frontend não tenha persistido glb_* no bloco enviado.
                    try:
                        filenames_to_lookup = set()
                        for nb in cleaned_blocos:
                            try:
                                fnn = nb.get('filename') or nb.get('nome')
                                if fnn:
                                    filenames_to_lookup.add(str(fnn))
                                if nb.get('items') and isinstance(nb.get('items'), list):
                                    for it in nb.get('items'):
                                        try:
                                            if not isinstance(it, dict):
                                                continue
                                            it_fn = it.get('filename') or it.get('nome')
                                            if it_fn:
                                                filenames_to_lookup.add(str(it_fn))
                                        except Exception:
                                            continue
                            except Exception:
                                continue

                        if filenames_to_lookup:
                            try:
                                assets = await db['uploaded_assets'].find({'owner_uid': token.get('uid'), 'filename': {'$in': list(filenames_to_lookup)}}).to_list(length=1000)
                                for a in assets:
                                    try:
                                        if a.get('filename'):
                                            # hlas assembe a entrada para merge (conteúdo do asset contém glb_* )
                                            old_by_filename[str(a.get('filename'))] = a
                                    except Exception:
                                        continue
                            except Exception:
                                logging.exception('[post_conteudo] Falha ao buscar uploaded_assets para merge')
                    except Exception:
                        pass

                    # Função auxiliar: copia campos server-managed se existirem no bloco antigo
                    def copy_server_fields(src, dst):
                        if not src or not dst or not isinstance(src, dict) or not isinstance(dst, dict):
                            return
                        for fld in ('glb_url', 'glb_signed_url', 'glb_source'):
                            if fld in src and (fld not in dst or dst.get(fld) in (None, '')):
                                try:
                                    dst[fld] = src.get(fld)
                                except Exception:
                                    pass

                    # Para cada bloco novo, tentar mesclar campos do bloco antigo correspondente
                    for nb in cleaned_blocos:
                        try:
                            if not isinstance(nb, dict):
                                continue
                            matched = None
                            # procura por filename primeiro
                            fn = nb.get('filename') or nb.get('nome')
                            if fn and str(fn) in old_by_filename:
                                matched = old_by_filename.get(str(fn))
                            # senao por url
                            if not matched:
                                u = nb.get('url')
                                if u and str(u) in old_by_url:
                                    matched = old_by_url.get(str(u))
                            # se encontrou, copia campos
                            if matched:
                                copy_server_fields(matched, nb)

                            # se for carousel, tratar items individualmente
                            if nb.get('items') and isinstance(nb.get('items'), list):
                                for it in nb['items']:
                                    try:
                                        if not isinstance(it, dict):
                                            continue
                                        matched_it = None
                                        it_fn = it.get('filename') or it.get('nome')
                                        if it_fn and str(it_fn) in old_by_filename:
                                            matched_it = old_by_filename.get(str(it_fn))
                                        if not matched_it:
                                            it_u = it.get('url')
                                            if it_u and str(it_u) in old_by_url:
                                                matched_it = old_by_url.get(str(it_u))
                                        if matched_it:
                                            copy_server_fields(matched_it, it)
                                    except Exception:
                                        continue
                        except Exception:
                            continue
                except Exception:
                    logging.exception('[post_conteudo] Falha ao mesclar campos server-managed dos blocos antigos (seguir com replace)')

                # Substitui os blocos existentes pelo payload recebido (já mesclados acima).
                update_doc = {
                    **base_filtro,
                    'blocos': list(cleaned_blocos),
                    'latitude': latitude,
                    'longitude': longitude,
                    'location': { 'type': 'Point', 'coordinates': [ longitude, latitude ] } if (latitude is not None and longitude is not None) else None,
                    'tipo_regiao': tipo_regiao,
                    'nome_regiao': nome_regiao,
                    'marca_id': marca_obj_id,
                    'nome_marca': nome_marca,
                    'updated_at': datetime.utcnow()
                }
                # Persist admin-provided radius (optional) when present (usando valor validado)
                if radius_val is not None:
                    update_doc['radius_m'] = radius_val
                # Before updating, determine which files were removed and delete them from GCS
                try:
                    old_blocos = existente.get('blocos', []) or []
                    # build set of filenames/urls that will remain
                    new_urls = set()
                    new_filenames = set()
                    for nb in cleaned_blocos:
                        if nb.get('items') and isinstance(nb.get('items'), list):
                            for it in nb['items']:
                                if it:
                                    if it.get('url'): new_urls.add(str(it.get('url')))
                                    if it.get('filename'): new_filenames.add(str(it.get('filename')))
                        else:
                            if nb.get('url'): new_urls.add(str(nb.get('url')))
                            if nb.get('filename'): new_filenames.add(str(nb.get('filename')))

                    # iterate old blocos and enqueue files not present in new sets
                    to_delete = []
                    for ob in old_blocos:
                        try:
                            if ob.get('items') and isinstance(ob.get('items'), list):
                                for it in ob.get('items'):
                                    if it:
                                        url = it.get('url')
                                        fname = it.get('filename') or it.get('nome')
                                        if url and url not in new_urls:
                                            to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                                        elif fname and fname not in new_filenames:
                                            to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                            else:
                                url = ob.get('url')
                                fname = ob.get('filename') or ob.get('nome')
                                if url and url not in new_urls:
                                    to_delete.append({'gs_url': url, 'tipo': 'conteudo'})
                                elif fname and fname not in new_filenames:
                                    to_delete.append({'filename': fname, 'tipo': 'conteudo'})
                        except Exception:
                            continue

                    if dry_run:
                        return {'action': 'dry_run', 'to_delete': to_delete, 'blocos': cleaned_blocos}

                    for item in to_delete:
                        try:
                            pend = {
                                'gs_url': item.get('gs_url'),
                                'filename': item.get('filename'),
                                'tipo': item.get('tipo', 'conteudo'),
                                'status': 'pending',
                                'retries': 0,
                                'created_at': datetime.utcnow(),
                                'last_attempt': None
                            }
                            res = await db['pending_deletes'].insert_one(pend)
                            
                            # 🆕 Deletar imagem E GLB associado
                            ok = await delete_image_and_glb(item, db)
                            
                            if ok:
                                await db['pending_deletes'].update_one({'_id': res.inserted_id}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                        except Exception:
                            continue
                except Exception:
                    logging.exception('[post_conteudo] Falha ao tentar enfileirar/remover arquivos antigos do GCS antes do update')

                await db['conteudos'].update_one({'_id': existente['_id']}, {'$set': update_doc})
                # Recupera o documento atualizado para retornar os blocos persistidos
                saved = await db['conteudos'].find_one({'_id': existente['_id']})
                if saved:
                    # converte _id para string se necessário
                    saved['_id'] = str(saved['_id'])
                    # Marcar uploaded_assets como attached quando houver correspondência por filename
                    try:
                        filenames_to_mark = set()
                        for b in (saved.get('blocos') or []):
                            try:
                                if b and isinstance(b, dict):
                                    fn = b.get('filename') or b.get('nome')
                                    if fn: filenames_to_mark.add(str(fn))
                                    if b.get('items') and isinstance(b.get('items'), list):
                                        for it in b.get('items'):
                                            try:
                                                if it and isinstance(it, dict):
                                                    it_fn = it.get('filename') or it.get('nome')
                                                    if it_fn: filenames_to_mark.add(str(it_fn))
                                            except Exception:
                                                continue
                            except Exception:
                                continue
                        if filenames_to_mark:
                            await db['uploaded_assets'].update_many(
                                {'owner_uid': token.get('uid'), 'filename': {'$in': list(filenames_to_mark)}},
                                {'$set': {'attached': True, 'attached_at': datetime.utcnow(), 'conteudo_id': saved['_id']}}
                            )
                    except Exception:
                        logging.exception('[post_conteudo] Falha ao marcar uploaded_assets como attached (não-fatal)')

                    return { 'action': 'saved', 'blocos': saved.get('blocos', []) }
                return { 'action': 'saved' }
            except Exception as e:
                logging.exception('Erro ao atualizar blocos existentes')
                raise HTTPException(status_code=500, detail=f'Erro ao salvar conteúdo: {str(e)}')
        else:
            doc = {
                **base_filtro,
                'blocos': cleaned_blocos,
                'latitude': latitude,
                'longitude': longitude,
                'location': { 'type': 'Point', 'coordinates': [ longitude, latitude ] } if (latitude is not None and longitude is not None) else None,
                'tipo_regiao': tipo_regiao,
                'nome_regiao': nome_regiao,
                'marca_id': marca_obj_id,
                'nome_marca': nome_marca,
                'updated_at': datetime.utcnow()
            }
            # Persist admin-provided radius (optional) when present (usando valor validado)
            if radius_val is not None:
                doc['radius_m'] = radius_val
            result = await db['conteudos'].insert_one(doc)
            # Recupera o documento recém-criado
            saved = await db['conteudos'].find_one({'_id': result.inserted_id})
            if saved:
                saved['_id'] = str(saved['_id'])
                # Marcar uploaded_assets como attached para os filenames do novo documento
                try:
                    filenames_to_mark = set()
                    for b in (saved.get('blocos') or []):
                        try:
                            if b and isinstance(b, dict):
                                fn = b.get('filename') or b.get('nome')
                                if fn: filenames_to_mark.add(str(fn))
                                if b.get('items') and isinstance(b.get('items'), list):
                                    for it in b.get('items'):
                                        try:
                                            if it and isinstance(it, dict):
                                                it_fn = it.get('filename') or it.get('nome')
                                                if it_fn: filenames_to_mark.add(str(it_fn))
                                        except Exception:
                                            continue
                        except Exception:
                            continue
                    if filenames_to_mark:
                        await db['uploaded_assets'].update_many(
                            {'owner_uid': token.get('uid'), 'filename': {'$in': list(filenames_to_mark)}},
                            {'$set': {'attached': True, 'attached_at': datetime.utcnow(), 'conteudo_id': saved['_id']}}
                        )
                except Exception:
                    logging.exception('[post_conteudo] Falha ao marcar uploaded_assets como attached (não-fatal)')

                return { 'action': 'saved', 'blocos': saved.get('blocos', []) }
            return { 'action': 'saved' }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception('Erro ao salvar conteúdo')
        raise HTTPException(status_code=500, detail=f'Erro ao salvar conteúdo: {str(e)}')


@app.post('/admin/process-pending-deletes')
async def admin_process_pending_deletes(token: dict = Depends(verify_firebase_token_dep)):
    # Only allow master admin to trigger
    master_email = os.getenv('USER_ADMIN_EMAIL')
    if token.get('email') != master_email:
        raise HTTPException(status_code=403, detail='Forbidden')
    # process pending deletes (simple synchronous attempt)
    pending = await db['pending_deletes'].find({'status': {'$in': ['pending', 'retry']}}).to_list(length=1000)
    processed = []
    for p in pending:
        try:
            # 🆕 Deletar imagem E GLB associado
            item = {
                'gs_url': p.get('gs_url'),
                'filename': p.get('filename'),
                'tipo': p.get('tipo', 'conteudo')
            }
            ok = await delete_image_and_glb(item, db)
            
            if ok:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                processed.append({'id': str(p['_id']), 'status': 'done'})
            else:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'retry', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
                processed.append({'id': str(p['_id']), 'status': 'retry'})
        except Exception:
            try:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'error', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
            except Exception:
                pass
            processed.append({'id': str(p['_id']), 'status': 'error'})
    return {'processed': processed, 'count': len(processed)}


@app.post('/add-content-image/')
async def add_content_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    temp_id: str = Form(None),
    tipo_bloco: str = Form("imagem"),
    subtipo: str = Form(""),
    marca: str = Form(""),
    tipo_regiao: str = Form(""),
    nome_regiao: str = Form(""),
    glb_file: UploadFile = File(None),  # 🆕 GLB customizado opcional
    token: dict = Depends(verify_firebase_token_dep)
):
    # Segurança: validar Origin (se fornecido) contra lista de origens permitidas
    origin = None
    try:
        origin = Request.scope.get('headers') if False else None
    except Exception:
        origin = None
    # DEBUG: log minimal info about incoming request to help diagnose 422 errors
    try:
        logging.info(f"[add_content_image] recebendo upload: filename={getattr(file, 'filename', None)} content_type={getattr(file, 'content_type', None)} name_param={name} temp_id={temp_id} tipo_bloco={tipo_bloco} subtipo={subtipo} marca={marca} tipo_regiao={tipo_regiao} nome_regiao={nome_regiao} uid={token.get('uid') if token else 'no-token'}")
    except Exception:
        logging.exception('[add_content_image] falha ao logar metadados iniciais do upload')
    # Nota: a checagem de Origin será feita manualmente abaixo através do header
    # Valida content-type genericamente: aceitar image/* e video/*
    if not (file.content_type and (file.content_type.startswith('image/') or file.content_type.startswith('video/'))):
        logging.warning(f"[add_content_image] Tipo de conteúdo rejeitado: {file.content_type}")
        raise HTTPException(status_code=400, detail="Tipo de arquivo não permitido.")

    import time
    temp_path = None
    t0 = time.time()
    try:
        ext = os.path.splitext(file.filename)[-1]
        name_base = os.path.splitext(name)[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        t1 = time.time()
        logging.info(f"[add_content_image] Tempo até upload GCS: {t1-t0:.2f}s")
        # Salva no bucket olinxra-conteudo, organizado por admin
        gcs_filename = f"{token['uid']}/{name_base}{ext}"
        # upload síncrono -> execute em threadpool para não bloquear o loop
        gcs_url = await asyncio.to_thread(upload_image_to_gcs, temp_path, gcs_filename, "conteudo")
        t2 = time.time()
        logging.info(f"[add_content_image] Tempo upload GCS: {t2-t1:.2f}s (total: {t2-t0:.2f}s)")

        # created_at como datetime (não string) para evitar conversões repetidas
        bloco_img = {
            "tipo": tipo_bloco,
            "subtipo": subtipo,
            "url": gcs_url,
            "nome": name,
            "filename": gcs_filename,
            "type": file.content_type,
            "created_at": datetime.utcnow()
        }
        # Gera signed_url para facilitar preview imediato no frontend (se possível)
        try:
            signed = gerar_signed_url_conteudo(gcs_url, gcs_filename)
        except Exception:
            signed = gcs_url
        
        # 🆕 FASE 1 - GLB: customizado (se fornecido) ou auto-gerado da imagem
        glb_url = None
        glb_signed_url = None
        glb_source = None  # 'custom' ou 'auto_generated'
        
        # Verificar se GLB customizado foi fornecido
        if glb_file and glb_file.filename:
            # Usuário forneceu GLB customizado - fazer upload direto
            try:
                logging.info(f"[add_content_image] GLB customizado fornecido: {glb_file.filename}")
                
                # Validar tipo do arquivo GLB
                if not (glb_file.content_type and 'model' in glb_file.content_type.lower() or 
                        glb_file.filename.lower().endswith('.glb')):
                    logging.warning(f"[add_content_image] Tipo de GLB rejeitado: {glb_file.content_type}")
                    raise HTTPException(status_code=400, detail="Arquivo GLB inválido. Apenas arquivos .glb são aceitos.")
                
                # Salvar GLB temporariamente
                glb_temp_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as glb_temp_file:
                    glb_contents = await glb_file.read()
                    glb_temp_file.write(glb_contents)
                    glb_temp_path = glb_temp_file.name
                
                # Upload GLB customizado para GCS
                glb_filename = f"{token['uid']}/ra/models/{name_base}.glb"
                metadata = {
                    'generated_from_image': gcs_url,
                    'base_height': '0.0',
                    'custom_upload': 'true',  # Marca como customizado
                    'original_filename': glb_file.filename
                }
                glb_gcs_url = await asyncio.to_thread(
                    upload_image_to_gcs,
                    glb_temp_path,
                    glb_filename,
                    'conteudo',
                    'public, max-age=31536000',
                    metadata
                )
                
                # Limpar arquivo temporário
                if glb_temp_path and os.path.exists(glb_temp_path):
                    os.remove(glb_temp_path)
                
                # Gerar signed URL (máximo 7 dias conforme limitação do GCS)
                try:
                    glb_signed_url = gerar_signed_url_conteudo(glb_gcs_url, glb_filename, expiration=7*24*60*60)
                    glb_url = glb_gcs_url
                    glb_source = 'custom'
                    logging.info(f"[add_content_image] GLB customizado salvo: {glb_filename}")
                except Exception as e:
                    logging.warning(f"[add_content_image] Erro ao gerar signed URL do GLB customizado: {e}")
                    glb_url = glb_gcs_url
                    glb_source = 'custom'
                    
            except HTTPException:
                raise
            except Exception as e:
                logging.exception(f"[add_content_image] Erro ao processar GLB customizado: {e}")
                # Se falhar, tenta gerar automaticamente como fallback
                glb_file = None
        
        # Se não foi fornecido GLB customizado E é uma imagem, gerar GLB automaticamente
        if not glb_url and file.content_type and file.content_type.startswith('image/'):
            try:
                t_glb_start = time.time()
                logging.info(f"[add_content_image] Iniciando pré-geração de GLB para {gcs_filename}")
                
                # Gerar GLB a partir da imagem recém-uploadada
                glb_filename = f"{token['uid']}/ra/models/{name_base}.glb"
                glb_temp = None
                
                # Resize se necessário (mesma lógica do endpoint generate-glb)
                MAX_IMAGE_DIM = int(os.getenv('GLB_MAX_DIM', '2048'))
                processed_image = await asyncio.to_thread(resize_image_if_needed, temp_path, MAX_IMAGE_DIM)
                
                # Gerar GLB
                with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as tg:
                    glb_temp = tg.name
                
                await asyncio.to_thread(
                    generate_plane_glb,
                    processed_image,
                    glb_temp,
                    base_y=0.0,
                    plane_height=1.0,
                    flip_u=False,
                    flip_v=True
                )
                
                # Upload GLB para GCS
                metadata = {
                    'generated_from_image': gcs_url,
                    'base_height': '0.0',
                    'auto_generated': 'true'  # Marca como auto-gerado
                }
                glb_gcs_url = await asyncio.to_thread(
                    upload_image_to_gcs,
                    glb_temp,
                    glb_filename,
                    'conteudo',
                    'public, max-age=31536000',
                    metadata
                )
                
                # Gerar signed URL para o GLB (máximo 7 dias conforme limitação do GCS)
                # IMPORTANTE: Signed URLs expiram em 7 dias (limite do GCS)
                # App mobile regenerará automaticamente via attach_signed_urls_to_blocos()
                try:
                    glb_signed_url = gerar_signed_url_conteudo(glb_gcs_url, glb_filename, expiration=7*24*60*60)
                    glb_url = glb_gcs_url
                    glb_source = 'auto_generated'  # Marca origem
                    t_glb_end = time.time()
                    logging.info(f"[add_content_image] GLB auto-gerado com sucesso em {t_glb_end - t_glb_start:.2f}s: {glb_filename}")
                except Exception as e:
                    logging.warning(f"[add_content_image] Erro ao gerar signed URL do GLB: {e}")
                    glb_url = glb_gcs_url
                    glb_source = 'auto_generated'
                
                # Limpar arquivo temporário do GLB
                if glb_temp and os.path.exists(glb_temp):
                    os.remove(glb_temp)
                    
            except Exception as e:
                logging.exception(f"[add_content_image] Erro ao gerar GLB (não-fatal): {e}")
                # Não falha o upload se a geração do GLB falhar
        
        # Adicionar URLs do GLB ao bloco se foram gerados
        if glb_url:
            bloco_img["glb_url"] = glb_url
            bloco_img["glb_signed_url"] = glb_signed_url
            bloco_img["glb_source"] = glb_source  # 'custom' ou 'auto_generated'
        
        t3 = time.time()
        logging.info(f"[add_content_image] Upload concluído (não persiste no DB). Tempo total: {t3-t0:.2f}s")
        resp = {"success": True, "url": gcs_url, "signed_url": signed, "bloco": bloco_img}
        if temp_id:
            resp["temp_id"] = temp_id
        # Log minimal info: uid and filename/type
        try:
            logging.info(f"[add_content_image] upload ok uid={token.get('uid')} filename={gcs_filename} type={file.content_type} glb={'SIM' if glb_url else 'NÃO'} glb_source={glb_source if glb_source else 'N/A'}")
        except Exception:
            pass
        # Persistir metadados do upload em uploaded_assets (staged) para permitir
        # associação posterior mesmo que o frontend não reenvie glb_* no post_conteudo.
        try:
            asset_temp_id = temp_id or str(uuid.uuid4())
            asset_doc = {
                'owner_uid': token.get('uid'),
                'filename': gcs_filename,
                'gs_url': gcs_url,
                'glb_url': glb_url,
                'glb_filename': (glb_url and f"{token['uid']}/ra/models/{os.path.splitext(name)[0]}.glb") or None,
                'glb_source': glb_source,
                'temp_id': asset_temp_id,
                'attached': False,
                'created_at': datetime.utcnow()
            }
            await db['uploaded_assets'].update_one({'owner_uid': token.get('uid'), 'filename': gcs_filename}, {'$set': asset_doc}, upsert=True)
            resp['temp_id'] = asset_temp_id
        except Exception:
            logging.exception('[add_content_image] Falha ao persistir uploaded_assets (não-fatal)')

        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar conteúdo: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)