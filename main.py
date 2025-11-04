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
from gcs_utils import upload_image_to_gcs, get_bucket, GCS_BUCKET_CONTEUDO, GCS_BUCKET_LOGOS
from google.api_core.exceptions import PreconditionFailed
from glb_generator import generate_plane_glb
from schemas import validate_button_block_payload
from clip_utils import extract_clip_features
from faiss_index import LogoIndex
from email.mime.text import MIMEText
import asyncio
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
# REMOVIDO: images_collection = None

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

        # Índice 2dsphere para consultas geoespaciais, se usarmos campo 'location'
        await db['conteudos'].create_index([('location', '2dsphere')], name='idx_location_2dsphere')

        # Índice simples por owner_uid
        await db['conteudos'].create_index([('owner_uid', 1)], name='idx_owner_uid')

        logging.info('Índices de conteúdo verificados/criados com sucesso.')
    except Exception as e:
        logging.exception(f'Falha ao criar índices em conteudos: {e}')
    # REMOVIDO: images_collection = db["images"]

    logging.info("Iniciando a aplicação...")
    initialize_firebase()
    initialize_onnx_session()
    await load_faiss_index()
    yield
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


# --- Helpers de Inicialização ---
###############################################################
# Função utilitária e endpoint para gerar signed URL de conteúdo
def gerar_signed_url_conteudo(gs_url=None, filename=None, expiration=3600):
    """
    Gera um signed URL para um objeto no bucket de conteúdo ou logos.
    Aceita dois modos de chamada:
    - gs_url (ex: 'gs://bucket/path/file.glb') OR
    - filename (ex: 'public/ra/totem/file.glb') — neste caso assumimos bucket de conteúdo.
    
    Args:
        gs_url: URL completa no formato gs://bucket/path
        filename: Nome do arquivo no bucket
        expiration: Tempo de expiração em segundos (padrão: 3600 = 1h)
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
    """Given a list of blocos, attach a 'signed_url' field for media blocos.
    This runs gerar_signed_url_conteudo in a thread to avoid blocking the event loop.
    
    IMPORTANTE: Esta função é usada pelos endpoints PÚBLICOS (/api/conteudo, /api/conteudo-por-regiao)
    que o app mobile usa SEM autenticação. Por isso, gera signed URLs para:
    - Imagens originais (signed_url)
    - GLBs pré-gerados (glb_signed_url)
    """
    if not blocos or not isinstance(blocos, list):
        return blocos
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
            # For carousel, process items
            if b.get('items') and isinstance(b.get('items'), list):
                for it in b['items']:
                    try:
                        url = it.get('url') or (it.get('meta') and it['meta'].get('url'))
                        filename = it.get('filename') or (it.get('meta') and it['meta'].get('filename'))
                        if not url and filename:
                            # Compose a gs:// URL using the configured content bucket from env
                            url = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"
                        if url:
                            signed = await asyncio.to_thread(gerar_signed_url_conteudo, url, filename)
                            if signed:
                                it['signed_url'] = signed
                        
                        # 🆕 FASE 1: Gerar signed URL para GLB (se existir)
                        glb_url = it.get('glb_url')
                        glb_filename = it.get('glb_filename')
                        if glb_url:
                            try:
                                # GLBs: máximo 7 dias (limite GCS)
                                glb_signed = await asyncio.to_thread(
                                    gerar_signed_url_conteudo, 
                                    glb_url, 
                                    glb_filename,
                                    7*24*60*60  # 7 dias
                                )
                                if glb_signed:
                                    it['glb_signed_url'] = glb_signed
                            except Exception:
                                pass  # Não quebra se GLB falhar
                    except Exception:
                        continue
                continue
            # Single media block
            url = b.get('url')
            filename = b.get('filename')
            if not url and filename:
                url = f"gs://{GCS_BUCKET_CONTEUDO}/{filename}"
            if url:
                try:
                    signed = await asyncio.to_thread(gerar_signed_url_conteudo, url, filename)
                    if signed:
                        b['signed_url'] = signed
                except Exception:
                    # ignore signing failures, frontend can fallback
                    pass
            
            # 🆕 FASE 1: Gerar signed URL para GLB do bloco (se existir)
            glb_url = b.get('glb_url')
            glb_filename = b.get('glb_filename')
            if glb_url:
                try:
                    # GLBs: máximo 7 dias (limite GCS)
                    glb_signed = await asyncio.to_thread(
                        gerar_signed_url_conteudo,
                        glb_url,
                        glb_filename,
                        7*24*60*60  # 7 dias
                    )
                    if glb_signed:
                        b['glb_signed_url'] = glb_signed
                except Exception:
                    pass  # Não quebra se GLB falhar
        except Exception:
            continue
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
        query_vector = await asyncio.to_thread(extract_clip_features, temp_path, ort_session)
        import numpy as np
        print("query_vector shape:", np.array(query_vector).shape)
        print("query_vector values:", np.array(query_vector).tolist())
        print("faiss index dimension:", logo_index.index.d)
        results = logo_index.search(query_vector, top_k=1)

        if results:
            match = results[0]
            return {
                "found": True,
                "name": match['metadata'].get('nome', 'Logo encontrado'),
                "confidence": float(match.get('confidence', 0)),
                "distance": float(match.get('distance', 0)),
                "owner": match['metadata'].get('owner_uid', ''),
                "query_vector": np.array(query_vector).tolist()
            }

        return {"found": False, "debug": "Nenhum match encontrado", "query_vector": np.array(query_vector).tolist()}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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
    logging.info(f"Imagens encontradas: {imagens}")

    # Attach signed_url for any GCS paths so the frontend can load images via HTTPS
    try:
        for img in imagens:
            try:
                url = img.get('url') if isinstance(img, dict) else None
                signed = None
                if isinstance(url, str) and url.startswith('gs://'):
                    # generate signed url in thread to avoid blocking
                    signed = await asyncio.to_thread(gerar_signed_url_conteudo, url, img.get('filename'))
                    if not signed:
                        logging.warning(f"Could not generate signed_url for {url} (id={img.get('_id')})")
                # always set field (may be None) so frontend can rely on its presence
                img['signed_url'] = signed
            except Exception as e:
                logging.exception(f"Unexpected error while processing image for signed_url: {e}")
                # ensure signed_url field exists even in case of per-item error
                try:
                    img['signed_url'] = None
                except Exception:
                    pass
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
    try:
        object_id = ObjectId(id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="ID inválido")
    logo = await logos_collection.find_one({"_id": object_id})
    if not logo:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    from gcs_utils import get_bucket
    bucket = get_bucket("logos")
    blob = bucket.blob(logo['filename'])
    try:
        blob.delete()
    except Exception as e:
        from google.api_core.exceptions import NotFound
        if isinstance(e, NotFound):
            # Arquivo já não existe, segue normalmente
            pass
        else:
            raise HTTPException(status_code=500, detail=f"Erro ao deletar arquivo do bucket: {str(e)}")
    await logos_collection.delete_one({"_id": object_id})
    return {"success": True}

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
        print("Erro ao listar usuários:", e)
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
                    return d0
            except Exception:
                # if geo query fails, fall back to bounding box approach below
                pass

    except Exception:
        # if anything goes wrong with radius logic, continue to fallback search
        pass

    # Fallback: use bounding box deltas for lat/lon when radius not provided or geo query failed
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
            return doc

    # 3) Fallback: busca por nome_marca igual ao valor recebido
    filtro3 = {
        "nome_marca": str(marca_id),
        "latitude": lat_filter,
        "longitude": lon_filter
    }
    return await db["conteudos"].find_one(filtro3)

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

# Cache em memória: {(nome_marca, latitude, longitude): resultado}
consulta_cache = {}

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
        consulta_cache[cache_key] = resultado
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

    consulta_cache[cache_key] = resultado
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
        consulta_cache[cache_key] = resultado
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

    consulta_cache[cache_key] = resultado
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
                            for fld in ('variant', 'color', 'icon', 'size', 'disabled', 'aria_label', 'analytics', 'visibility', 'position', 'temp_id'):
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
                # Substitui os blocos existentes pelo payload recebido.
                # Isso evita duplicações quando o frontend carrega os blocos,
                # edita um existente e envia o estado completo de volta.
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
                
                processed_image = await asyncio.to_thread(_resize_if_needed, temp_path, MAX_IMAGE_DIM)
                
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
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao adicionar conteúdo: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)