
import os
import json
from google.cloud import storage
from google.api_core.exceptions import NotFound
import os
import json
import logging
import mimetypes
from google.cloud import storage
from google.api_core.exceptions import NotFound
from dotenv import load_dotenv

load_dotenv()

# Variáveis de ambiente para múltiplos buckets
GCS_BUCKET_LOGOS = os.getenv("GCS_BUCKET_LOGOS")
GCS_BUCKET_CONTEUDO = os.getenv("GCS_BUCKET_CONTEUDO")
GCS_CREDENTIALS_JSON = os.getenv("CLOUD_STORAGE_CRED_JSON")

if not GCS_CREDENTIALS_JSON:
    raise RuntimeError("Variável de ambiente CLOUD_STORAGE_CRED_JSON não encontrada.")

if GCS_CREDENTIALS_JSON.startswith('"') and GCS_CREDENTIALS_JSON.endswith('"'):
    GCS_CREDENTIALS_JSON = GCS_CREDENTIALS_JSON[1:-1]
GCS_CREDENTIALS_JSON = GCS_CREDENTIALS_JSON.replace('\\"', '"')

try:
    cred_dict = json.loads(GCS_CREDENTIALS_JSON)
    if 'private_key' in cred_dict:
        cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
    storage_client = storage.Client.from_service_account_info(cred_dict)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Erro ao decodificar JSON das credenciais do GCS: {e}")


def get_bucket(tipo="logos"):
    if tipo == "conteudo":
        bucket_name = GCS_BUCKET_CONTEUDO
    else:
        bucket_name = GCS_BUCKET_LOGOS
    if not bucket_name:
        raise RuntimeError(f"Bucket para tipo '{tipo}' não configurado.")
    return storage_client.bucket(bucket_name)


def upload_image_to_gcs(local_path, filename, tipo):
    """
    Faz o upload de um arquivo local para o Google Cloud Storage no bucket correto.

    Parâmetros:
    - local_path: caminho local do arquivo a enviar
    - filename: path dentro do bucket (por exemplo 'owner_uid/ra/file.glb')
    - tipo: 'logos' ou 'conteudo'
    Retorna: gs://bucket-name/path
    """
    if tipo not in ("logos", "conteudo"):
        raise RuntimeError(f"Parâmetro 'tipo' inválido: {tipo}. Use 'logos' ou 'conteudo'.")

    bucket = get_bucket(tipo)

    # Proteção: assegura que estamos a usar o bucket de conteúdo quando tipo=='conteudo'
    if tipo == "conteudo":
        if not GCS_BUCKET_CONTEUDO:
            raise RuntimeError("GCS_BUCKET_CONTEUDO não configurado. Defina o bucket de conteúdo.")
        if bucket.name == GCS_BUCKET_LOGOS and GCS_BUCKET_CONTEUDO and (GCS_BUCKET_CONTEUDO != GCS_BUCKET_LOGOS):
            bucket = storage_client.bucket(GCS_BUCKET_CONTEUDO)

    logging.debug(f"[gcs_utils] Uploading file '{filename}' to bucket '{bucket.name}' (tipo={tipo})")

    blob = bucket.blob(filename)
    try:
        content_type, _ = mimetypes.guess_type(filename)
        if filename.lower().endswith('.glb'):
            content_type = 'model/gltf-binary'
        if content_type:
            blob.content_type = content_type
    except Exception:
        # ignore content-type guessing failures
        pass

    blob.upload_from_filename(local_path)
    gcs_url = f"gs://{bucket.name}/{filename}"
    return gcs_url


def delete_file(filename, tipo="conteudo"):
    """Delete a file or prefix from the bucket. If filename ends with '/', treats as prefix."""
    if not filename:
        return False
    bucket = get_bucket(tipo)

    # Normalize gs:// URL
    if isinstance(filename, str) and filename.startswith('gs://'):
        parts = filename.split('/', 3)
        if len(parts) >= 4:
            filename = parts[3]
        else:
            filename = filename.split('/')[-1]

    try:
        if isinstance(filename, str) and filename.endswith('/'):
            # delete all blobs with prefix
            blobs = list(storage_client.list_blobs(bucket.name, prefix=filename))
            if not blobs:
                logging.debug(f"[gcs_utils] No objects found with prefix '{filename}' in bucket '{bucket.name}'")
                return True
            deleted = 0
            for b in blobs:
                try:
                    b.delete()
                    deleted += 1
                except Exception as e:
                    logging.warning(f"[gcs_utils] Failed to delete blob '{b.name}' under prefix '{filename}': {e}")
            logging.debug(f"[gcs_utils] Deleted {deleted} objects with prefix '{filename}' from bucket '{bucket.name}'")
            return True

        blob = bucket.blob(filename)
        try:
            blob.delete()
            logging.debug(f"[gcs_utils] Deleted file '{filename}' from bucket '{bucket.name}'")
            return True
        except Exception as e:
            if isinstance(e, NotFound):
                logging.debug(f"[gcs_utils] Object '{filename}' not found in bucket '{bucket.name}' (treated as deleted).")
                return True
            logging.warning(f"[gcs_utils] Failed to delete '{filename}' from bucket '{bucket.name}': {e}")
            return False
    except Exception as e:
        logging.warning(f"[gcs_utils] Unexpected error when deleting '{filename}' from bucket '{bucket.name}': {e}")
        return False


def delete_gs_path(gs_url):
    """Convenience: delete object by gs:// URL."""
    if not gs_url or not isinstance(gs_url, str):
        return False
    if gs_url.startswith('gs://'):
        try:
            without_prefix = gs_url[len('gs://'):]
            bucket_name, _, path = without_prefix.partition('/')
            tipo = 'conteudo' if 'conteudo' in bucket_name else 'logos'
            return delete_file(path, tipo)
        except Exception:
            return False
    return False
    

