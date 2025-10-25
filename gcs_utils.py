
import os
import json
import logging
from google.cloud import storage
from google.api_core.exceptions import NotFound, PreconditionFailed
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


def upload_image_to_gcs(local_path, filename, tipo, cache_control=None, metadata=None):
    """
    Faz o upload de um arquivo local para o Google Cloud Storage no bucket correto.

    Nota: o parâmetro `tipo` é obrigatório e deve ser 'logos' ou 'conteudo'.
    Esta função faz checagens adicionais para evitar que arquivos de conteúdo
    sejam escritos acidentalmente no bucket de logos.
    """
    if tipo not in ("logos", "conteudo"):
        raise RuntimeError(f"Parâmetro 'tipo' inválido: {tipo}. Use 'logos' ou 'conteudo'.")

    # Resolve o bucket esperado
    bucket = get_bucket(tipo)

    # Segurança adicional: se pediram 'conteudo', garanta que o bucket final
    # não seja o mesmo do logos. Caso haja inconsistência nas variáveis de
    # ambiente, abortamos com erro explícito para evitar poluição do bucket.
    if tipo == "conteudo":
        if not GCS_BUCKET_CONTEUDO:
            raise RuntimeError("GCS_BUCKET_CONTEUDO não configurado. Defina o bucket de conteúdo.")
        # Se por alguma razão bucket.name ainda aponta para o bucket de logos,
        # substituímos explicitamente pelo configurado em GCS_BUCKET_CONTEUDO.
        if bucket.name == GCS_BUCKET_LOGOS:
            if GCS_BUCKET_CONTEUDO and (GCS_BUCKET_CONTEUDO != GCS_BUCKET_LOGOS):
                bucket = storage_client.bucket(GCS_BUCKET_CONTEUDO)
            else:
                raise RuntimeError(
                    f"Bucket de conteúdo não configurado corretamente. Evitando upload em '{GCS_BUCKET_LOGOS}'. Defina GCS_BUCKET_CONTEUDO para o bucket de conteúdo (ex: 'olinxra-conteudo')."
                )

    # Log útil para debug em runtime (mas não exponha segredos)
    try:
        logging.info("[gcs_utils] Upload requested: local_path=%s filename=%s bucket=%s tipo=%s",
                     local_path, filename, bucket.name, tipo)
    except Exception:
        pass
    blob = bucket.blob(filename)
    gcs_url = f"gs://{bucket.name}/{filename}"

    # Apply cache control or metadata if provided so uploads are cacheable by CDN/clients
    try:
        if cache_control:
            try:
                blob.cache_control = cache_control
            except Exception:
                pass
        if metadata and isinstance(metadata, dict):
            try:
                blob.metadata = metadata
            except Exception:
                pass

        # Try to upload with a generation precondition to avoid race conditions
        # If another writer created the object concurrently, the precondition will
        # fail and we should treat it as a cache hit (object already exists).
        try:
            logging.info("[gcs_utils] attempting upload with if_generation_match=0 for %s", gcs_url)
            # upload_from_filename will persist blob.cache_control and blob.metadata if set above
            blob.upload_from_filename(local_path, if_generation_match=0)
            logging.info("[gcs_utils] upload successful: %s", gcs_url)
            return gcs_url
        except PreconditionFailed:
            # Another process created the blob concurrently. Treat as success.
            logging.info("[gcs_utils] PreconditionFailed: object already exists (created by another process): %s", gcs_url)
            return gcs_url
        except Exception as e:
            # For unexpected errors, log and re-raise so caller can handle.
            logging.exception("[gcs_utils] upload failed for %s: %s", gcs_url, e)
            raise
    finally:
        # Ensure we try to flush metadata changes (best-effort)
        try:
            if hasattr(blob, 'patch'):
                try:
                    blob.patch()
                except Exception:
                    pass
        except Exception:
            pass


def delete_file(filename, tipo="conteudo"):
    """
    Delete a file from the given bucket by its filename (path inside the bucket).
    Returns True if deleted or False if an error occurred.
    """
    if not filename:
        return False
    bucket = get_bucket(tipo)
    try:
        # Normalize filename: if given a gs:// path, extract the filename part
        if isinstance(filename, str) and filename.startswith('gs://'):
            # remove gs://bucketname/
            parts = filename.split('/', 3)
            if len(parts) >= 4:
                filename = parts[3]
            else:
                # fallback: last segment
                filename = filename.split('/')[-1]

        # If the provided filename looks like a "folder" (ends with '/'),
        # treat it as a prefix and delete all objects under that prefix.
        if isinstance(filename, str) and filename.endswith('/'):
            prefix = filename
            try:
                blobs = list(storage_client.list_blobs(bucket.name, prefix=prefix))
            except Exception as e:
                try:
                    print(f"[gcs_utils] Error listing blobs with prefix '{prefix}' in bucket '{bucket.name}': {e}")
                except Exception:
                    pass
                return False
            if not blobs:
                try:
                    print(f"[gcs_utils] No objects found with prefix '{prefix}' in bucket '{bucket.name}'. Nothing to delete.")
                except Exception:
                    pass
                return True
            deleted = 0
            for b in blobs:
                try:
                    b.delete()
                    deleted += 1
                except Exception as e:
                    try:
                        print(f"[gcs_utils] Failed to delete blob '{b.name}' under prefix '{prefix}': {e}")
                    except Exception:
                        pass
                    # continue deleting other blobs
            try:
                print(f"[gcs_utils] Deleted {deleted} objects with prefix '{prefix}' from bucket '{bucket.name}'")
            except Exception:
                pass
            return True

        # Normal file delete path
        blob = bucket.blob(filename)
        try:
            blob.delete()
            try:
                print(f"[gcs_utils] Deleted file '{filename}' from bucket '{bucket.name}'")
            except Exception:
                pass
            return True
        except Exception as e:
            # If the blob was not found, treat as success (idempotent behaviour)
            if isinstance(e, NotFound):
                try:
                    print(f"[gcs_utils] Object '{filename}' not found in bucket '{bucket.name}' (treated as deleted).")
                except Exception:
                    pass
                return True
            try:
                print(f"[gcs_utils] Failed to delete '{filename}' from bucket '{bucket.name}': {e}")
            except Exception:
                pass
            return False
    except Exception as e:
        try:
            print(f"[gcs_utils] Unexpected error when deleting '{filename}' from bucket '{bucket.name}': {e}")
        except Exception:
            pass
        return False


def delete_gs_path(gs_url):
    """
    Convenience: delete using a gs:// URL. Detects bucket name and delegates to delete_file.
    """
    if not gs_url or not isinstance(gs_url, str):
        return False
    if gs_url.startswith('gs://'):
        # gs://bucket-name/path/to/file
        try:
            without_prefix = gs_url[len('gs://'):]
            bucket_name, _, path = without_prefix.partition('/')
            # infer tipo from bucket name
            tipo = 'conteudo' if 'conteudo' in bucket_name else 'logos'
            return delete_file(path, tipo)
        except Exception:
            return False
    return False

