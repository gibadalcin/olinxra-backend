
import os
import json
from google.cloud import storage
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
        print(f"[gcs_utils] Uploading file '{filename}' to bucket '{bucket.name}' (tipo={tipo})")
    except Exception:
        pass

    blob = bucket.blob(filename)
    blob.upload_from_filename(local_path)
    gcs_url = f"gs://{bucket.name}/{filename}"
    return gcs_url


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
        blob = bucket.blob(filename)
        blob.delete()
        try:
            print(f"[gcs_utils] Deleted file '{filename}' from bucket '{bucket.name}'")
        except Exception:
            pass
        return True
    except Exception as e:
        try:
            print(f"[gcs_utils] Failed to delete '{filename}' from bucket '{bucket.name}': {e}")
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

