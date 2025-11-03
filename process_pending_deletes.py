"""
Worker simple para processar a coleção pending_deletes.
Uso: python process_pending_deletes.py
"""
import os
import asyncio
import json
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from gcs_utils import delete_gs_path, delete_file

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME', 'olinxra')
GCS_BUCKET_CONTEUDO = os.getenv('GCS_BUCKET_CONTEUDO', 'olinxra-conteudo')

logging.basicConfig(level=logging.INFO)

def get_glb_path_from_image_url(image_url):
    """
    Deriva o path do GLB a partir de uma URL de imagem.
    
    Exemplo:
        gs://bucket/TR77xSOJ.../totem_header.jpg 
        → gs://bucket/TR77xSOJ.../ra/models/totem_header.glb
    """
    try:
        if not image_url or not isinstance(image_url, str):
            return None
        
        if not image_url.startswith('gs://'):
            return None
        
        # Parse: gs://bucket/owner_uid/image.jpg
        parts = image_url.split('/', 3)
        if len(parts) < 4:
            return None
        
        bucket = parts[2]
        path = parts[3]
        
        # Extrair owner_uid e filename
        path_parts = path.split('/', 1)
        if len(path_parts) < 2:
            filename = path.split('/')[-1]
            owner_uid = None
        else:
            owner_uid = path_parts[0]
            filename = path_parts[1].split('/')[-1]
        
        # Remover extensão e adicionar .glb
        name_without_ext = filename.rsplit('.', 1)[0]
        glb_filename = f"{name_without_ext}.glb"
        
        # Construir path do GLB
        if owner_uid:
            glb_path = f"{owner_uid}/ra/models/{glb_filename}"
        else:
            glb_path = f"public/ra/models/{glb_filename}"
        
        glb_url = f"gs://{bucket}/{glb_path}"
        return glb_url
    except Exception as e:
        logging.error(f"Erro ao derivar GLB path de {image_url}: {e}")
        return None

async def delete_image_and_glb(item):
    """
    Deleta uma imagem e seu GLB associado do GCS.
    """
    deleted_image = False
    deleted_glb = False
    
    try:
        image_url = item.get('gs_url')
        image_filename = item.get('filename')
        
        if image_url:
            glb_url = get_glb_path_from_image_url(image_url)
            
            # Deletar imagem
            deleted_image = delete_gs_path(image_url)
            logging.info(f"Imagem deletada: {image_url} (sucesso: {deleted_image})")
            
            # Deletar GLB associado
            if glb_url:
                try:
                    deleted_glb = delete_gs_path(glb_url)
                    logging.info(f"GLB deletado: {glb_url} (sucesso: {deleted_glb})")
                except Exception as e:
                    logging.warning(f"Erro ao deletar GLB {glb_url}: {e}")
        
        elif image_filename:
            image_url = f"gs://{GCS_BUCKET_CONTEUDO}/{image_filename}"
            glb_url = get_glb_path_from_image_url(image_url)
            
            # Deletar imagem
            deleted_image = delete_file(image_filename, item.get('tipo', 'conteudo'))
            logging.info(f"Imagem deletada: {image_filename} (sucesso: {deleted_image})")
            
            # Deletar GLB associado
            if glb_url:
                try:
                    deleted_glb = delete_gs_path(glb_url)
                    logging.info(f"GLB deletado: {glb_url} (sucesso: {deleted_glb})")
                except Exception as e:
                    logging.warning(f"Erro ao deletar GLB {glb_url}: {e}")
        
        return deleted_image
    
    except Exception as e:
        logging.error(f"Erro ao deletar imagem e GLB: {e}")
        return False

async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    pending = await db['pending_deletes'].find({'status': {'$in': ['pending', 'retry']}}).to_list(length=1000)
    
    for p in pending:
        try:
            item = {
                'gs_url': p.get('gs_url'),
                'filename': p.get('filename'),
                'tipo': p.get('tipo', 'conteudo')
            }
            
            # 🆕 Deletar imagem E GLB associado
            ok = await delete_image_and_glb(item)
            
            if ok:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                print(f"✅ Deleted pending {p.get('_id')}")
            else:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'retry', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
                print(f"⚠️  Retry pending {p.get('_id')}")
        except Exception as e:
            try:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'error', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
            except Exception:
                pass
            print(f"❌ Error processing {p.get('_id')}: {e}")
    
    client.close()

if __name__ == '__main__':
    asyncio.run(main())
