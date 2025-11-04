#!/usr/bin/env python3
"""
Script para gerar GLBs faltantes em blocos de imagem existentes.
Processa blocos que têm 'url' mas não têm 'glb_url'.

Uso:
    python tools/generate_missing_glbs.py [--marca MARCA] [--dry-run]

Argumentos:
    --marca: Nome da marca para processar (ex: "g3"). Se omitido, processa todas.
    --dry-run: Apenas simula, sem salvar alterações no banco.

Exemplos:
    python tools/generate_missing_glbs.py --marca g3
    python tools/generate_missing_glbs.py --dry-run
"""

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path

# Adicionar diretório pai ao path para importar módulos do backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from dotenv import load_dotenv
from glb_generator import generate_plane_glb
import tempfile
from PIL import Image as PILImage
import requests
from gcs_utils import upload_image_to_gcs

# Importar apenas funções específicas para evitar erro de dependências
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar gerar_signed_url_conteudo diretamente do código
from google.cloud import storage
from datetime import timedelta

def gerar_signed_url_conteudo(gcs_url: str, filename: str = None, expiration: int = 7*24*60*60) -> str:
    """Gera signed URL para conteúdo no GCS."""
    try:
        if not gcs_url or not isinstance(gcs_url, str):
            return None
        
        # Extrair bucket e blob_name
        if gcs_url.startswith('gs://'):
            parts = gcs_url.replace('gs://', '').split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''
        elif gcs_url.startswith('https://storage.googleapis.com/'):
            parts = gcs_url.replace('https://storage.googleapis.com/', '').split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''
        else:
            return None
        
        # Conectar ao GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Gerar signed URL
        url = blob.generate_signed_url(
            version='v4',
            expiration=timedelta(seconds=expiration),
            method='GET'
        )
        
        return url
    except Exception as e:
        logging.error(f"Erro ao gerar signed URL: {e}")
        return None

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Carregar variáveis de ambiente
load_dotenv()

# Configurar credenciais do GCS (relativo ao diretório do backend)
backend_dir = Path(__file__).parent.parent
GCS_CRED_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not GCS_CRED_PATH or not os.path.exists(GCS_CRED_PATH):
    # Tentar path relativo ao backend
    GCS_CRED_PATH = str(backend_dir / 'cloud-storage-cred.json')

if not os.path.exists(GCS_CRED_PATH):
    logging.error(f"Arquivo de credenciais GCS não encontrado: {GCS_CRED_PATH}")
    logging.error(f"Configure GOOGLE_APPLICATION_CREDENTIALS no .env ou coloque cloud-storage-cred.json no diretório backend")
    sys.exit(1)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCS_CRED_PATH
logging.info(f"✅ Credenciais GCS carregadas: {GCS_CRED_PATH}")

# Conectar ao MongoDB
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    logging.error("MONGO_URI não configurada no .env")
    sys.exit(1)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client['olinxra']
conteudos_col = db['conteudos']

# Configurações GLB
MAX_IMAGE_DIM = int(os.getenv('GLB_MAX_DIM', '2048'))
GCS_BUCKET_CONTEUDO = os.getenv('GCS_BUCKET_CONTEUDO', 'olinxra-conteudo')


def download_image(url: str, dest_path: str) -> bool:
    """Download imagem de URL pública (signed URL)."""
    try:
        logging.info(f"📥 Baixando imagem: {url[:80]}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        
        logging.info(f"✅ Download concluído: {os.path.getsize(dest_path)} bytes")
        return True
    except Exception as e:
        logging.error(f"❌ Erro ao baixar imagem: {e}")
        return False


def resize_if_needed(src_path: str, max_dim: int) -> str:
    """Redimensiona imagem se exceder dimensão máxima."""
    try:
        img = PILImage.open(src_path)
        w, h = img.size
        
        if max(w, h) > max_dim:
            logging.info(f"📐 Redimensionando de {w}x{h} para max={max_dim}")
            ratio = max_dim / float(max(w, h))
            new_size = (int(w * ratio), int(h * ratio))
            img = img.convert('RGB')
            img = img.resize(new_size, PILImage.LANCZOS)
            
            dst = src_path + '.resized.jpg'
            img.save(dst, format='JPEG', quality=90)
            logging.info(f"✅ Imagem redimensionada: {new_size[0]}x{new_size[1]}")
            return dst
        
        return src_path
    except Exception as e:
        logging.error(f"❌ Erro ao redimensionar: {e}")
        return src_path


async def generate_glb_for_block(bloco: dict, marca: str, dry_run: bool = False) -> dict | None:
    """Gera GLB para um bloco de imagem."""
    try:
        # Extrair informações do bloco
        image_url = bloco.get('url')
        signed_url = bloco.get('signed_url')
        filename = bloco.get('filename')
        tipo = bloco.get('tipo', 'imagem')
        
        if not image_url:
            logging.warning(f"⏭️ Bloco sem URL de imagem, pulando: {tipo}")
            return None
        
        # Gerar signed URL se não tiver (para download)
        if not signed_url:
            logging.info(f"🔑 Gerando signed URL para download...")
            signed_url = await asyncio.to_thread(
                gerar_signed_url_conteudo,
                image_url,
                filename,
                7*24*60*60  # 7 dias
            )
        
        if not signed_url:
            logging.error(f"❌ Não foi possível obter signed URL")
            return None
        
        # Criar diretório temporário para processamento
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Download da imagem
            image_temp = os.path.join(temp_dir, 'original.jpg')
            if not download_image(signed_url, image_temp):
                return None
            
            # 2. Redimensionar se necessário
            logging.info(f"📐 Verificando dimensões...")
            processed_image = resize_if_needed(image_temp, MAX_IMAGE_DIM)
            
            # 3. Gerar GLB
            logging.info(f"🎨 Gerando modelo 3D GLB...")
            glb_temp = os.path.join(temp_dir, 'model.glb')
            
            await asyncio.to_thread(
                generate_plane_glb,
                processed_image,
                glb_temp,
                base_y=0.0,
                plane_height=1.0,
                flip_u=False,
                flip_v=True
            )
            
            if not os.path.exists(glb_temp):
                logging.error(f"❌ GLB não foi gerado")
                return None
            
            glb_size = os.path.getsize(glb_temp)
            logging.info(f"✅ GLB gerado: {glb_size} bytes")
            
            if dry_run:
                logging.info(f"🏷️ DRY-RUN: Não fazendo upload do GLB")
                return {
                    'glb_url': f'gs://{GCS_BUCKET_CONTEUDO}/dry-run-{marca}.glb',
                    'glb_filename': f'dry-run-{marca}.glb',
                    'glb_source': 'auto_generated',
                    'size': glb_size
                }
            
            # 4. Upload para GCS
            logging.info(f"☁️ Fazendo upload do GLB para GCS...")
            
            # Extrair nome base do filename original
            if filename:
                name_base = os.path.splitext(os.path.basename(filename))[0]
            else:
                name_base = f"{marca}_{tipo.lower().replace(' ', '_')}"
            
            glb_filename = f"marcas/{marca}/ra/models/{name_base}.glb"
            
            metadata = {
                'generated_from_image': image_url,
                'base_height': '0.0',
                'auto_generated': 'true',
                'retroactive_generation': 'true',
                'original_block_type': tipo
            }
            
            glb_gcs_url = await asyncio.to_thread(
                upload_image_to_gcs,
                glb_temp,
                glb_filename,
                'conteudo',
                'public, max-age=31536000',
                metadata
            )
            
            logging.info(f"✅ Upload concluído: {glb_gcs_url}")
            
            # 5. Gerar signed URL
            logging.info(f"🔑 Gerando signed URL do GLB...")
            glb_signed_url = await asyncio.to_thread(
                gerar_signed_url_conteudo,
                glb_gcs_url,
                glb_filename,
                7*24*60*60  # 7 dias
            )
            
            return {
                'glb_url': glb_gcs_url,
                'glb_filename': glb_filename,
                'glb_signed_url': glb_signed_url,
                'glb_source': 'auto_generated',
                'size': glb_size
            }
            
    except Exception as e:
        logging.exception(f"❌ Erro ao gerar GLB: {e}")
        return None


async def process_marca(marca: str, dry_run: bool = False):
    """Processa todos os blocos de uma marca."""
    logging.info(f"\n{'='*60}")
    logging.info(f"🎯 Processando marca: {marca}")
    logging.info(f"{'='*60}\n")
    
    # Buscar conteúdo da marca
    conteudo_doc = conteudos_col.find_one({'nome_marca': marca})
    
    if not conteudo_doc:
        logging.warning(f"⚠️ Marca '{marca}' não encontrada no banco")
        return
    
    blocos = conteudo_doc.get('blocos', [])
    if not blocos:
        logging.warning(f"⚠️ Marca '{marca}' não tem blocos")
        return
    
    logging.info(f"📦 Total de blocos: {len(blocos)}")
    
    # Processar cada bloco
    blocos_processados = 0
    blocos_atualizados = 0
    blocos_com_glb = 0
    blocos_sem_imagem = 0
    
    for idx, bloco in enumerate(blocos):
        tipo = bloco.get('tipo', 'sem tipo')
        logging.info(f"\n[{idx+1}/{len(blocos)}] 📋 Processando: {tipo}")
        
        # Verificar se já tem GLB
        if bloco.get('glb_url'):
            logging.info(f"✓ Bloco já tem GLB, pulando...")
            blocos_com_glb += 1
            continue
        
        # Verificar se tem URL de imagem
        if not bloco.get('url'):
            logging.info(f"⏭️ Bloco sem URL de imagem, pulando...")
            blocos_sem_imagem += 1
            continue
        
        # Processar items de carousel (se existir)
        if bloco.get('items') and isinstance(bloco.get('items'), list):
            logging.info(f"🎠 Bloco tipo carousel com {len(bloco['items'])} items")
            for item_idx, item in enumerate(bloco['items']):
                if item.get('glb_url'):
                    logging.info(f"  [{item_idx+1}] ✓ Item já tem GLB: {item.get('nome', 'sem nome')}")
                    continue
                
                if not item.get('url'):
                    logging.info(f"  [{item_idx+1}] ⏭️ Item sem URL: {item.get('nome', 'sem nome')}")
                    continue
                
                logging.info(f"  [{item_idx+1}] 🔨 Gerando GLB para item: {item.get('nome', 'sem nome')}")
                
                glb_data = await generate_glb_for_block(item, marca, dry_run)
                
                if glb_data:
                    logging.info(f"  [{item_idx+1}] ✅ GLB gerado: {glb_data['size']} bytes")
                    
                    if not dry_run:
                        # Atualizar item no banco
                        item['glb_url'] = glb_data['glb_url']
                        item['glb_filename'] = glb_data['glb_filename']
                        item['glb_signed_url'] = glb_data['glb_signed_url']
                        item['glb_source'] = glb_data['glb_source']
                        blocos_atualizados += 1
                else:
                    logging.error(f"  [{item_idx+1}] ❌ Falha ao gerar GLB")
            
            blocos_processados += 1
            continue
        
        # Gerar GLB para bloco simples
        logging.info(f"🔨 Gerando GLB para bloco...")
        glb_data = await generate_glb_for_block(bloco, marca, dry_run)
        
        if glb_data:
            logging.info(f"✅ GLB gerado: {glb_data['size']} bytes")
            
            if not dry_run:
                # Atualizar bloco no banco
                bloco['glb_url'] = glb_data['glb_url']
                bloco['glb_filename'] = glb_data['glb_filename']
                bloco['glb_signed_url'] = glb_data['glb_signed_url']
                bloco['glb_source'] = glb_data['glb_source']
                blocos_atualizados += 1
        else:
            logging.error(f"❌ Falha ao gerar GLB")
        
        blocos_processados += 1
    
    # Salvar alterações no banco
    if blocos_atualizados > 0 and not dry_run:
        logging.info(f"\n💾 Salvando alterações no banco...")
        conteudos_col.update_one(
            {'_id': conteudo_doc['_id']},
            {'$set': {'blocos': blocos}}
        )
        logging.info(f"✅ Banco atualizado com sucesso")
    
    # Resumo
    logging.info(f"\n{'='*60}")
    logging.info(f"📊 RESUMO - {marca}")
    logging.info(f"{'='*60}")
    logging.info(f"Total de blocos: {len(blocos)}")
    logging.info(f"Blocos processados: {blocos_processados}")
    logging.info(f"Blocos já tinham GLB: {blocos_com_glb}")
    logging.info(f"Blocos sem imagem: {blocos_sem_imagem}")
    logging.info(f"Blocos atualizados: {blocos_atualizados}")
    logging.info(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description='Gera GLBs faltantes em blocos de imagem')
    parser.add_argument('--marca', type=str, help='Nome da marca (ex: g3)')
    parser.add_argument('--dry-run', action='store_true', help='Simula sem salvar')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logging.info("🏷️ MODO DRY-RUN ATIVADO - Nenhuma alteração será salva")
    
    if args.marca:
        # Processar marca específica
        await process_marca(args.marca, args.dry_run)
    else:
        # Processar todas as marcas
        logging.info("🔍 Buscando todas as marcas...")
        marcas = conteudos_col.distinct('nome_marca')
        logging.info(f"📌 Marcas encontradas: {len(marcas)}")
        
        for marca in marcas:
            await process_marca(marca, args.dry_run)


if __name__ == '__main__':
    asyncio.run(main())
