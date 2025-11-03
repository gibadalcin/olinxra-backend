#!/usr/bin/env python3
"""
Script para gerar GLBs a partir de imagens já existentes no MongoDB.

Este script lê todos os documentos de conteúdo, encontra blocos de imagens,
e gera GLBs para cada uma usando o endpoint /api/generate-glb-from-image.
Os GLBs gerados são salvos no GCS e os documentos são atualizados com glb_url.

Uso:
    python generate_glbs_from_existing_images.py [--dry-run] [--marca NOME]

Opções:
    --dry-run       Não faz modificações, apenas mostra o que seria feito
    --marca NOME    Processa apenas conteúdos da marca especificada
    --limit N       Limita o número de imagens processadas (para testes)
"""

import os
import sys
import asyncio
import argparse
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "olinxra")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI não configurado no .env")


async def get_signed_url(gs_url: str) -> str:
    """Gera signed URL para uma URL do GCS."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{BACKEND_URL}/api/conteudo-signed-url",
                params={"gs_url": gs_url}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("signed_url")
            else:
                logging.warning(f"Erro ao gerar signed URL para {gs_url}: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Exceção ao gerar signed URL: {e}")
            return None


async def generate_glb_from_image(image_gs_url: str, owner_uid: str, image_name: str) -> dict:
    """
    Chama o endpoint /api/generate-glb-from-image para gerar GLB.
    
    Args:
        image_gs_url: URL do GCS da imagem (gs://bucket/path)
        owner_uid: UID do dono do conteúdo
        image_name: Nome base da imagem (sem extensão)
    
    Returns:
        dict com glb_signed_url e gs_url, ou None se falhar
    """
    # Primeiro, gerar signed URL da imagem original
    signed_image_url = await get_signed_url(image_gs_url)
    if not signed_image_url:
        logging.error(f"Não foi possível gerar signed URL para {image_gs_url}")
        return None
    
    # Payload para gerar GLB
    payload = {
        "image_url": signed_image_url,
        "owner_uid": owner_uid,
        "filename": f"{image_name}.glb"  # backend vai adicionar o path correto
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{BACKEND_URL}/api/generate-glb-from-image",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"✅ GLB gerado: {result.get('gs_url')} (cached: {result.get('cached', False)})")
                return result
            else:
                logging.error(f"❌ Erro ao gerar GLB: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Exceção ao gerar GLB: {e}")
            return None


async def process_conteudo(doc: dict, dry_run: bool, db) -> dict:
    """
    Processa um documento de conteúdo, gerando GLBs para blocos de imagem.
    
    Returns:
        dict com estatísticas: {processed: int, updated: int, errors: int}
    """
    stats = {"processed": 0, "updated": 0, "errors": 0, "skipped": 0}
    
    doc_id = doc.get("_id")
    owner_uid = doc.get("owner_uid", "unknown")
    marca = doc.get("nome_marca") or doc.get("marca_id", "sem_marca")
    blocos = doc.get("blocos", [])
    
    if not blocos:
        logging.info(f"⏭️  Documento {doc_id} sem blocos, pulando...")
        return stats
    
    logging.info(f"📄 Processando documento: {doc_id} (marca: {marca}, blocos: {len(blocos)})")
    
    updated_blocos = []
    doc_modified = False
    
    for idx, bloco in enumerate(blocos):
        tipo = bloco.get("tipo")
        
        # Processar apenas blocos de imagem
        if tipo == "imagem":
            url = bloco.get("url")
            nome = bloco.get("nome", f"imagem_{idx}")
            glb_url_existing = bloco.get("glb_url")
            
            stats["processed"] += 1
            
            # Se já tem GLB, pular
            if glb_url_existing:
                logging.info(f"  ⏭️  Bloco {idx} já tem GLB: {glb_url_existing}")
                stats["skipped"] += 1
                updated_blocos.append(bloco)
                continue
            
            # Gerar GLB
            logging.info(f"  🔄 Bloco {idx}: gerando GLB para {nome} ({url})")
            
            if not dry_run:
                # Extrair nome base sem extensão
                import os
                name_base = os.path.splitext(nome)[0]
                
                glb_result = await generate_glb_from_image(url, owner_uid, name_base)
                
                if glb_result:
                    # Adicionar glb_url e glb_signed_url ao bloco
                    bloco["glb_url"] = glb_result.get("gs_url")
                    bloco["glb_signed_url"] = glb_result.get("glb_signed_url")
                    bloco["glb_generated_at"] = datetime.utcnow()
                    logging.info(f"  ✅ GLB adicionado ao bloco {idx}")
                    stats["updated"] += 1
                    doc_modified = True
                else:
                    logging.error(f"  ❌ Erro ao gerar GLB para bloco {idx}")
                    stats["errors"] += 1
            else:
                logging.info(f"  🔍 [DRY-RUN] Seria gerado GLB para {nome}")
            
            updated_blocos.append(bloco)
            
        elif tipo == "carousel":
            # Processar imagens dentro do carousel
            imagens = bloco.get("imagens", [])
            if imagens:
                logging.info(f"  🎠 Bloco {idx}: carousel com {len(imagens)} imagens")
                updated_imagens = []
                
                for img_idx, img in enumerate(imagens):
                    img_url = img.get("url")
                    img_nome = img.get("nome", f"carousel_{idx}_{img_idx}")
                    img_glb_existing = img.get("glb_url")
                    
                    stats["processed"] += 1
                    
                    if img_glb_existing:
                        logging.info(f"    ⏭️  Imagem {img_idx} já tem GLB")
                        stats["skipped"] += 1
                        updated_imagens.append(img)
                        continue
                    
                    logging.info(f"    🔄 Imagem {img_idx}: gerando GLB para {img_nome}")
                    
                    if not dry_run:
                        import os
                        name_base = os.path.splitext(img_nome)[0]
                        
                        glb_result = await generate_glb_from_image(img_url, owner_uid, name_base)
                        
                        if glb_result:
                            img["glb_url"] = glb_result.get("gs_url")
                            img["glb_signed_url"] = glb_result.get("glb_signed_url")
                            img["glb_generated_at"] = datetime.utcnow()
                            logging.info(f"    ✅ GLB adicionado à imagem {img_idx}")
                            stats["updated"] += 1
                            doc_modified = True
                        else:
                            logging.error(f"    ❌ Erro ao gerar GLB para imagem {img_idx}")
                            stats["errors"] += 1
                    else:
                        logging.info(f"    🔍 [DRY-RUN] Seria gerado GLB para {img_nome}")
                    
                    updated_imagens.append(img)
                
                bloco["imagens"] = updated_imagens
            
            updated_blocos.append(bloco)
        else:
            # Outros tipos de bloco (vídeo, botão, etc.)
            updated_blocos.append(bloco)
    
    # Atualizar documento no MongoDB se houve mudanças
    if doc_modified and not dry_run:
        try:
            result = await db.conteudos.update_one(
                {"_id": doc_id},
                {
                    "$set": {
                        "blocos": updated_blocos,
                        "glb_last_updated": datetime.utcnow()
                    }
                }
            )
            if result.modified_count > 0:
                logging.info(f"✅ Documento {doc_id} atualizado no MongoDB")
            else:
                logging.warning(f"⚠️  Documento {doc_id} não foi modificado no MongoDB")
        except Exception as e:
            logging.error(f"❌ Erro ao atualizar documento {doc_id}: {e}")
            stats["errors"] += 1
    
    return stats


async def main():
    parser = argparse.ArgumentParser(description="Gerar GLBs para imagens existentes")
    parser.add_argument("--dry-run", action="store_true", help="Não faz modificações, apenas simula")
    parser.add_argument("--marca", type=str, help="Processar apenas conteúdos da marca especificada")
    parser.add_argument("--limit", type=int, help="Limitar número de documentos processados")
    
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("🚀 Iniciando geração de GLBs para imagens existentes")
    logging.info("=" * 80)
    
    if args.dry_run:
        logging.info("🔍 Modo DRY-RUN ativado (nenhuma modificação será feita)")
    
    # Conectar ao MongoDB
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    # Construir query
    query = {}
    if args.marca:
        # Buscar por nome_marca OU marca_id
        query = {
            "$or": [
                {"nome_marca": args.marca},
                {"marca_id": args.marca}
            ]
        }
        logging.info(f"📌 Filtrando por marca: {args.marca}")
    
    # Contar documentos
    total_docs = await db.conteudos.count_documents(query)
    logging.info(f"📊 Total de documentos encontrados: {total_docs}")
    
    if args.limit:
        logging.info(f"⚠️  Limitado a {args.limit} documentos")
    
    # Processar documentos
    cursor = db.conteudos.find(query)
    if args.limit:
        cursor = cursor.limit(args.limit)
    
    total_stats = {"processed": 0, "updated": 0, "errors": 0, "skipped": 0}
    doc_count = 0
    
    async for doc in cursor:
        doc_count += 1
        logging.info(f"\n--- Documento {doc_count}/{total_docs if not args.limit else args.limit} ---")
        
        stats = await process_conteudo(doc, args.dry_run, db)
        
        # Acumular estatísticas
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Resumo final
    logging.info("\n" + "=" * 80)
    logging.info("📊 RESUMO FINAL")
    logging.info("=" * 80)
    logging.info(f"Documentos processados: {doc_count}")
    logging.info(f"Imagens processadas: {total_stats['processed']}")
    logging.info(f"GLBs gerados: {total_stats['updated']}")
    logging.info(f"GLBs já existentes (pulados): {total_stats['skipped']}")
    logging.info(f"Erros: {total_stats['errors']}")
    
    if args.dry_run:
        logging.info("\n🔍 DRY-RUN concluído. Execute novamente sem --dry-run para aplicar as mudanças.")
    else:
        logging.info("\n✅ Processo concluído!")
    
    # Fechar conexão
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
