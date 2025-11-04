"""
Script para gerar GLBs retroativamente para itens do carousel que não têm GLB
"""
import os
import asyncio
import tempfile
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from PIL import Image as PILImage
from glb_generator import generate_plane_glb
from gcs_utils import upload_image_to_gcs, get_bucket, GCS_BUCKET_CONTEUDO
from main import gerar_signed_url_conteudo
from datetime import datetime

load_dotenv()

async def generate_glbs_for_carousel():
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("MONGO_DB_NAME", "olinxra")
    
    print(f"🔌 Conectando ao MongoDB Atlas...")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    try:
        # Buscar conteúdo g3
        doc = await db['conteudos'].find_one({"nome_marca": "g3"})
        
        if not doc:
            print("❌ Conteúdo g3 não encontrado")
            return
        
        print(f"✅ Conteúdo g3 encontrado: {doc.get('_id')}")
        print("="*80)
        
        blocos = doc.get('blocos', [])
        modified = False
        
        for bloco_idx, bloco in enumerate(blocos):
            tipo = bloco.get('tipo', '')
            
            # Processar carousel
            if 'items' in bloco and isinstance(bloco.get('items'), list):
                items = bloco['items']
                print(f"\n🎠 Processando carousel (Bloco {bloco_idx}) com {len(items)} itens...")
                
                for item_idx, item in enumerate(items):
                    # Verificar se item já tem GLB
                    if item.get('glb_url'):
                        print(f"   ⏭️  Item {item_idx} ({item.get('nome')}) já tem GLB, pulando...")
                        continue
                    
                    nome = item.get('nome', 'sem-nome')
                    url = item.get('url')
                    filename = item.get('filename')
                    
                    print(f"\n   🔨 Item {item_idx}: {nome}")
                    print(f"      Imagem: {filename}")
                    
                    try:
                        # 1. Baixar imagem do GCS
                        print(f"      📥 Baixando imagem...")
                        bucket = get_bucket('conteudo')
                        blob = bucket.blob(filename)
                        
                        # Download para arquivo temporário
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
                            blob.download_to_filename(temp_img.name)
                            img_path = temp_img.name
                        
                        # 2. Resize se necessário
                        print(f"      🖼️  Processando imagem...")
                        MAX_DIM = int(os.getenv('GLB_MAX_DIM', '2048'))
                        
                        img = PILImage.open(img_path)
                        w, h = img.size
                        if max(w, h) > MAX_DIM:
                            ratio = MAX_DIM / float(max(w, h))
                            new_size = (int(w * ratio), int(h * ratio))
                            img = img.convert('RGB')
                            img = img.resize(new_size, PILImage.LANCZOS)
                            resized_path = img_path + '.resized.jpg'
                            img.save(resized_path, format='JPEG', quality=90)
                            processed_img = resized_path
                        else:
                            processed_img = img_path
                        
                        # 3. Gerar GLB
                        print(f"      🔨 Gerando GLB...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as temp_glb:
                            glb_path = temp_glb.name
                        
                        generate_plane_glb(
                            processed_img,
                            glb_path,
                            base_y=0.0,
                            plane_height=1.0,
                            flip_u=False,
                            flip_v=True
                        )
                        
                        # 4. Upload GLB para GCS
                        print(f"      ☁️  Fazendo upload do GLB...")
                        owner_uid = doc.get('owner_uid', 'unknown')
                        nome_base = nome.rsplit('.', 1)[0]  # Remove extensão
                        glb_filename = f"{owner_uid}/ra/models/{nome_base}.glb"
                        
                        metadata = {
                            'generated_from_image': url,
                            'base_height': '0.0',
                            'auto_generated': 'true',
                            'generated_at': datetime.utcnow().isoformat()
                        }
                        
                        glb_gcs_url = await asyncio.to_thread(
                            upload_image_to_gcs,
                            glb_path,
                            glb_filename,
                            'conteudo',
                            'public, max-age=31536000',
                            metadata
                        )
                        
                        # 5. Gerar signed URL
                        print(f"      🔐 Gerando signed URL...")
                        glb_signed_url = await asyncio.to_thread(
                            gerar_signed_url_conteudo,
                            glb_gcs_url,
                            glb_filename,
                            365*24*60*60  # 1 ano
                        )
                        
                        # 6. Atualizar item no documento
                        print(f"      💾 Atualizando documento...")
                        item['glb_url'] = glb_gcs_url
                        item['glb_filename'] = glb_filename
                        item['glb_signed_url'] = glb_signed_url
                        item['glb_source'] = 'auto_generated_retroactive'
                        
                        modified = True
                        
                        print(f"      ✅ GLB gerado e salvo: {glb_filename}")
                        
                        # Limpar arquivos temporários
                        try:
                            os.remove(img_path)
                            if os.path.exists(processed_img) and processed_img != img_path:
                                os.remove(processed_img)
                            os.remove(glb_path)
                        except:
                            pass
                        
                    except Exception as e:
                        print(f"      ❌ Erro ao processar item {item_idx}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Salvar documento atualizado
        if modified:
            print(f"\n💾 Salvando alterações no MongoDB...")
            result = await db['conteudos'].update_one(
                {"_id": doc.get('_id')},
                {"$set": {"blocos": blocos}}
            )
            print(f"✅ Documento atualizado: {result.modified_count} documento(s)")
        else:
            print(f"\n⏭️  Nenhuma alteração necessária")
        
        print("\n" + "="*80)
        print("✅ Processo concluído!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(generate_glbs_for_carousel())
