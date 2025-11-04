"""
Script para listar todas as marcas e conteúdos no banco
"""
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import json

load_dotenv()

async def list_all():
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("MONGO_DB_NAME", "olinxra")
    
    if not MONGO_URI:
        print("❌ MONGO_URI não encontrado no .env")
        return
    
    print(f"🔌 Conectando ao MongoDB: {DB_NAME}")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    try:
        # Listar todas as marcas
        marcas = await db['logos'].find({}).to_list(length=None)
        print(f"\n📋 MARCAS CADASTRADAS ({len(marcas)}):")
        print("="*80)
        for marca in marcas:
            print(f"   • {marca.get('nome', 'SEM NOME')} (ID: {marca.get('_id')})")
            print(f"     Owner: {marca.get('owner_uid', 'N/A')}")
        
        # Listar todos os conteúdos
        conteudos = await db['conteudos'].find({}).to_list(length=None)
        print(f"\n📦 CONTEÚDOS CADASTRADOS ({len(conteudos)}):")
        print("="*80)
        for idx, doc in enumerate(conteudos, 1):
            print(f"\n{idx}. Conteúdo ID: {doc.get('_id')}")
            print(f"   marca_id: {doc.get('marca_id', 'N/A')}")
            print(f"   nome_marca: {doc.get('nome_marca', 'N/A')}")
            print(f"   owner_uid: {doc.get('owner_uid', 'N/A')}")
            print(f"   tipo_regiao: {doc.get('tipo_regiao', 'N/A')}")
            print(f"   nome_regiao: {doc.get('nome_regiao', 'N/A')}")
            print(f"   Total de blocos: {len(doc.get('blocos', []))}")
            
            # Mostrar resumo dos blocos
            blocos = doc.get('blocos', [])
            for bloco_idx, bloco in enumerate(blocos):
                tipo = bloco.get('tipo', 'sem tipo')
                has_items = 'items' in bloco and isinstance(bloco.get('items'), list)
                num_items = len(bloco.get('items', [])) if has_items else 0
                has_glb = 'glb_url' in bloco
                
                icon = "🎠" if has_items else "📦"
                glb_icon = "✅" if has_glb else "❌"
                
                print(f"   {icon} Bloco {bloco_idx}: {tipo} (GLB: {glb_icon})")
                if has_items:
                    print(f"      └─ {num_items} itens no carousel")
                    # Verificar GLBs nos items
                    items_with_glb = sum(1 for item in bloco.get('items', []) if 'glb_url' in item)
                    print(f"      └─ Items com GLB: {items_with_glb}/{num_items}")
        
        print("\n" + "="*80)
        print("\n✅ Listagem concluída!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(list_all())
