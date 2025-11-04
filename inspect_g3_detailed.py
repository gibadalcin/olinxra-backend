"""
Script para verificar detalhes do conteúdo g3 e adicionar GLBs manualmente se necessário
"""
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import json

load_dotenv()

async def inspect_g3_content():
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
        
        print(f"✅ Conteúdo g3 encontrado!")
        print(f"   _id: {doc.get('_id')}")
        print("="*80)
        
        blocos = doc.get('blocos', [])
        
        for idx, bloco in enumerate(blocos):
            print(f"\n📦 BLOCO {idx}: {bloco.get('tipo', 'sem tipo')}")
            print(f"   Subtipo: {bloco.get('subtipo', 'N/A')}")
            
            # Se é carousel, mostrar items
            if 'items' in bloco and isinstance(bloco.get('items'), list):
                items = bloco['items']
                print(f"\n   🎠 CAROUSEL com {len(items)} itens:")
                
                for item_idx, item in enumerate(items):
                    print(f"\n   Item {item_idx}:")
                    print(f"      nome: {item.get('nome', 'N/A')}")
                    print(f"      url: {item.get('url', 'N/A')}")
                    print(f"      filename: {item.get('filename', 'N/A')}")
                    print(f"      type: {item.get('type', 'N/A')}")
                    print(f"      glb_url: {item.get('glb_url', '❌ NÃO TEM')}")
                    print(f"      glb_signed_url: {item.get('glb_signed_url', '❌ NÃO TEM')}")
                    print(f"      glb_filename: {item.get('glb_filename', '❌ NÃO TEM')}")
                    print(f"      glb_source: {item.get('glb_source', 'N/A')}")
                    
                    # Estrutura completa
                    print(f"\n      📋 Estrutura completa:")
                    print(json.dumps(item, indent=8, default=str))
            else:
                # Bloco simples
                print(f"   url: {bloco.get('url', 'N/A')}")
                print(f"   filename: {bloco.get('filename', 'N/A')}")
                print(f"   glb_url: {bloco.get('glb_url', '❌ NÃO TEM')}")
                print(f"   glb_signed_url: {bloco.get('glb_signed_url', '❌ NÃO TEM')}")
        
        print("\n" + "="*80)
        print("\n💡 PRÓXIMO PASSO:")
        print("   Os GLBs precisam ser gerados para cada item do carousel.")
        print("   Opções:")
        print("   1. Recriar o conteúdo no AdminUI (vai gerar GLBs automaticamente)")
        print("   2. Executar script para gerar GLBs retroativamente")
        print("   3. Adicionar GLBs manualmente ao documento (se já existirem no GCS)")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(inspect_g3_content())
