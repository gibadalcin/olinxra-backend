#!/usr/bin/env python3
"""Lista a estrutura de arquivos no GCS para entender onde estão os GLBs."""
from google.cloud import storage
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud-storage-cred.json'

client = storage.Client()
bucket = client.bucket('olinxra-conteudo')

print("🔍 Listando estrutura do GCS bucket: olinxra-conteudo")
print("=" * 80)

# Listar todos os arquivos
blobs = list(bucket.list_blobs(prefix='TR77xSOJrigOHfkoYQtx1iim6ok1/'))

# Agrupar por tipo
images = []
glbs = []
outros = []

for blob in blobs:
    if blob.name.endswith('.png') or blob.name.endswith('.jpg'):
        images.append(blob.name)
    elif blob.name.endswith('.glb'):
        glbs.append(blob.name)
    else:
        outros.append(blob.name)

print(f"\n📁 Total de arquivos: {len(blobs)}")
print(f"   🖼️  Imagens (.png/.jpg): {len(images)}")
print(f"   🎮 GLBs (.glb): {len(glbs)}")
print(f"   📄 Outros: {len(outros)}\n")

print("🎮 GLBs encontrados:")
print("-" * 80)
for glb in sorted(glbs):
    # Extrair nome do arquivo
    filename = glb.split('/')[-1]
    print(f"   {filename:30s} → {glb}")

print("\n🖼️  Imagens encontradas:")
print("-" * 80)
for img in sorted(images):
    filename = img.split('/')[-1]
    # Verificar se tem GLB correspondente
    glb_name = filename.replace('.png', '.glb').replace('.jpg', '.glb')
    has_glb = any(glb_name in g for g in glbs)
    status = "✅" if has_glb else "❌"
    print(f"   {status} {filename:30s} → {img}")

print("\n🗂️  Estrutura de pastas:")
print("-" * 80)
folders = set()
for blob in blobs:
    parts = blob.name.split('/')
    for i in range(1, len(parts)):
        folder = '/'.join(parts[:i])
        folders.add(folder)

for folder in sorted(folders):
    level = folder.count('/')
    indent = "   " * level
    name = folder.split('/')[-1]
    print(f"{indent}📁 {name}/")
