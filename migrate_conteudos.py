"""
Script de migração para a coleção `conteudos`.

Funcionalidades:
- --dry-run: não aplica alterações, apenas loga o que faria
- --apply: aplica alterações em batches
- Converte marca_id (string) para ObjectId quando possível
- Cria campo `location` GeoJSON a partir de latitude/longitude quando ausente
- Converte `updated_at` strings para datetimes
- Normaliza blocos.created_at para datetimes

Uso:
    python migrate_conteudos.py --dry-run --batch-size 500
    python migrate_conteudos.py --apply --batch-size 500

Importante: faça backup do banco antes de rodar.
"""

import argparse
import asyncio
import os
import json
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME', 'olinxra')

async def convert_doc(doc, logos_collection, dry_run=True):
    updates = {}
    _id = doc.get('_id')

    # 1) marca_id: se for string e corresponder a um logo _id, converter para ObjectId
    marca_id = doc.get('marca_id')
    if marca_id and isinstance(marca_id, str):
        try:
            possible_oid = ObjectId(marca_id)
            marca = await logos_collection.find_one({'_id': possible_oid})
            if marca:
                updates['marca_id'] = possible_oid
        except Exception:
            # não é um ObjectId válido, ignorar
            pass

    # 2) location: adicionar se latitude/longitude presentes e location ausente
    if 'location' not in doc or not doc.get('location'):
        lat = doc.get('latitude')
        lon = doc.get('longitude')
        if lat is not None and lon is not None:
            updates['location'] = { 'type': 'Point', 'coordinates': [ float(lon), float(lat) ] }

    # 3) updated_at: converter string -> datetime
    updated_at = doc.get('updated_at')
    if updated_at and isinstance(updated_at, str):
        try:
            updates['updated_at'] = datetime.fromisoformat(updated_at)
        except Exception:
            try:
                updates['updated_at'] = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                # fallback: não sobrescrever
                pass

    # 4) blocos.created_at: normalizar
    blocos = doc.get('blocos', [])
    blocos_updates = False
    new_blocos = []
    for b in blocos:
        if not isinstance(b, dict):
            new_blocos.append(b)
            continue
        created = b.get('created_at')
        if isinstance(created, str):
            try:
                dt = datetime.fromisoformat(created)
            except Exception:
                try:
                    dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S.%f")
                except Exception:
                    dt = datetime.utcnow()
            b['created_at'] = dt
            blocos_updates = True
        elif created is None:
            b['created_at'] = datetime.utcnow()
            blocos_updates = True
        new_blocos.append(b)
    if blocos_updates:
        updates['blocos'] = new_blocos

    return updates

async def run(dry_run=True, batch_size=500):
    if not MONGO_URI:
        print('MONGO_URI não configurado. Configure a variável de ambiente e tente novamente.')
        return
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    conteudos = db['conteudos']
    logos = db['logos']

    total = await conteudos.count_documents({})
    print(f'Total documentos em conteudos: {total}')
    cursor = conteudos.find({})

    processed = 0
    modified = 0

    batch = []
    async for doc in cursor:
        updates = await convert_doc(doc, logos, dry_run=dry_run)
        if updates:
            batch.append((doc['_id'], updates))
        processed += 1

        if len(batch) >= batch_size:
            print(f'Processados: {processed} — aplicando batch de {len(batch)}')
            for _id, upd in batch:
                print(f'  - {_id} -> updates: {list(upd.keys())}')
                if not dry_run:
                    await conteudos.update_one({'_id': _id}, {'$set': upd})
                    modified += 1
            batch = []

    # final batch
    if batch:
        print(f'Processados: {processed} — aplicando batch final de {len(batch)}')
        for _id, upd in batch:
            print(f'  - {_id} -> updates: {list(upd.keys())}')
            if not dry_run:
                await conteudos.update_one({'_id': _id}, {'$set': upd})
                modified += 1

    print(f'Concluído. Processados: {processed}. Modificados: {modified} (dry_run={dry_run})')
    client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migração de documentos em conteudos')
    parser.add_argument('--dry-run', action='store_true', help='Não aplica alterações, apenas loga')
    parser.add_argument('--apply', action='store_true', help='Aplica alterações (contraparte do dry-run)')
    parser.add_argument('--batch-size', type=int, default=500)
    args = parser.parse_args()

    if args.dry_run and args.apply:
        print('Escolha --dry-run ou --apply, não ambos.')
        exit(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(dry_run=not args.apply, batch_size=args.batch_size))
