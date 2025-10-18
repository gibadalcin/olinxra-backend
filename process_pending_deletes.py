"""
Worker simple para processar a coleção pending_deletes.
Uso: python process_pending_deletes.py
"""
import os
import asyncio
import json
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from gcs_utils import delete_gs_path, delete_file

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME', 'olinxra')

async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    pending = await db['pending_deletes'].find({'status': {'$in': ['pending', 'retry']}}).to_list(length=1000)
    for p in pending:
        try:
            ok = False
            if p.get('gs_url'):
                ok = delete_gs_path(p.get('gs_url'))
            elif p.get('filename'):
                ok = delete_file(p.get('filename'), p.get('tipo', 'conteudo'))
            if ok:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'done', 'last_attempt': datetime.utcnow()}})
                print(f"Deleted pending {p.get('_id')}")
            else:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'retry', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
                print(f"Retry pending {p.get('_id')}")
        except Exception as e:
            try:
                await db['pending_deletes'].update_one({'_id': p['_id']}, {'$set': {'status': 'error', 'last_attempt': datetime.utcnow()}, '$inc': {'retries': 1}})
            except Exception:
                pass
            print(f"Error processing {p.get('_id')}: {e}")
    client.close()

if __name__ == '__main__':
    asyncio.run(main())
