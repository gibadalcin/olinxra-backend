import pymongo
import os

URI = os.getenv("MONGO_URI")

if URI:
    uri = URI  # O valor já deve ser o URI completo, não precisa adicionar 'mongodb+srv://'
else:
    print("Variável de ambiente MONGO_URI não definida.")
    uri = None

if uri:
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        print(client.list_database_names())
        print("Conexão bem-sucedida!")
    except Exception as e:
        print("Erro de conexão:", e)
else:
    print("URI de conexão não definida.")
