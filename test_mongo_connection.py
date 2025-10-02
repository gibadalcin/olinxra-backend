import pymongoimport pymongo

import osimport os



uri = os.getenv("MONGO_URI")uri = os.getenv("MONGO_URI")

if not uri:if not uri:

    print("Variável de ambiente MONGO_URI não definida. Configure antes de rodar o teste.")    uri = "mongodb+srv://dalcin_db_user:C9KXaAHV9PfhP1Ud@olinxra-db.vulvtre.mongodb.net/?retryWrites=true&w=majority&tls=true"

    exit(1)

try:

try:    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)    print(client.list_database_names())

    print(client.list_database_names())    print("Conexão bem-sucedida!")

    print("Conexão bem-sucedida!")except Exception as e:

except Exception as e:    print("Erro de conexão:", e)

    print("Erro de conexão:", e)
