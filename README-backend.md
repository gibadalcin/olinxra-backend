olinxra-backend — instruções rápidas

Este diretório contém a API backend (FastAPI) usada pelo projeto OlinxRA.

1) Variáveis de ambiente
- Não armazene segredos em arquivos de projeto no repositório. Use um secret manager do provedor (DigitalOcean App Platform, AWS Secrets Manager, etc.).
- Há um arquivo de exemplo `.env.example` com as chaves necessárias. Copie-o para `.env` apenas em ambientes de desenvolvimento locais.

2) Segurança
- NÃO commite `.env` nem arquivos com credenciais (ex.: `cloud-storage-cred.json`, `firebase-cred.json`). O `.gitignore` já inclui esses nomes.
- Para deploy em produção, utilize as configurações de ambiente do provedor e garanta acesso mínimo às credenciais (principle of least privilege).

3) Reinício / Deploy
- Se estiver usando uma App Platform, um push/merge para a branch configurada desencadeará o deploy automático.
- Em uma VM com systemd, reinicie o serviço que executa o Uvicorn (ex.: `sudo systemctl restart olinxra-backend`).

4) Testes rápidos
- Verifique endpoints:
  - `/debug/logos` — verifica a coleção de logos e o índice FAISS
  - `/images` — lista imagens (agora retorna `signed_url` para arquivos GCS)

5) Contato
- Para operações sensíveis (rotacionar chaves, atualizar credenciais), coordene com o time de ops.
