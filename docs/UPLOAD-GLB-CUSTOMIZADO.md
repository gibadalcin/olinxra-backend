# 🎨 Upload de GLB Customizado - Documentação

## 📋 Visão Geral

O endpoint `/add-content-image/` agora suporta upload opcional de modelos GLB customizados para cada imagem.

## 🔄 Fluxo de Funcionamento

### Cenário 1: Apenas Imagem (Comportamento Original)
```
Usuário → Upload Imagem (PNG/JPG/SVG)
Backend → Gera GLB automaticamente
Resposta → { glb_url, glb_signed_url, glb_source: 'auto_generated' }
```

### Cenário 2: Imagem + GLB Customizado (NOVO)
```
Usuário → Upload Imagem + GLB customizado
Backend → Usa GLB fornecido (sem geração)
Resposta → { glb_url, glb_signed_url, glb_source: 'custom' }
```

### Cenário 3: Falha no GLB Customizado (Fallback)
```
Usuário → Upload Imagem + GLB inválido
Backend → Detecta erro → Gera GLB automaticamente
Resposta → { glb_url, glb_signed_url, glb_source: 'auto_generated' }
```

## 🔧 Endpoint Atualizado

### POST `/add-content-image/`

**Parâmetros (Form Data):**

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `file` | File | ✅ Sim | Imagem (PNG/JPG/SVG/etc) |
| `glb_file` | File | ❌ Não | **NOVO**: Modelo GLB customizado |
| `name` | String | ✅ Sim | Nome do arquivo |
| `temp_id` | String | ❌ Não | ID temporário para tracking |
| `tipo_bloco` | String | ❌ Não | Tipo do bloco (padrão: "imagem") |
| `subtipo` | String | ❌ Não | Subtipo do bloco |
| `marca` | String | ❌ Não | Marca associada |
| `tipo_regiao` | String | ❌ Não | Tipo de região |
| `nome_regiao` | String | ❌ Não | Nome da região |

**Response:**

```json
{
  "success": true,
  "url": "gs://bucket/user/image.jpg",
  "signed_url": "https://storage.googleapis.com/...",
  "bloco": {
    "tipo": "imagem",
    "subtipo": "",
    "url": "gs://bucket/user/image.jpg",
    "nome": "image.jpg",
    "filename": "userId/image.jpg",
    "type": "image/jpeg",
    "created_at": "2025-11-04T00:00:00Z",
    "glb_url": "gs://bucket/user/ra/models/image.glb",
    "glb_signed_url": "https://storage.googleapis.com/...",
    "glb_source": "custom"  // ou "auto_generated"
  },
  "temp_id": "temp-123"
}
```

## 🎯 Campo `glb_source`

Indica a origem do modelo GLB:

- **`"custom"`**: GLB foi fornecido pelo usuário via `glb_file`
- **`"auto_generated"`**: GLB foi gerado automaticamente pelo backend a partir da imagem

## 📦 Metadata no GCS

### GLB Customizado
```json
{
  "generated_from_image": "gs://bucket/user/image.jpg",
  "base_height": "0.0",
  "custom_upload": "true",
  "original_filename": "modelo.glb"
}
```

### GLB Auto-gerado
```json
{
  "generated_from_image": "gs://bucket/user/image.jpg",
  "base_height": "0.0",
  "auto_generated": "true"
}
```

## 🔍 Validação de GLB

O backend valida:
1. **Content-Type**: Deve conter "model" ou ter extensão `.glb`
2. **Formato**: Arquivo deve ser GLB válido

Se a validação falhar, o backend **automaticamente gera** um GLB da imagem como fallback.

## 📝 Logs

### GLB Customizado
```
[add_content_image] GLB customizado fornecido: modelo.glb
[add_content_image] GLB customizado salvo: userId/ra/models/image.glb
[add_content_image] upload ok uid=userId filename=userId/image.jpg type=image/jpeg glb=SIM glb_source=custom
```

### GLB Auto-gerado
```
[add_content_image] Iniciando pré-geração de GLB para userId/image.jpg
[add_content_image] GLB auto-gerado com sucesso em 2.34s: userId/ra/models/image.glb
[add_content_image] upload ok uid=userId filename=userId/image.jpg type=image/jpeg glb=SIM glb_source=auto_generated
```

## 🧪 Exemplo de Uso (Frontend)

```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('glb_file', glbFile);  // Opcional
formData.append('name', 'minha-imagem.jpg');

const response = await fetch('/add-content-image/', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData
});

const data = await response.json();
console.log('GLB Source:', data.bloco.glb_source);  // 'custom' ou 'auto_generated'
```

## ✅ Benefícios

1. **Flexibilidade**: Usuários podem fornecer GLBs otimizados/customizados
2. **Fallback Automático**: Sempre há um GLB disponível, mesmo sem upload customizado
3. **Rastreabilidade**: Campo `glb_source` permite saber origem de cada modelo
4. **Backward Compatible**: Comportamento original mantido se `glb_file` não for fornecido
5. **Metadata Completa**: GCS armazena informações sobre origem do GLB

## 🚀 Próximos Passos

1. ✅ Backend implementado
2. ⏳ Frontend: Adicionar campo de upload de GLB no AdminUI
3. ⏳ Frontend: Indicador visual mostrando se GLB é custom ou auto-gerado
4. ⏳ App Mobile: Carregar múltiplos GLBs com navegação
