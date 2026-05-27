# Artefato 03 — App Streamlit RAG CPC 51

## Objetivo

Este app demonstra um RAG consultivo sobre o CPC 51, usando:

- Supabase/Postgres;
- pgvector;
- função `match_cpc_items`;
- Gemini Embedding;
- Gemini para interpretação e resposta;
- tabela `rag_auditoria`.

## Pré-requisitos

Antes de executar este app, os Artefatos 01 e 02 devem estar concluídos:

1. Artefato 01: parser manual do CPC 51.
2. Artefato 02: criação do banco, geração dos embeddings e carga no Supabase.

## Instalação local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuração de chaves

Você pode informar as chaves pela barra lateral do app ou criar o arquivo:

```text
.streamlit/secrets.toml
```

com o conteúdo:

```toml
SUPABASE_URL="https://seu-projeto.supabase.co"
SUPABASE_KEY="sua-chave"
GOOGLE_API_KEY="sua-chave"

EMBEDDING_MODEL="gemini-embedding-2"
RESPONSE_MODEL="gemini-2.5-flash"
EMBEDDING_DIMENSION=3072
```

## Funcionamento

1. O usuário pergunta.
2. A IA interpretadora transforma a pergunta em consulta técnica.
3. A consulta técnica é vetorizada.
4. A função `match_cpc_items` recupera itens do CPC 51.
5. A IA especialista responde com base nos itens recuperados.
6. A consulta é registrada em `rag_auditoria`.

## Observação de segurança

Para beta local, a `SERVICE_ROLE_KEY` facilita a demonstração.

Para produção, não exponha a Service Role Key no frontend. Use backend, RLS e chaves restritas.
