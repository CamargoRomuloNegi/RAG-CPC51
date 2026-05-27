
"""
Artefato 03 — App Streamlit RAG Consultivo CPC 51

Objetivo:
    Criar uma interface prática para consultar o banco vetorial Supabase/pgvector
    carregado no Artefato 02.

Fluxo implementado:
    1. Usuário informa uma pergunta contábil.
    2. IA interpretadora reescreve a pergunta para busca técnica.
    3. Gemini Embedding gera vetor da pergunta interpretada.
    4. Supabase executa função match_cpc_items.
    5. Itens recuperados são exibidos ao usuário.
    6. IA especialista responde exclusivamente com base nos itens recuperados.
    7. Consulta, itens, prompt e resposta são gravados em rag_auditoria.

Premissas:
    - O Supabase já foi criado pelo Artefato 02.
    - A tabela cpc_itens já contém embeddings em 3072 dimensões.
    - A função match_cpc_items já existe no banco.
    - O modelo de embedding configurado é gemini-embedding-2.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from supabase import create_client, Client
from google import genai


# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="RAG CPC 51",
    page_icon="📘",
    layout="wide",
)

st.title("📘 RAG Consultivo — CPC 51")
st.caption(
    "Protótipo beta: consulta vetorial em Supabase/pgvector com resposta fundamentada nos itens recuperados."
)


# ============================================================
# 2. CONSTANTES DO PROJETO
# ============================================================

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_RESPONSE_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_DIMENSION = 3072
DEFAULT_MATCH_COUNT = 8
DEFAULT_SIMILARITY_THRESHOLD = 0.0


# ============================================================
# 3. FUNÇÕES DE CONEXÃO
# ============================================================

@st.cache_resource(show_spinner=False)
def get_supabase_client(url: str, key: str) -> Client:
    """
    Cria cliente Supabase.

    Uso de cache:
        Evita recriar a conexão a cada interação do Streamlit.
    """
    return create_client(url, key)


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str) -> genai.Client:
    """
    Cria cliente Gemini.
    """
    return genai.Client(api_key=api_key)


def validate_required_config(
    supabase_url: str,
    supabase_key: str,
    google_api_key: str,
) -> bool:
    """
    Verifica se as credenciais mínimas foram preenchidas.
    """
    return bool(
        supabase_url.strip()
        and supabase_key.strip()
        and google_api_key.strip()
    )


# ============================================================
# 4. FUNÇÕES DE IA
# ============================================================

def extract_embedding_values(embedding_obj: Any) -> List[float]:
    """
    Extrai os valores numéricos do embedding retornado pela biblioteca.

    A função é defensiva para lidar com pequenas variações de retorno.
    """
    if isinstance(embedding_obj, dict):
        if "values" in embedding_obj:
            return list(embedding_obj["values"])
        if "embedding" in embedding_obj:
            return list(embedding_obj["embedding"])

    values = getattr(embedding_obj, "values", None)
    if values is not None:
        return list(values)

    embedding = getattr(embedding_obj, "embedding", None)
    if embedding is not None:
        return list(embedding)

    raise ValueError("Não foi possível extrair os valores do embedding.")


def generate_embedding(
    gemini_client: genai.Client,
    text: str,
    embedding_model: str,
    output_dimensionality: int,
) -> List[float]:
    """
    Gera embedding de uma string usando Gemini.

    O banco foi criado com vector(3072), portanto exigimos 3072 dimensões.
    """
    response = gemini_client.models.embed_content(
        model=embedding_model,
        contents=[text],
        config={"output_dimensionality": output_dimensionality},
    )

    if not getattr(response, "embeddings", None):
        raise RuntimeError("A API não retornou embeddings.")

    vector = extract_embedding_values(response.embeddings[0])

    if len(vector) != output_dimensionality:
        raise RuntimeError(
            f"Dimensão inválida do embedding. "
            f"Esperado {output_dimensionality}, recebido {len(vector)}."
        )

    return vector


def generate_text_response(
    gemini_client: genai.Client,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> str:
    """
    Gera resposta textual com Gemini.

    Temperatura baixa:
        Reduz criatividade e favorece resposta técnica/fundamentada.
    """
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature},
    )

    text = getattr(response, "text", None)
    if text:
        return text.strip()

    # Fallback para eventuais formatos alternativos.
    return str(response).strip()


def interpretar_pergunta(
    gemini_client: genai.Client,
    pergunta_usuario: str,
    response_model: str,
) -> str:
    """
    IA interpretadora.

    Função:
        Não responde à pergunta.
        Apenas reescreve a pergunta em linguagem mais adequada para busca vetorial.

    Exemplo:
        Usuário: "Como fica despesa financeira?"
        Saída esperada: "classificação de receitas e despesas financeiras na demonstração do resultado,
        categoria de financiamento, categoria operacional, atividades de negócio principais..."
    """
    prompt = f"""
Você é uma IA interpretadora para busca em uma base de Pronunciamentos Técnicos do CPC.

Sua tarefa NÃO é responder à pergunta.

Sua tarefa é reescrever a pergunta do usuário em uma consulta técnica curta, clara e rica em termos contábeis, adequada para busca vetorial.

Inclua:
- termos contábeis prováveis;
- sinônimos relevantes;
- conceitos relacionados;
- possível linguagem normativa.

Não invente conclusão.
Não cite itens se não tiver certeza.
Não responda juridicamente ou contabilmente.
Apenas devolva a consulta técnica reescrita.

Pergunta do usuário:
{pergunta_usuario}

Consulta técnica para busca vetorial:
""".strip()

    resposta = generate_text_response(
        gemini_client=gemini_client,
        model=response_model,
        prompt=prompt,
        temperature=0.1,
    )

    return resposta.strip()


def montar_prompt_especialista(
    pergunta_usuario: str,
    pergunta_interpretada: str,
    itens: List[Dict[str, Any]],
) -> str:
    """
    Monta o prompt final da IA especialista.

    Regra central:
        A resposta deve se limitar aos itens recuperados.
    """
    blocos_contexto = []

    for i, item in enumerate(itens, start=1):
        blocos_contexto.append(
            f"[Trecho {i}]\n"
            f"Documento: {item.get('documento_codigo')}\n"
            f"Título: {item.get('documento_titulo')}\n"
            f"Seção: {item.get('secao')}\n"
            f"Item: {item.get('item_codigo')}\n"
            f"Similaridade: {item.get('similarity')}\n"
            f"Texto:\n{item.get('texto_completo')}\n"
        )

    contexto = "\n---\n".join(blocos_contexto)

    return f"""
Você é um especialista contábil em Pronunciamentos Técnicos do CPC.

Sua tarefa é responder à pergunta do usuário exclusivamente com base nos trechos recuperados do banco vetorial.

Regras obrigatórias:
1. Use somente os trechos fornecidos.
2. Cite documento, seção e item em cada fundamento relevante.
3. Não use conhecimento externo.
4. Não invente itens, exceções ou conclusões.
5. Se os trechos recuperados forem insuficientes, diga expressamente que a base recuperada não permite conclusão segura.
6. Quando houver tensão entre trechos, explique a limitação.
7. Mantenha tom técnico, objetivo e didático.

Pergunta original do usuário:
{pergunta_usuario}

Pergunta interpretada para busca:
{pergunta_interpretada}

Trechos recuperados do banco vetorial:
{contexto}

Resposta fundamentada:
""".strip()


# ============================================================
# 5. FUNÇÕES DE BANCO
# ============================================================

def buscar_itens_cpc(
    supabase: Client,
    gemini_client: genai.Client,
    pergunta_interpretada: str,
    embedding_model: str,
    embedding_dimension: int,
    match_count: int,
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Gera embedding da pergunta interpretada e executa a função match_cpc_items.
    """
    query_embedding = generate_embedding(
        gemini_client=gemini_client,
        text=pergunta_interpretada,
        embedding_model=embedding_model,
        output_dimensionality=embedding_dimension,
    )

    result = supabase.rpc(
        "match_cpc_items",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "similarity_threshold": similarity_threshold,
        },
    ).execute()

    return result.data or []


def registrar_auditoria(
    supabase: Client,
    pergunta_usuario: str,
    pergunta_interpretada: str,
    itens_recuperados: List[Dict[str, Any]],
    prompt_final: str,
    resposta_ia: str,
    modelo_embedding: str,
    modelo_resposta: str,
) -> None:
    """
    Registra trilha de auditoria no Supabase.

    Essa tabela é essencial para aula:
        Permite demonstrar o que o usuário perguntou,
        o que foi recuperado,
        o que foi enviado à IA
        e o que a IA respondeu.
    """
    payload = {
        "pergunta_usuario": pergunta_usuario,
        "pergunta_interpretada": pergunta_interpretada,
        "itens_recuperados_json": itens_recuperados,
        "prompt_final": prompt_final,
        "resposta_ia": resposta_ia,
        "modelo_embedding": modelo_embedding,
        "modelo_resposta": modelo_resposta,
        "quantidade_itens_recuperados": len(itens_recuperados),
    }

    supabase.table("rag_auditoria").insert(payload).execute()


# ============================================================
# 6. SIDEBAR — CONFIGURAÇÕES
# ============================================================

with st.sidebar:
    st.header("⚙️ Configurações")

    st.markdown("### Credenciais")

    supabase_url = st.text_input(
        "SUPABASE_URL",
        value=st.secrets.get("SUPABASE_URL", ""),
        type="password",
        help="URL do projeto Supabase.",
    )

    supabase_key = st.text_input(
        "SUPABASE_KEY",
        value=st.secrets.get("SUPABASE_KEY", ""),
        type="password",
        help="Para beta local, use Service Role Key. Não exponha em produção.",
    )

    google_api_key = st.text_input(
        "GOOGLE_API_KEY",
        value=st.secrets.get("GOOGLE_API_KEY", ""),
        type="password",
        help="Chave da Gemini API / Google AI Studio.",
    )

    st.markdown("### Modelos")

    embedding_model = st.text_input(
        "Modelo de embedding",
        value=st.secrets.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )

    response_model = st.text_input(
        "Modelo de resposta",
        value=st.secrets.get("RESPONSE_MODEL", DEFAULT_RESPONSE_MODEL),
    )

    embedding_dimension = st.number_input(
        "Dimensão do embedding",
        min_value=1,
        max_value=4096,
        value=int(st.secrets.get("EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION)),
        step=1,
    )

    st.markdown("### Busca")

    match_count = st.slider(
        "Quantidade de itens recuperados",
        min_value=3,
        max_value=20,
        value=DEFAULT_MATCH_COUNT,
        step=1,
    )

    similarity_threshold = st.slider(
        "Similaridade mínima",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SIMILARITY_THRESHOLD,
        step=0.01,
    )

    mostrar_prompt = st.checkbox(
        "Mostrar prompt final enviado à IA",
        value=False,
    )

    gravar_auditoria = st.checkbox(
        "Gravar auditoria no Supabase",
        value=True,
    )


# ============================================================
# 7. ORIENTAÇÃO INICIAL
# ============================================================

with st.expander("📌 Como este protótipo funciona", expanded=True):
    st.markdown(
        """
        Este app executa um RAG consultivo em quatro camadas:

        1. **Interpretação da pergunta**  
           Uma IA reescreve a pergunta para melhorar a busca vetorial.

        2. **Busca vetorial no Supabase**  
           A pergunta interpretada é transformada em embedding e enviada à função `match_cpc_items`.

        3. **Resposta especialista**  
           Uma segunda chamada à IA responde com base apenas nos itens recuperados.

        4. **Auditoria**  
           O app pode registrar pergunta, itens recuperados, prompt e resposta na tabela `rag_auditoria`.

        A resposta não deve ser tratada como parecer contábil final. É uma demonstração técnica de RAG fundamentado.
        """
    )


# ============================================================
# 8. FORMULÁRIO PRINCIPAL
# ============================================================

st.subheader("Consulta ao CPC 51")

pergunta_usuario = st.text_area(
    "Digite sua pergunta:",
    height=120,
    placeholder=(
        "Exemplo: Como o CPC 51 orienta a classificação de receitas e despesas "
        "na demonstração do resultado?"
    ),
)

executar = st.button("Consultar base CPC 51", type="primary")


# ============================================================
# 9. EXECUÇÃO DO RAG
# ============================================================

if executar:
    if not pergunta_usuario.strip():
        st.warning("Digite uma pergunta antes de consultar.")
        st.stop()

    if not validate_required_config(supabase_url, supabase_key, google_api_key):
        st.error("Preencha SUPABASE_URL, SUPABASE_KEY e GOOGLE_API_KEY.")
        st.stop()

    try:
        supabase = get_supabase_client(supabase_url, supabase_key)
        gemini_client = get_gemini_client(google_api_key)

        with st.status("Executando RAG consultivo...", expanded=True) as status:
            st.write("1. Interpretando a pergunta para busca vetorial...")

            pergunta_interpretada = interpretar_pergunta(
                gemini_client=gemini_client,
                pergunta_usuario=pergunta_usuario,
                response_model=response_model,
            )

            st.write("2. Gerando embedding e buscando itens no Supabase...")

            itens_recuperados = buscar_itens_cpc(
                supabase=supabase,
                gemini_client=gemini_client,
                pergunta_interpretada=pergunta_interpretada,
                embedding_model=embedding_model,
                embedding_dimension=int(embedding_dimension),
                match_count=int(match_count),
                similarity_threshold=float(similarity_threshold),
            )

            if not itens_recuperados:
                status.update(label="Nenhum item recuperado.", state="error")
                st.error("A busca vetorial não retornou itens. Reduza o threshold ou revise a pergunta.")
                st.stop()

            st.write("3. Montando prompt especialista...")

            prompt_final = montar_prompt_especialista(
                pergunta_usuario=pergunta_usuario,
                pergunta_interpretada=pergunta_interpretada,
                itens=itens_recuperados,
            )

            st.write("4. Gerando resposta fundamentada...")

            resposta_final = generate_text_response(
                gemini_client=gemini_client,
                model=response_model,
                prompt=prompt_final,
                temperature=0.2,
            )

            if gravar_auditoria:
                st.write("5. Gravando auditoria...")
                registrar_auditoria(
                    supabase=supabase,
                    pergunta_usuario=pergunta_usuario,
                    pergunta_interpretada=pergunta_interpretada,
                    itens_recuperados=itens_recuperados,
                    prompt_final=prompt_final,
                    resposta_ia=resposta_final,
                    modelo_embedding=embedding_model,
                    modelo_resposta=response_model,
                )

            status.update(label="Consulta concluída.", state="complete")

        # ------------------------------------------------------------
        # Exibição dos resultados
        # ------------------------------------------------------------

        st.markdown("## Resposta fundamentada")
        st.write(resposta_final)

        st.markdown("## Pergunta interpretada")
        st.info(pergunta_interpretada)

        st.markdown("## Itens recuperados")

        for i, item in enumerate(itens_recuperados, start=1):
            similarity = item.get("similarity")
            similarity_txt = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else str(similarity)

            with st.expander(
                f"{i}. {item.get('documento_codigo')} — Item {item.get('item_codigo')} — Similaridade {similarity_txt}",
                expanded=(i <= 3),
            ):
                st.markdown(f"**Seção:** {item.get('secao')}")
                st.markdown(f"**Item:** {item.get('item_codigo')}")
                st.write(item.get("texto_completo"))

        if mostrar_prompt:
            st.markdown("## Prompt final enviado à IA")
            st.code(prompt_final, language="markdown")

        st.success("Consulta finalizada com sucesso.")

    except Exception as exc:
        st.error("Ocorreu erro durante a execução.")
        st.exception(exc)


# ============================================================
# 10. RODAPÉ DIDÁTICO
# ============================================================

st.divider()
st.caption(
    "Artefato de Consulta — App Streamlit. Este protótipo depende dos Artefatos 01 e 02 já executados com sucesso."
)
