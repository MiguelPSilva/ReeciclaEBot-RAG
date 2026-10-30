"""
Módulo de geração de resposta (Generation) do RAG.
Consulta um modelo de chat via Hugging Face InferenceClient ou usa fallback direto.
"""

from pathlib import Path
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.rag import MENSAGEM_SEM_INFORMACAO

# Carrega as variáveis definidas no arquivo .env (se existir)
CAMINHO_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=CAMINHO_ENV, override=True)

MODELO_LLM_PADRAO = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_SISTEMA = (
    "Você é o ReeciclaEBot, um assistente sobre descarte correto e "
    "reciclagem de resíduos no Brasil. "
    "Responda em português APENAS com base nos trechos recuperados fornecidos abaixo. "
    "Não invente informações, números de leis ou locais que não estejam nos trechos. "
    "Cite a categoria do resíduo na resposta e lembre o usuário "
    "de confirmar os pontos de coleta com a prefeitura ou cooperativa local."
)


def montar_contexto(trechos: List[Dict[str, Any]]) -> str:
    """Junta os trechos recuperados em um texto único para enviar no prompt do LLM."""
    blocos = []
    for indice, item in enumerate(trechos, start=1):
        bloco = (
            f"[Trecho {indice} | Categoria: {item['categoria']} | "
            f"Similaridade: {item['pontuacao']:.2f}]\n"
            f"Pergunta: {item['pergunta']}\n"
            f"Resposta: {item['resposta']}\n"
            f"Fonte: {item['fonte']}"
        )
        blocos.append(bloco)
    return "\n\n".join(blocos)


def _limpar_todo_da_fonte(fonte: str) -> str:
    """Retira marcações internas de revisão '(TODO: ...)' do texto da fonte."""
    return fonte.split("(TODO:")[0].strip()


def gerar_resposta_fallback(melhor_item: Dict[str, Any]) -> str:
    """
    Retorna diretamente a resposta do documento mais parecido (Top-1)
    quando não há HF_TOKEN configurado ou quando a API falha.
    """
    fonte_limpa = _limpar_todo_da_fonte(melhor_item["fonte"])
    return (
        f"{melhor_item['resposta']}\n\n"
        f"Categoria: {melhor_item['categoria']} | "
        f"Referência: {fonte_limpa}"
    )


def _token_valido(token: str | None) -> bool:
    """Checa se o token foi preenchido e não é o valor padrão de exemplo."""
    if not token:
        return False
    token_limpo = token.strip()
    return bool(token_limpo) and token_limpo != "seu_token_aqui"


def gerar_resposta(
    pergunta: str,
    trechos_recuperados: List[Dict[str, Any]],
    confianca_suficiente: bool = True,
) -> str:
    """
    Fluxo principal de geração:
    1. Se a pontuação for menor que o limiar, retorna aviso de que não possui a informação.
    2. Se houver HF_TOKEN válido, envia os trechos para o LLM gerar a resposta.
    3. Caso contrário (ou se a API der erro), retorna o fallback com o item mais parecido.
    """
    if not trechos_recuperados or not confianca_suficiente:
        return MENSAGEM_SEM_INFORMACAO

    token = os.getenv("HF_TOKEN")
    if not _token_valido(token):
        return gerar_resposta_fallback(trechos_recuperados[0])

    modelo = os.getenv("HF_MODEL", MODELO_LLM_PADRAO)
    contexto = montar_contexto(trechos_recuperados)
    mensagem_usuario = (
        f"Trechos recuperados da base:\n{contexto}\n\n"
        f"Pergunta do usuário: {pergunta}\n\n"
        "Responda usando apenas os trechos acima e cite a categoria:"
    )

    try:
        cliente = InferenceClient(model=modelo, token=token)
        resposta_chat = cliente.chat_completion(
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": mensagem_usuario},
            ],
            max_tokens=350,
            temperature=0.1,
        )
        texto_gerado = resposta_chat.choices[0].message.content
        if texto_gerado and texto_gerado.strip():
            return texto_gerado.strip()
    except Exception:
        # TODO: Adicionar log de erro caso queira monitorar falhas da API
        pass

    return gerar_resposta_fallback(trechos_recuperados[0])
