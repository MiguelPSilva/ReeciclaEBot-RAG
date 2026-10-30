"""
Módulo de Geração (Generation) do RAG.
Usa o InferenceClient da Hugging Face com fallback automático para o trecho mais relevante.
"""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.rag import MENSAGEM_SEM_INFORMACAO

# Carrega variáveis do arquivo .env, caso exista localmente
load_dotenv()

# Modelo de chat leve e gratuito na Inference API da Hugging Face
MODELO_LLM_PADRAO = "Qwen/Qwen2.5-7B-Instruct"

PROMPT_SISTEMA = (
    "Você é o ReeciclaEBot, um assistente educativo sobre descarte correto e "
    "reciclagem de resíduos no Brasil. "
    "Responda em português APENAS com base nos trechos recuperados fornecidos abaixo. "
    "Não invente informações, números de leis ou endereços que não estejam nos trechos. "
    "Cite explicitamente a categoria do resíduo na sua resposta e lembre o usuário "
    "de confirmar os pontos de coleta com a prefeitura ou cooperativa local."
)


def montar_contexto(trechos: List[Dict[str, Any]]) -> str:
    """Formata os 3 trechos recuperados em texto legível para enviar ao LLM."""
    blocos = []
    for indice, item in enumerate(trechos, start=1):
        bloco = (
            f"[Trecho {indice} | Categoria: {item['categoria']} | "
            f"Similaridade: {item['pontuacao']:.2f}]\n"
            f"Pergunta base: {item['pergunta']}\n"
            f"Resposta base: {item['resposta']}\n"
            f"Fonte: {item['fonte']}"
        )
        blocos.append(bloco)
    return "\n\n".join(blocos)


def _limpar_todo_da_fonte(fonte: str) -> str:
    """Remove anotações internas de TODO da fonte antes de exibir ao usuário final."""
    return fonte.split("(TODO:")[0].strip()


def gerar_resposta_fallback(melhor_item: Dict[str, Any]) -> str:
    """
    Modo de segurança (fallback): usado quando não há HF_TOKEN configurado
    ou quando a chamada externa à API da Hugging Face falha.
    """
    fonte_limpa = _limpar_todo_da_fonte(melhor_item["fonte"])
    return (
        f"{melhor_item['resposta']}\n\n"
        f"Categoria: {melhor_item['categoria']} | "
        f"Referência: {fonte_limpa}"
    )


def _token_valido(token: str | None) -> bool:
    """Verifica se o token de ambiente existe e não é apenas o texto de exemplo."""
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
    Gera a resposta final ao usuário.
    1. Se a pontuação máxima ficou abaixo do limiar, informa que não sabe.
    2. Se houver HF_TOKEN válido, consulta o modelo de chat com os trechos.
    3. Se não houver token ou ocorrer erro de rede/API, usa o fallback direto.
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
        # TODO: Em produção, registrar o erro em um log estruturado (logging)
        pass

    return gerar_resposta_fallback(trechos_recuperados[0])
