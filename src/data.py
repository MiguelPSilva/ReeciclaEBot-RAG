"""
Carregamento e preparação dos dados da base de conhecimento (JSON).
"""

import json
from pathlib import Path
from typing import Dict, List

# Define o caminho até o arquivo data/base_conhecimento.json
CAMINHO_PADRAO_BASE = (
    Path(__file__).resolve().parent.parent / "data" / "base_conhecimento.json"
)

# Campos que todo item do JSON precisa ter
CAMPOS_OBRIGATORIOS = ("id", "pergunta", "resposta", "categoria", "fonte")

# TODO: Revisar caso novas categorias de resíduos sejam adicionadas à base
CATEGORIAS_ESPERADAS = {
    "pilhas e baterias",
    "óleo de cozinha",
    "lâmpadas",
    "eletroeletrônicos",
    "medicamentos vencidos",
    "vidro",
    "papel",
    "plástico",
    "metal",
    "entulho",
    "pneus",
    "orgânicos",
}


def validar_item(item: Dict[str, str], indice: int) -> None:
    """Verifica se o item possui todos os campos obrigatórios preenchidos."""
    for campo in CAMPOS_OBRIGATORIOS:
        if campo not in item or not str(item[campo]).strip():
            raise ValueError(
                f"Item na posição {indice} está sem o campo obrigatório '{campo}'."
            )


def carregar_base_conhecimento(
    caminho: Path | str = CAMINHO_PADRAO_BASE,
) -> List[Dict[str, str]]:
    """Abre o arquivo JSON, valida os itens e retorna a lista de documentos."""
    caminho_arquivo = Path(caminho)
    if not caminho_arquivo.exists():
        raise FileNotFoundError(
            f"Base de conhecimento não encontrada em: {caminho_arquivo}"
        )

    # Lê o arquivo usando UTF-8 para preservar acentos do português
    with open(caminho_arquivo, mode="r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    if not isinstance(dados, list) or len(dados) == 0:
        raise ValueError("A base de conhecimento deve ser uma lista JSON não vazia.")

    # Valida documento por documento
    for indice, item in enumerate(dados):
        validar_item(item, indice)

    return dados


def montar_texto_para_embedding(item: Dict[str, str]) -> str:
    """
    Prepara o texto de cada item antes de transformá-lo em vetor (embedding).
    Remove a última frase padrão ('Confirme com a prefeitura...') apenas do cálculo
    do vetor, pois ela se repete em todos os itens e deixaria os vetores muito parecidos.
    """
    # Divide a resposta em frases separadas por ponto final
    frases = [f.strip() for f in item["resposta"].split(".") if f.strip()]

    # Se a última frase for o aviso padrão da prefeitura, tira do texto de busca
    if len(frases) > 1 and "prefeitura" in frases[-1].lower():
        frases = frases[:-1]
    resposta_sem_rodape = ". ".join(frases) + "."

    # Junta pergunta, categoria e explicação principal
    return (
        f"{item['pergunta']} "
        f"(Categoria: {item['categoria']}). "
        f"{item['pergunta']} "
        f"{resposta_sem_rodape}"
    )
