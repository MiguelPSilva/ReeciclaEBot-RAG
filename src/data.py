"""
Módulo responsável por carregar e validar a base de conhecimento em JSON.
"""

import json
from pathlib import Path
from typing import Dict, List

# Caminho padrão para o arquivo JSON da base de conhecimento
CAMINHO_PADRAO_BASE = (
    Path(__file__).resolve().parent.parent / "data" / "base_conhecimento.json"
)

# Campos obrigatórios que cada item do JSON deve ter
CAMPOS_OBRIGATORIOS = ("id", "pergunta", "resposta", "categoria", "fonte")

# TODO: Revisar se deseja adicionar novos temas regionais na lista abaixo
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
    """Verifica se um item da base contém todos os campos obrigatórios preenchidos."""
    for campo in CAMPOS_OBRIGATORIOS:
        if campo not in item or not str(item[campo]).strip():
            raise ValueError(
                f"Item na posição {indice} está sem o campo obrigatório '{campo}'."
            )


def carregar_base_conhecimento(
    caminho: Path | str = CAMINHO_PADRAO_BASE,
) -> List[Dict[str, str]]:
    """Lê o arquivo JSON e retorna a lista de documentos validada."""
    caminho_arquivo = Path(caminho)
    if not caminho_arquivo.exists():
        raise FileNotFoundError(
            f"Base de conhecimento não encontrada em: {caminho_arquivo}"
        )

    with open(caminho_arquivo, mode="r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    if not isinstance(dados, list) or len(dados) == 0:
        raise ValueError("A base de conhecimento deve ser uma lista JSON não vazia.")

    for indice, item in enumerate(dados):
        validar_item(item, indice)

    return dados


def montar_texto_para_embedding(item: Dict[str, str]) -> str:
    """
    Monta o texto curto usado para gerar o embedding de cada documento.
    Como o modelo paraphrase-multilingual-MiniLM-L12-v2 foi treinado em sentenças
    curtas (paráfrases), usamos a pergunta, a categoria e a primeira frase da resposta
    para evitar que o aviso padrão final dilua o vetor semântico.
    """
    # TODO: Experimentar diferentes combinações de campos na avaliação (eval/avaliar.py)
    primeira_frase_resposta = item["resposta"].split(".")[0]
    return (
        f"{item['pergunta']} "
        f"(Categoria: {item['categoria']}). "
        f"{primeira_frase_resposta}."
    )

