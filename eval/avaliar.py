"""
Script de avaliação da recuperação (Retrieval) do ReeciclaEBot-RAG.
Mede a taxa de acerto (Hit Rate / Recall) no Top-1 e no Top-3 usando 20 perguntas reformuladas.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Any

# Garante que a pasta raiz do projeto esteja no sys.path ao rodar "python eval/avaliar.py"
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from src.rag import BuscadorRAG  # noqa: E402

CAMINHO_PERGUNTAS_TESTE = Path(__file__).resolve().parent / "perguntas_teste.json"


def carregar_perguntas_teste(
    caminho: Path = CAMINHO_PERGUNTAS_TESTE,
) -> List[Dict[str, str]]:
    """Carrega a lista de perguntas de teste com o id_esperado."""
    with open(caminho, mode="r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def avaliar_recuperacao(
    buscador: BuscadorRAG, casos_teste: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Executa cada pergunta de teste no buscador e contabiliza acertos no Top-1 e no Top-3.
    """
    total = len(casos_teste)
    acertos_top1 = 0
    acertos_top3 = 0
    detalhes = []

    for caso in casos_teste:
        pergunta = caso["pergunta"]
        id_esperado = caso["id_esperado"]

        resultados = buscador.buscar(pergunta, top_k=3)
        ids_retornados = [item["id"] for item in resultados]
        nota_top1 = resultados[0]["pontuacao"] if resultados else 0.0

        acertou_1 = len(ids_retornados) > 0 and ids_retornados[0] == id_esperado
        acertou_3 = id_esperado in ids_retornados

        if acertou_1:
            acertos_top1 += 1
        if acertou_3:
            acertos_top3 += 1

        detalhes.append(
            {
                "pergunta": pergunta,
                "id_esperado": id_esperado,
                "ids_retornados": ids_retornados,
                "nota_top1": nota_top1,
                "acertou_top1": acertou_1,
                "acertou_top3": acertou_3,
            }
        )

    return {
        "total": total,
        "acertos_top1": acertos_top1,
        "acertos_top3": acertos_top3,
        "taxa_top1": (acertos_top1 / total) * 100.0 if total else 0.0,
        "taxa_top3": (acertos_top3 / total) * 100.0 if total else 0.0,
        "detalhes": detalhes,
    }


def imprimir_relatorio(metricas: Dict[str, Any]) -> None:
    """Imprime no terminal o resultado detalhado por pergunta e o resumo final."""
    print("=" * 72)
    print("AVALIAÇÃO DE RECUPERAÇÃO (RETRIEVAL) - ReeciclaEBot-RAG")
    print("=" * 72)

    for idx, item in enumerate(metricas["detalhes"], start=1):
        status = (
            "[TOP-1 OK]"
            if item["acertou_top1"]
            else ("[TOP-3 OK]" if item["acertou_top3"] else "[ERRO]    ")
        )
        print(
            f"{idx:02d}. {status} Esperado: {item['id_esperado']} | "
            f"Retornados: {item['ids_retornados']} | "
            f"Score Top-1: {item['nota_top1']:.3f}"
        )

    print("-" * 72)
    print(f"Total de perguntas avaliadas : {metricas['total']}")
    print(
        f"Taxa de acerto Top-1         : {metricas['acertos_top1']}/{metricas['total']} "
        f"({metricas['taxa_top1']:.1f}%)"
    )
    print(
        f"Taxa de acerto Top-3         : {metricas['acertos_top3']}/{metricas['total']} "
        f"({metricas['taxa_top3']:.1f}%)"
    )
    print("=" * 72)


if __name__ == "__main__":
    casos = carregar_perguntas_teste()
    buscador_rag = BuscadorRAG()
    resultado = avaliar_recuperacao(buscador_rag, casos)
    imprimir_relatorio(resultado)
