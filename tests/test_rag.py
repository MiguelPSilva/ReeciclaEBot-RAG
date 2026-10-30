"""
Testes automatizados com pytest para validar a base de dados, o cálculo de cosseno e a busca RAG.
"""

import numpy as np
import pytest

from src.data import carregar_base_conhecimento, CAMPOS_OBRIGATORIOS
from src.rag import (
    BuscadorRAG,
    calcular_similaridade_cosseno,
    MENSAGEM_SEM_INFORMACAO,
)
from src.llm import gerar_resposta


@pytest.fixture(scope="module")
def buscador() -> BuscadorRAG:
    """Carrega o buscador uma única vez para reutilizar em todos os testes."""
    return BuscadorRAG(limiar=0.45)


def test_base_conhecimento_possui_cerca_de_30_itens() -> None:
    """Confere se o JSON tem os 30 itens e se todos possuem os campos obrigatórios."""
    dados = carregar_base_conhecimento()
    assert len(dados) == 30

    for item in dados:
        for campo in CAMPOS_OBRIGATORIOS:
            assert campo in item and len(item[campo].strip()) > 0
        # Verifica se a resposta orienta a confirmar com a prefeitura ou cooperativa
        resposta_min = item["resposta"].lower()
        assert "prefeitura" in resposta_min and "cooperativa" in resposta_min


def test_calculo_similaridade_cosseno_numpy() -> None:
    """Testa a fórmula matemática do cosseno com vetores conhecidos."""
    vetor_pergunta = np.array([1.0, 0.0], dtype=np.float32)
    matriz_base = np.array(
        [
            [1.0, 0.0],   # Vetor igual -> resultado esperado: 1.0
            [0.0, 1.0],   # Vetor de 90 graus -> resultado esperado: 0.0
            [-1.0, 0.0],  # Vetor oposto -> resultado esperado: -1.0
        ],
        dtype=np.float32,
    )
    scores = calcular_similaridade_cosseno(vetor_pergunta, matriz_base)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.0)
    assert np.isclose(scores[2], -1.0)


def test_busca_retorna_top_3_ordenado_com_pontuacao(buscador: BuscadorRAG) -> None:
    """Verifica se a busca devolve 3 documentos ordenados da maior para a menor nota."""
    resultados = buscador.buscar("Onde descartar pilhas e baterias usadas?", top_k=3)
    assert len(resultados) == 3

    for item in resultados:
        assert "pontuacao" in item
        assert -1.0 <= item["pontuacao"] <= 1.0001

    assert resultados[0]["pontuacao"] >= resultados[1]["pontuacao"]
    assert resultados[1]["pontuacao"] >= resultados[2]["pontuacao"]
    assert resultados[0]["id"] == "doc_01"


def test_pergunta_fora_do_tema_fica_abaixo_do_limiar(buscador: BuscadorRAG) -> None:
    """Garante que perguntas sem relação com reciclagem fiquem abaixo do limiar."""
    buscador_rigoroso = BuscadorRAG(itens=buscador.itens, limiar=0.95)
    resultados = buscador_rigoroso.buscar("Qual a receita de bolo de chocolate?")
    confiavel = buscador_rigoroso.tem_confianca_suficiente(resultados)

    assert confiavel is False
    resposta = gerar_resposta(
        pergunta="Qual a receita de bolo de chocolate?",
        trechos_recuperados=resultados,
        confianca_suficiente=confiavel,
    )
    assert resposta == MENSAGEM_SEM_INFORMACAO


def test_fallback_sem_token_retorna_item_mais_parecido(
    buscador: BuscadorRAG, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Testa se o modo fallback retorna a resposta da base quando não há HF_TOKEN."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    resultados = buscador.buscar("Como descartar óleo de fritura usado?", top_k=3)
    resposta = gerar_resposta(
        pergunta="Como descartar óleo de fritura usado?",
        trechos_recuperados=resultados,
        confianca_suficiente=True,
    )
    assert resultados[0]["resposta"] in resposta
    assert resultados[0]["categoria"] in resposta
