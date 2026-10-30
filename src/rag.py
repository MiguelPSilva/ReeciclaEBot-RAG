"""
Módulo de Recuperação (Retrieval) do RAG.
Gera embeddings com sentence-transformers e calcula a similaridade de cosseno com numpy.
"""

from typing import Any, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data import carregar_base_conhecimento, montar_texto_para_embedding

# Modelo multilíngue leve que entende bem frases em português
MODELO_EMBEDDING_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# TODO: Ajustar este limiar (entre 0.0 e 1.0) após observar os testes práticos
LIMIAR_PADRAO = 0.45

MENSAGEM_SEM_INFORMACAO = (
    "Desculpe, não encontrei informações confiáveis na minha base sobre esse assunto. "
    "Por favor, consulte a prefeitura ou a cooperativa de reciclagem da sua cidade."
)


def calcular_similaridade_cosseno(
    vetor_pergunta: np.ndarray, matriz_base: np.ndarray
) -> np.ndarray:
    """
    Calcula a similaridade de cosseno entre o vetor da pergunta (1D)
    e todos os vetores da base de conhecimento (2D) usando numpy.
    Fórmula: (A . B) / (||A|| * ||B||)
    """
    produto_escalar = np.dot(matriz_base, vetor_pergunta)
    norma_pergunta = np.linalg.norm(vetor_pergunta)
    normas_base = np.linalg.norm(matriz_base, axis=1)

    # Evita divisão por zero caso algum vetor tenha norma 0
    denominador = np.maximum(normas_base * norma_pergunta, 1e-10)
    return produto_escalar / denominador


class BuscadorRAG:
    """Indexa a base de conhecimento em memória e busca os itens mais parecidos."""

    def __init__(
        self,
        itens: List[Dict[str, str]] | None = None,
        nome_modelo: str = MODELO_EMBEDDING_PADRAO,
        limiar: float = LIMIAR_PADRAO,
    ) -> None:
        self.itens = itens if itens is not None else carregar_base_conhecimento()
        self.limiar = limiar
        self.modelo = SentenceTransformer(nome_modelo)
        self.embeddings_base = self._gerar_embeddings_base()

    def _gerar_embeddings_base(self) -> np.ndarray:
        """Transforma todos os itens da base em vetores numéricos (embeddings)."""
        textos = [montar_texto_para_embedding(item) for item in self.itens]
        vetores = self.modelo.encode(textos, convert_to_numpy=True)
        return np.asarray(vetores, dtype=np.float32)

    def buscar(self, pergunta: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retorna os `top_k` itens mais parecidos com a pergunta,
        incluindo o campo 'pontuacao' (similaridade de cosseno).
        """
        if not pergunta or not pergunta.strip():
            return []

        vetor_pergunta = self.modelo.encode(pergunta.strip(), convert_to_numpy=True)
        vetor_pergunta = np.asarray(vetor_pergunta, dtype=np.float32)

        pontuacoes = calcular_similaridade_cosseno(vetor_pergunta, self.embeddings_base)

        # Ordena os índices da maior pontuação para a menor
        indices_ordenados = np.argsort(pontuacoes)[::-1][:top_k]

        resultados: List[Dict[str, Any]] = []
        for idx in indices_ordenados:
            item_copia = dict(self.itens[int(idx)])
            item_copia["pontuacao"] = float(pontuacoes[int(idx)])
            resultados.append(item_copia)

        return resultados

    def tem_confianca_suficiente(self, resultados: List[Dict[str, Any]]) -> bool:
        """Verifica se a maior pontuação atingiu o limiar mínimo configurado."""
        if not resultados:
            return False
        return resultados[0]["pontuacao"] >= self.limiar

