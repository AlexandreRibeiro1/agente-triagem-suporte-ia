from pathlib import Path

import pytest

from src.base_conhecimento import BaseConhecimento
from src.classificador import ClassificadorUrgencia

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_FAQ = RAIZ / "data" / "base_conhecimento.csv"
CAMINHO_CASOS_RAG = RAIZ / "data" / "avaliacao_rag.csv"


@pytest.fixture(scope="session")
def classificador():
    clf = ClassificadorUrgencia()
    clf.treinar(verbose=False)
    return clf


@pytest.fixture
def base_conhecimento():
    return BaseConhecimento(CAMINHO_FAQ)


class ClassificadorFixo:
    """Dublê do classificador: devolve sempre a mesma urgência e conta as chamadas."""

    def __init__(self, urgencia):
        self.urgencia = urgencia
        self.chamadas = 0

    def prever(self, texto):
        self.chamadas += 1
        return self.urgencia


class BaseFixa:
    """Dublê da base de conhecimento: devolve sempre o mesmo resultado."""

    def __init__(self, resultado):
        self.resultado = resultado
        self.consultas = 0

    def gerar_resposta_automatica(self, texto):
        self.consultas += 1
        return self.resultado
