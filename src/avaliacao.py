# =============================================================================
# Módulo: Avaliação dos modelos
#
# Mede as duas peças de IA do agente com números, em vez de só mostrar
# exemplos que funcionam:
#
#   1) Classificador de urgência -> validação cruzada estratificada repetida
#      (a base tem só 32 frases, então um único split treino/teste seria
#      pouco confiável), comparada com um baseline que sempre chuta a mesma
#      classe. Gera matriz de confusão, precisão, recall e F1.
#
#   2) Retriever do RAG -> um conjunto de chamados reescritos com o item da
#      FAQ esperado (ou nenhum, para perguntas fora do escopo). Mede se o
#      agente acerta o item quando deveria responder e se recusa quando não
#      deveria, variando o limiar de similaridade.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from src.classificador import DADOS_TREINO, criar_pipeline

ROTULOS = ["urgente", "normal"]


def avaliar_classificador(dados=None, n_splits=4, n_repeats=25, random_state=42):
    """
    Validação cruzada estratificada repetida do classificador de urgência.

    Retorna um dicionário com média e desvio de acurácia e F1 macro (modelo e
    baseline), a matriz de confusão somada em todas as dobras e precisão/recall/F1
    por classe calculados sobre todas as previsões fora da dobra de treino.
    """
    dados = dados or DADOS_TREINO
    df = pd.DataFrame(dados, columns=["texto", "urgencia"])
    X, y = df["texto"].to_numpy(), df["urgencia"].to_numpy()

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    modelo_base = criar_pipeline()
    baseline_base = DummyClassifier(strategy="most_frequent")

    acc, f1, acc_base, f1_base = [], [], [], []
    y_real, y_prev = [], []

    for idx_treino, idx_teste in cv.split(X, y):
        modelo = clone(modelo_base).fit(X[idx_treino], y[idx_treino])
        previsto = modelo.predict(X[idx_teste])
        acc.append(accuracy_score(y[idx_teste], previsto))
        f1.append(f1_score(y[idx_teste], previsto, average="macro", zero_division=0))

        baseline = clone(baseline_base).fit(X[idx_treino], y[idx_treino])
        previsto_base = baseline.predict(X[idx_teste])
        acc_base.append(accuracy_score(y[idx_teste], previsto_base))
        f1_base.append(f1_score(y[idx_teste], previsto_base, average="macro", zero_division=0))

        y_real.extend(y[idx_teste])
        y_prev.extend(previsto)

    precisao, recall, f1_classe, suporte = precision_recall_fscore_support(
        y_real, y_prev, labels=ROTULOS, zero_division=0
    )
    por_classe = pd.DataFrame(
        {"precisao": precisao, "recall": recall, "f1": f1_classe, "suporte": suporte},
        index=ROTULOS,
    )

    return {
        "n_exemplos": len(df),
        "n_dobras": n_splits * n_repeats,
        "acuracia": (float(np.mean(acc)), float(np.std(acc))),
        "f1_macro": (float(np.mean(f1)), float(np.std(f1))),
        "baseline_acuracia": (float(np.mean(acc_base)), float(np.std(acc_base))),
        "baseline_f1_macro": (float(np.mean(f1_base)), float(np.std(f1_base))),
        "matriz_confusao": pd.DataFrame(
            confusion_matrix(y_real, y_prev, labels=ROTULOS),
            index=[f"real: {r}" for r in ROTULOS],
            columns=[f"previsto: {r}" for r in ROTULOS],
        ),
        "por_classe": por_classe,
    }


def carregar_casos_rag(caminho_csv):
    """Lê o CSV de avaliação: chamado + id da FAQ esperado (vazio = fora do escopo)."""
    df = pd.read_csv(caminho_csv)
    df["id_esperado"] = df["id_esperado"].astype("Int64")
    return df


def avaliar_retriever(base_conhecimento, casos, limiar):
    """
    Avalia o retriever em um limiar de similaridade.

    - acerto_no_escopo: dos chamados que TÊM resposta na FAQ, quantos o agente
      respondeu com o item certo (errar o item ou recusar contam como falha).
    - recusa_fora_escopo: dos chamados SEM resposta na FAQ, quantos o agente
      corretamente mandou para a fila humana.
    - resposta_errada: respostas automáticas dadas com o item errado (ou para
      chamado fora do escopo), sobre o total de respostas dadas. É o erro mais
      caro: o cliente recebe uma resposta que não tem nada a ver.
    """
    limiar_original = base_conhecimento.limiar_similaridade
    base_conhecimento.limiar_similaridade = limiar
    try:
        no_escopo_ok = no_escopo_total = 0
        fora_ok = fora_total = 0
        respostas = respostas_erradas = 0

        for chamado, esperado in zip(casos["chamado"], casos["id_esperado"]):
            resultado = base_conhecimento.buscar(chamado)
            id_retornado = None if resultado is None else int(resultado["id"])

            if id_retornado is not None:
                respostas += 1
                if pd.isna(esperado) or id_retornado != int(esperado):
                    respostas_erradas += 1

            if pd.isna(esperado):
                fora_total += 1
                fora_ok += id_retornado is None
            else:
                no_escopo_total += 1
                no_escopo_ok += id_retornado == int(esperado)
    finally:
        base_conhecimento.limiar_similaridade = limiar_original

    return {
        "limiar": limiar,
        "acerto_no_escopo": no_escopo_ok / no_escopo_total if no_escopo_total else 0.0,
        "recusa_fora_escopo": fora_ok / fora_total if fora_total else 0.0,
        "resposta_errada": respostas_erradas / respostas if respostas else 0.0,
        "respostas_automaticas": respostas,
    }


def varrer_limiares(base_conhecimento, casos, limiares):
    """Roda avaliar_retriever() para vários limiares e devolve um DataFrame."""
    return pd.DataFrame([avaliar_retriever(base_conhecimento, casos, l) for l in limiares])
