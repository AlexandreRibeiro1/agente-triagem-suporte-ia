# =============================================================================
# Avaliação do agente: gera reports/metricas.md com as métricas do
# classificador de urgência e do retriever do RAG.
#
# Uso:  python avaliar.py
# =============================================================================

from pathlib import Path

from src.avaliacao import (
    avaliar_classificador,
    carregar_casos_rag,
    varrer_limiares,
)
from src.base_conhecimento import LIMIAR_PADRAO, BaseConhecimento

RAIZ = Path(__file__).resolve().parent
CAMINHO_BASE_CONHECIMENTO = RAIZ / "data" / "base_conhecimento.csv"
CAMINHO_CASOS_RAG = RAIZ / "data" / "avaliacao_rag.csv"
CAMINHO_RELATORIO = RAIZ / "reports" / "metricas.md"

# Limiar usado pelo agente (fica destacado na tabela)
LIMIAR_EM_USO = LIMIAR_PADRAO
LIMIARES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]


def pct(valor):
    return f"{valor:.0%}"


def media_desvio(par):
    media, desvio = par
    return f"{media:.0%} ± {desvio:.0%}"


def tabela_markdown(df, indice=True):
    """Converte um DataFrame em tabela markdown sem depender do pacote tabulate."""
    colunas = ([""] if indice else []) + [str(c) for c in df.columns]
    linhas = ["| " + " | ".join(colunas) + " |", "|" + "---|" * len(colunas)]
    for rotulo, linha in df.iterrows():
        celulas = ([str(rotulo)] if indice else []) + [str(v) for v in linha]
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas)


def main():
    clf = avaliar_classificador()

    por_classe = clf["por_classe"].copy()
    for coluna in ["precisao", "recall", "f1"]:
        por_classe[coluna] = por_classe[coluna].map(pct)
    por_classe.columns = ["Precisão", "Recall", "F1", "Previsões avaliadas"]

    base = BaseConhecimento(CAMINHO_BASE_CONHECIMENTO, limiar_similaridade=LIMIAR_EM_USO)
    casos = carregar_casos_rag(CAMINHO_CASOS_RAG)
    limiares = varrer_limiares(base, casos, LIMIARES)

    tabela_limiares = limiares.copy()
    tabela_limiares["limiar"] = [
        f"**{l:.2f}** (em uso)" if abs(l - LIMIAR_EM_USO) < 1e-9 else f"{l:.2f}"
        for l in tabela_limiares["limiar"]
    ]
    for coluna in ["acerto_no_escopo", "recusa_fora_escopo", "resposta_errada"]:
        tabela_limiares[coluna] = tabela_limiares[coluna].map(pct)
    tabela_limiares.columns = [
        "Limiar", "Acerto no escopo", "Recusa fora do escopo",
        "Respostas erradas", "Respostas automáticas",
    ]

    # Melhor limiar: menor taxa de respostas erradas; empate -> maior acerto no escopo
    melhor = limiares.sort_values(
        ["resposta_errada", "acerto_no_escopo"], ascending=[True, False]
    ).iloc[0]

    n_no_escopo = int(casos["id_esperado"].notna().sum())
    n_fora = int(casos["id_esperado"].isna().sum())

    relatorio = f"""# Métricas do agente

> Arquivo gerado por `python avaliar.py`. Não edite à mão.

## 1. Classificador de urgência (TF-IDF + Naive Bayes)

Validação cruzada estratificada repetida: {clf['n_exemplos']} chamados rotulados,
{clf['n_dobras']} dobras (4 dobras × 25 repetições). Com uma base tão pequena,
um único split treino/teste deixaria só 8 frases no teste; repetir o sorteio
dá uma estimativa bem mais estável.

| Modelo | Acurácia | F1 macro |
|---|---|---|
| TF-IDF + Naive Bayes | {media_desvio(clf['acuracia'])} | {media_desvio(clf['f1_macro'])} |
| Baseline (sempre a classe mais frequente) | {media_desvio(clf['baseline_acuracia'])} | {media_desvio(clf['baseline_f1_macro'])} |

**Matriz de confusão** (somada em todas as dobras):

{tabela_markdown(clf['matriz_confusao'])}

**Por classe:**

{tabela_markdown(por_classe)}

O erro mais caro aqui é um chamado **urgente previsto como normal** (fica na
fila em vez de ir para o plantão), por isso o recall da classe `urgente` é a
métrica a acompanhar.

## 2. Retriever do RAG (TF-IDF + similaridade de cosseno)

Conjunto de avaliação em `data/avaliacao_rag.csv`: {n_no_escopo} chamados que
têm resposta na FAQ, escritos de outro jeito (parte reaproveita palavras da
pergunta original, parte usa só sinônimos), com o item certo anotado, e
{n_fora} chamados sem resposta na FAQ.

- **Acerto no escopo:** chamados com resposta na FAQ que receberam o item certo.
- **Recusa fora do escopo:** chamados sem resposta que foram para a fila humana.
- **Respostas erradas:** das respostas automáticas enviadas, quantas usaram o
  item errado. É o erro que o cliente vê.

{tabela_markdown(tabela_limiares, indice=False)}

Melhor limiar nesta varredura: **{melhor['limiar']:.2f}**, com
{pct(melhor['resposta_errada'])} de respostas erradas e
{pct(melhor['acerto_no_escopo'])} de acerto no escopo.

Limiar baixo responde mais, mas erra mais; limiar alto erra menos, mas manda
mais chamados para humano. O acerto no escopo não passa de um teto porque o
retriever usa TF-IDF (palavras, não significado): chamados escritos só com
sinônimos ("encerrar meu plano" x "cancelar assinatura") não casam com a FAQ.
Esse teto é a principal motivação para trocar o TF-IDF por embeddings
semânticos. Ressalva: o conjunto de avaliação é pequeno e o limiar foi
escolhido nele mesmo, então os números servem para comparar opções, não como
estimativa de produção.
"""

    CAMINHO_RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_RELATORIO.write_text(relatorio, encoding="utf-8")
    print(relatorio)
    print(f"Relatório salvo em: {CAMINHO_RELATORIO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
