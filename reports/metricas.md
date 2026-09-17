# Métricas do agente

> Arquivo gerado por `python avaliar.py`. Não edite à mão.

## 1. Classificador de urgência (TF-IDF + Naive Bayes)

Validação cruzada estratificada repetida: 32 chamados rotulados,
100 dobras (4 dobras × 25 repetições). Com uma base tão pequena,
um único split treino/teste deixaria só 8 frases no teste; repetir o sorteio
dá uma estimativa bem mais estável.

| Modelo | Acurácia | F1 macro |
|---|---|---|
| TF-IDF + Naive Bayes | 83% ± 11% | 83% ± 11% |
| Baseline (sempre a classe mais frequente) | 50% ± 0% | 33% ± 0% |

**Matriz de confusão** (somada em todas as dobras):

|  | previsto: urgente | previsto: normal |
|---|---|---|
| real: urgente | 330 | 70 |
| real: normal | 63 | 337 |

**Por classe:**

|  | Precisão | Recall | F1 | Previsões avaliadas |
|---|---|---|---|---|
| urgente | 84% | 82% | 83% | 400 |
| normal | 83% | 84% | 84% | 400 |

O erro mais caro aqui é um chamado **urgente previsto como normal** (fica na
fila em vez de ir para o plantão), por isso o recall da classe `urgente` é a
métrica a acompanhar.

## 2. Retriever do RAG (TF-IDF + similaridade de cosseno)

Conjunto de avaliação em `data/avaliacao_rag.csv`: 29 chamados que
têm resposta na FAQ, escritos de outro jeito (parte reaproveita palavras da
pergunta original, parte usa só sinônimos), com o item certo anotado, e
8 chamados sem resposta na FAQ.

- **Acerto no escopo:** chamados com resposta na FAQ que receberam o item certo.
- **Recusa fora do escopo:** chamados sem resposta que foram para a fila humana.
- **Respostas erradas:** das respostas automáticas enviadas, quantas usaram o
  item errado. É o erro que o cliente vê.

| Limiar | Acerto no escopo | Recusa fora do escopo | Respostas erradas | Respostas automáticas |
|---|---|---|---|---|
| 0.10 | 69% | 88% | 23% | 26 |
| 0.20 | 69% | 88% | 23% | 26 |
| 0.30 | 69% | 88% | 23% | 26 |
| 0.40 | 69% | 88% | 20% | 25 |
| **0.50** (em uso) | 69% | 100% | 5% | 21 |
| 0.60 | 62% | 100% | 5% | 19 |

Melhor limiar nesta varredura: **0.50**, com
5% de respostas erradas e
69% de acerto no escopo.

Limiar baixo responde mais, mas erra mais; limiar alto erra menos, mas manda
mais chamados para humano. O acerto no escopo não passa de um teto porque o
retriever usa TF-IDF (palavras, não significado): chamados escritos só com
sinônimos ("encerrar meu plano" x "cancelar assinatura") não casam com a FAQ.
Esse teto é a principal motivação para trocar o TF-IDF por embeddings
semânticos. Ressalva: o conjunto de avaliação é pequeno e o limiar foi
escolhido nele mesmo, então os números servem para comparar opções, não como
estimativa de produção.
