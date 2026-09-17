# 🤖 Agente de Triagem e Atendimento Automático de Suporte com IA

[![Testes](https://github.com/AlexandreRibeiro1/agente-triagem-suporte-ia/actions/workflows/testes.yml/badge.svg)](https://github.com/AlexandreRibeiro1/agente-triagem-suporte-ia/actions/workflows/testes.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Licença MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green)

> **🇺🇸 In English:** an AI agent that triages customer-support tickets. A
> TF-IDF + Naive Bayes classifier flags urgent tickets for escalation; normal
> tickets go through a local retrieval step (TF-IDF + cosine similarity over an
> FAQ) and are answered automatically only when the match is confident enough,
> otherwise they go to a human queue. Both models are evaluated with repeated
> stratified cross-validation, a baseline, a confusion matrix and a
> similarity-threshold sweep ([`reports/metricas.md`](reports/metricas.md)).
> Runs 100% locally, no paid API. Tested with pytest and GitHub Actions.

Evolução do projeto desenvolvido como desafio final da **Trilha Especialista
em Inteligência Artificial (Alura, com bolsa via Santander Academy)**.

Na v1, o agente apenas classificava a urgência de um chamado e decidia entre
escalar ou enfileirar. Na v2, o agente ganha uma capacidade real de
**resolver chamados sozinho**, consultando uma base de conhecimento (RAG),
sem depender de nenhuma API paga, e os dois modelos passam a ser **medidos**,
não só demonstrados.

## 🎯 O problema que este projeto resolve

Times de suporte gastam grande parte do tempo respondendo perguntas
repetitivas (senha, prazo de entrega, cancelamento, planos) que já têm
resposta pronta em uma FAQ, mas que continuam consumindo atendimento humano
porque ninguém automatiza a busca. Este agente resolve automaticamente os
chamados simples e só escala para humano o que realmente precisa de atenção
(urgências e dúvidas sem resposta conhecida).

## 🧠 Como funciona

O projeto é dividido em três camadas:

1. **Classificador de urgência** (`src/classificador.py`): pipeline
   `TF-IDF + Multinomial Naive Bayes` (scikit-learn) que classifica o
   chamado como `urgente` ou `normal`.
2. **Base de conhecimento / RAG local** (`src/base_conhecimento.py`):
   implementa a etapa de *Retrieval* de um RAG. Vetoriza uma FAQ fictícia via
   TF-IDF e busca o item mais similar ao chamado por similaridade de
   cosseno. A "geração" da resposta é feita por template a partir do item
   recuperado. É a mesma arquitetura de um RAG com LLM, mas sem custo de API.
3. **Agente de IA** (`src/agente.py`): a classe `AgenteTriagemSuporte`
   segue o ciclo *perceber → decidir → agir*, com **três** desfechos:
   - 🚨 **Urgente** → escalar imediatamente para o time de plantão.
   - ✅ **Normal + boa correspondência na FAQ** → responder automaticamente.
   - 📋 **Normal + sem correspondência confiável** → encaminhar para fila
     humana (o agente reconhece os próprios limites em vez de "chutar"
     uma resposta).

`main.py` treina o classificador, carrega a FAQ, roda o agente sobre chamados
simulados e gera um relatório. `avaliar.py` mede os modelos.

## 📏 Avaliação dos modelos

Relatório completo, gerado por `python avaliar.py`: **[reports/metricas.md](reports/metricas.md)**.

**Classificador de urgência**: validação cruzada estratificada repetida
(4 dobras × 25 repetições), porque com 32 exemplos um único split treino/teste
deixaria só 8 frases no teste.

| Modelo | Acurácia | F1 macro |
|---|---|---|
| TF-IDF + Naive Bayes | 83% ± 11% | 83% ± 11% |
| Baseline (sempre a mesma classe) | 50% | 33% |

Recall da classe `urgente`: **82%**. É a métrica que mais importa, porque um
urgente classificado como normal fica parado na fila.

**Retriever do RAG**: 37 chamados anotados à mão
(`data/avaliacao_rag.csv`): 29 com resposta na FAQ, escritos de outro jeito, e
8 fora do escopo. A varredura de limiar mostrou que o limiar antigo (0,30)
dava **23% de respostas automáticas erradas**. Por exemplo, "integração com
folha de pagamento" recebia a resposta sobre cartão de pagamento. Com **0,50**,
o acerto no escopo continua em 69%, a recusa fora do escopo sobe para 100% e
as respostas erradas caem para **5%**. O agente passou a usar 0,50.

## 📂 Estrutura do projeto

```
agente-triagem-suporte-ia/
├── .github/workflows/testes.yml   # CI: pytest + avaliação a cada push
├── data/
│   ├── base_conhecimento.csv      # FAQ fictícia usada pelo RAG
│   ├── avaliacao_rag.csv          # Chamados anotados para avaliar o retriever
│   └── relatorio_triagem_exemplo.csv
├── reports/
│   └── metricas.md                # Gerado por avaliar.py
├── src/
│   ├── classificador.py           # Classificador de urgência (TF-IDF + NB)
│   ├── base_conhecimento.py       # Retriever do RAG (TF-IDF + cosseno)
│   ├── agente.py                  # Agente (perceber -> decidir -> agir)
│   └── avaliacao.py               # Validação cruzada e varredura de limiar
├── tests/                         # Testes com pytest
├── main.py                        # Demonstração do pipeline completo
├── avaliar.py                     # Gera reports/metricas.md
├── requirements.txt
└── requirements-dev.txt
```

## 🚀 Como executar

```bash
git clone https://github.com/AlexandreRibeiro1/agente-triagem-suporte-ia.git
cd agente-triagem-suporte-ia

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements-dev.txt

python main.py      # demonstração
python avaliar.py   # métricas dos modelos
pytest              # testes
```

## 📊 Exemplo de saída

```
📨 Chamado: Como faço para trocar minha senha de acesso?
   → Urgência prevista: NORMAL
   → Ação do agente: ✅ Resolvido automaticamente pelo agente (RAG)
   → Similaridade com a FAQ: 0.5
   → Resposta automática:
   Acesse Configurações > Segurança > Alterar senha (...)

📨 Chamado: Meu cachorro comeu meu carregador de notebook, o que eu faço?
   → Urgência prevista: NORMAL
   → Ação do agente: 📋 Encaminhado para a fila humana (sem correspondência confiável na FAQ)
```

O relatório da demonstração (`data/relatorio_triagem_exemplo.csv`) traz, para
cada chamado: urgência prevista, ação tomada, origem da resposta (RAG,
escalonamento ou fila humana), score de similaridade e a resposta automática
(quando houver). O agente também reporta a **taxa de resolução automática**, o
KPI mais direto para mostrar o ganho de eficiência para o negócio.

## ⚠️ Limitações conhecidas

- **Base de treino pequena e fictícia** (32 frases). O desvio de ±11% na
  acurácia mostra o quanto o resultado depende de quais frases caem no teste.
  Em produção seria preciso um histórico real de chamados rotulados.
- **O classificador aprende palavras, não contexto.** "Pagamento" só aparece
  em chamados urgentes no treino, então "Vocês têm integração com folha de
  pagamento?" é marcado como urgente. O erro aparece de propósito na
  demonstração do `main.py`.
- **O retriever usa TF-IDF, não embeddings.** Chamados escritos só com
  sinônimos ("encerrar meu plano" x "cancelar assinatura") não casam com a
  FAQ, e isso limita o acerto no escopo a 69%.
- **O limiar foi escolhido no próprio conjunto de avaliação**, que é pequeno.
  Os números servem para comparar opções, não como estimativa de produção.

## 📈 Possíveis evoluções

- **Trocar TF-IDF por embeddings semânticos** (ex.: `sentence-transformers`)
  e comparar com a mesma varredura de `avaliar.py`.
- **Trocar o template de resposta por um LLM real**, usando os itens
  recuperados da FAQ como contexto, e medir se a resposta gerada é fiel à FAQ.
- Treinar o classificador com uma base pública de chamados rotulados.
- Conectar o agente a um canal real (e-mail, Slack, WhatsApp).
- Dashboard (Streamlit) acompanhando a taxa de resolução automática ao longo
  do tempo.

## 🛠️ Tecnologias

- Python 3
- pandas
- scikit-learn (TF-IDF, Naive Bayes, similaridade de cosseno, validação cruzada, métricas)
- pytest + GitHub Actions

## 👤 Autor

Alexandre Ribeiro, estudante de Ciência da Computação (UNIFACS) e Analista de
Dados/Automação.

---
Projeto evoluído a partir do desafio final da **Trilha Especialista em
Inteligência Artificial** da Alura, em parceria com o Santander Academy.
