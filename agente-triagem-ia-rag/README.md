# 🤖 Agente de Triagem e Atendimento Automático de Suporte com IA (v2 — RAG)

Evolução do projeto desenvolvido como desafio final da **Trilha Especialista
em Inteligência Artificial (Alura, com bolsa via Santander Academy)**.

Na v1, o agente apenas classificava a urgência de um chamado e decidia entre
escalar ou enfileirar. Nesta v2, o agente ganha uma capacidade real de
**resolver chamados sozinho**, consultando uma base de conhecimento (RAG),
sem depender de nenhuma API paga.

## 🎯 O problema que este projeto resolve

Times de suporte gastam grande parte do tempo respondendo perguntas
repetitivas (senha, prazo de entrega, cancelamento, planos) que já têm
resposta pronta em uma FAQ — mas que continuam consumindo atendimento humano
porque ninguém automatiza a busca. Este agente resolve automaticamente os
chamados simples e só escala para humano o que realmente precisa de atenção
(urgências e dúvidas sem resposta conhecida), liberando o time para o que
importa.

## 🧠 Como funciona

O projeto é dividido em três camadas:

1. **Classificador de urgência** (`src/classificador.py`) — pipeline
   `TF-IDF + Multinomial Naive Bayes` (scikit-learn) que classifica o
   chamado como `urgente` ou `normal`.
2. **Base de conhecimento / RAG local** (`src/base_conhecimento.py`) —
   implementa a etapa de *Retrieval* de um RAG: vetoriza uma FAQ fictícia via
   TF-IDF e busca o item mais similar ao chamado por similaridade de
   cosseno. A "geração" da resposta é feita por template a partir do item
   recuperado — mesma arquitetura de um RAG com LLM, mas sem custo de API
   (ver seção de evoluções abaixo para o caminho de trocar o template por um
   LLM real).
3. **Agente de IA** (`src/agente.py`) — a classe `AgenteTriagemSuporte`
   segue o ciclo clássico *perceber → decidir → agir*, agora com **três**
   desfechos possíveis por chamado:
   - 🚨 **Urgente** → escalar imediatamente para o time de plantão.
   - ✅ **Normal + boa correspondência na FAQ** → responder automaticamente.
   - 📋 **Normal + sem correspondência confiável** → encaminhar para fila
     humana (o agente reconhece os próprios limites em vez de "chutar"
     uma resposta).

Tudo isso é orquestrado em `main.py`, que treina o classificador, carrega a
FAQ, roda o agente sobre chamados simulados e gera um relatório final.

## 📂 Estrutura do projeto

```
agente-triagem-ia-rag/
├── data/
│   ├── base_conhecimento.csv       # FAQ fictícia usada pelo RAG
│   └── relatorio_triagem_exemplo.csv
├── src/
│   ├── classificador.py            # Classificador de urgência (TF-IDF + NB)
│   ├── base_conhecimento.py        # Retriever do RAG (TF-IDF + cosseno)
│   └── agente.py                   # Agente (perceber -> decidir -> agir)
├── main.py                         # Orquestra o pipeline completo
├── requirements.txt
└── README.md
```

## 🚀 Como executar

```bash
git clone https://github.com/SEU_USUARIO/agente-triagem-ia-rag.git
cd agente-triagem-ia-rag

pip install -r requirements.txt

python main.py
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

O relatório final (`data/relatorio_triagem_exemplo.csv`) traz, para cada
chamado: urgência prevista, ação tomada, origem da resposta (RAG,
escalonamento ou fila humana), score de similaridade e a resposta automática
gerada (quando houver). O agente também reporta a **taxa de resolução
automática** — o KPI mais direto para mostrar o ganho de eficiência para o
negócio.

## ⚠️ Limitações conhecidas (e por que valem a pena mencionar)

- O classificador de urgência é treinado com um dataset de exemplo pequeno
  (32 frases). Isso é suficiente para demonstrar o pipeline, mas em produção
  precisaria de um histórico real de chamados rotulados.
- O retriever usa TF-IDF puro (contagem de palavras), não embeddings
  semânticos. Isso significa que ele reconhece bem sinônimos textuais
  (palavras parecidas), mas não entende significado — por exemplo, um
  chamado sobre "integração de sistemas" pode casar parcialmente com uma FAQ
  sobre "pagamento" só porque compartilham uma palavra. É uma limitação real
  e documentada, não escondida.
- Por isso o agente sempre expõe o **score de similaridade** e tem um
  **limiar mínimo de confiança** — chamados abaixo do limiar vão para fila
  humana em vez de receber uma resposta arriscada.

## 📈 Possíveis evoluções

- **Trocar TF-IDF por embeddings semânticos** (ex: `sentence-transformers`)
  para melhorar a qualidade da busca — mesmo pipeline de RAG, retriever mais
  forte.
- **Trocar o template de resposta por um LLM real** (ex: API da Anthropic),
  usando os itens recuperados da FAQ como contexto — isso transforma o
  "RAG por template" em um RAG completo com geração de linguagem natural.
- Treinar o classificador de urgência com uma base real de chamados
  (histórico de CRM/Zendesk).
- Conectar o agente a um canal real (e-mail, Slack, WhatsApp) para agir de
  forma automática em produção.
- Adicionar um dashboard (Streamlit ou Power BI) sobre o relatório gerado,
  acompanhando a taxa de resolução automática ao longo do tempo.

## 🛠️ Tecnologias

- Python 3
- pandas
- scikit-learn (TF-IDF, Naive Bayes, similaridade de cosseno, métricas)

## 👤 Autor

Alexandre (Ale) — Estudante de Ciência da Computação (UNIFACS) e Analista de
Dados/Automação com experiência em KPIs, dashboards e automação de
processos (Praxis Empresa Júnior).

---
Projeto evoluído a partir do desafio final da **Trilha Especialista em
Inteligência Artificial** da Alura, em parceria com o Santander Academy.
