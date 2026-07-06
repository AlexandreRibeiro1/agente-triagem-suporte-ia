# 🤖 Agente de Triagem Automática de Chamados de Suporte com IA

Projeto desenvolvido como **desafio final da Trilha Especialista em Inteligência
Artificial (Alura, com bolsa via Santander Academy)**, aplicando Machine
Learning, lógica de Agentes de IA e Python para automatizar um processo real:
a triagem e priorização de chamados de suporte.

## 🎯 Objetivo

Reduzir o tempo de resposta a chamados críticos e organizar o backlog de
atendimento automaticamente, sem depender de triagem manual, usando um
modelo de IA treinado para reconhecer o nível de urgência de cada chamado.

## 🧠 Como funciona

O projeto é dividido em três partes:

1. **Machine Learning** — um pipeline `TF-IDF + Multinomial Naive Bayes`
   (scikit-learn) é treinado com um conjunto de chamados de suporte já
   rotulados como `urgente` ou `normal`.
2. **Agente de IA** — a classe `AgenteTriagemSuporte` implementa o ciclo
   clássico de um agente inteligente:
   - **Perceber**: recebe o texto do novo chamado e usa o modelo de ML para
     classificar a urgência.
   - **Decidir**: define a ação apropriada com base na urgência prevista.
   - **Agir**: executa a ação (escalar ou enfileirar) e registra tudo em log.
3. **Automação/Relatório** — ao final do processamento, o agente gera
   automaticamente um relatório (`.csv`) com o resumo da triagem — no mesmo
   espírito de um painel de KPIs.

## 🚀 Como executar

```bash
# Clonar o repositório
git clone https://github.com/SEU_USUARIO/agente-triagem-suporte-ia.git
cd agente-triagem-suporte-ia

# Instalar dependências
pip install -r requirements.txt

# Executar
python agente_triagem_ia.py
```

Também é possível rodar direto no **Google Colab**, colando o conteúdo de
`agente_triagem_ia.py` em uma célula.

## 📊 Exemplo de saída

```
📨 Chamado: O site caiu para todos os clientes, ninguém consegue comprar
   → Urgência prevista: URGENTE
   → Ação do agente: 🚨 ESCALAR IMEDIATAMENTE para o time de plantão

📨 Chamado: Como faço para trocar meu endereço de entrega?
   → Urgência prevista: NORMAL
   → Ação do agente: 📋 Adicionar à fila normal de atendimento
```

O relatório final gerado (`relatorio_triagem_exemplo.csv`) traz cada chamado
processado, a urgência prevista pelo modelo, a ação tomada pelo agente e o
timestamp de processamento.

## 🛠️ Tecnologias

- Python 3
- pandas
- scikit-learn (TF-IDF, Naive Bayes, métricas de avaliação)

## 📈 Possíveis evoluções

- Treinar o modelo com uma base real de chamados (histórico de CRM/Zendesk).
- Adicionar um segundo classificador de sentimento (positivo/negativo).
- Conectar o agente a um canal real (e-mail, Slack, WhatsApp) para agir de
  forma automática em produção.
- Adicionar um dashboard (Streamlit ou Power BI) sobre o relatório gerado.

## 👤 Autor

Alexandre (Ale) — Estudante de Ciência da Computação (UNIFACS) e Analista de
Dados/Automação com experiência em KPIs e dashboards (Praxis Empresa Júnior).

---
Projeto feito como parte da **Trilha Especialista em Inteligência Artificial**
da Alura, em parceria com o Santander Academy.
