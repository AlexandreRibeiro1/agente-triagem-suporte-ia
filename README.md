🎫 Agente de Triagem e Atendimento Automático de Suporte com IA

Agente de Inteligência Artificial que classifica chamados de suporte técnico por urgência e decide automaticamente a ação a ser tomada. Chamados urgentes são escalados imediatamente para o time de plantão; chamados normais passam por uma segunda camada de IA que tenta resolvê-los sozinha consultando uma base de conhecimento (RAG), só encaminhando para a fila humana quando não há confiança suficiente na resposta.

Evolução do desafio final da Trilha Especialista em Inteligência Artificial (Alura / Santander Academy).

🧠 Como funciona

O agente segue um ciclo de perceber → decidir → agir, com três desfechos possíveis:

Classificação de urgência: o texto do chamado é vetorizado com TF-IDF e classificado por um modelo Multinomial Naive Bayes como urgente ou normal.

Chamado urgente → 🚨 escalado imediatamente para o time de plantão, sem tentativa de resposta automática.

Chamado normal → o agente busca, via RAG local (TF-IDF + similaridade de cosseno), o item mais próximo em uma base de conhecimento (FAQ):
Se a similaridade ultrapassa o limiar de confiança (padrão: 0.30) → ✅ o agente responde automaticamente, com base no template da resposta encontrada.
Se não há correspondência confiável → 📋 o chamado é encaminhado para a fila humana.

O RAG roda 100% local com scikit-learn (sem custo de API): o retriever é o mesmo de um RAG "completo" com LLM — bastaria trocar o formatador de resposta por template por uma chamada a um LLM (ver seção Possíveis evoluções).

🛠️ Tecnologias

Python

scikit-learn — TfidfVectorizer, MultinomialNB, cosine_similarity

pandas — manipulação de dados e geração do relatório final

📁 Estrutura do projeto

agente-triagem-suporte-ia/

├── data/

│   ├── base_conhecimento.csv      # FAQ usada pelo RAG (categoria, pergunta, resposta)

│   └── relatorio_triagem.csv      # Log de saída gerado a cada execução

├── src/

│   ├── __init__.py

│   ├── classificador.py           # Pipeline TF-IDF + Naive Bayes (urgência)

│   ├── base_conhecimento.py       # RAG local: busca por similaridade na FAQ

│   └── agente.py                  # Orquestra classificador + RAG e gera o relatório

├── main.py                        # Ponto de entrada: roda o agente e imprime os resultados

├── requirements.txt

└── README.md

▶️ Como executar
bash
# Clonar o repositório
git clone https://github.com/AlexandreRibeiro1/agente-triagem-suporte-ia.git
cd agente-triagem-suporte-ia

# Instalar dependências
pip install -r requirements.txt

# Rodar o agente
python main.py

O script treina o classificador, carrega a base de conhecimento, processa 10 chamados de teste (nunca vistos pelo modelo) e salva o relatório final em data/relatorio_triagem.csv.

📊 Resultados

Execução de exemplo com 10 chamados novos:

Métrica	Valor
Chamados classificados como urgente	4
Resolvidos automaticamente via RAG	5
Encaminhados para fila humana	1
Taxa de resolução automática	50%

Ou seja: de todos os chamados não urgentes, o agente resolveu sozinho a maioria sem intervenção humana — reduzindo a fila e liberando o time de suporte para os casos que realmente precisam de atenção.

Exemplos reais de decisão do agente:

"O site caiu para todos os clientes, ninguém consegue comprar" → 🚨 urgente, escalado

"Qual o prazo para meu pedido chegar?" → ✅ resolvido automaticamente (similaridade 0.835)

"Não recebi o e-mail de confirmação do meu cadastro" → ✅ resolvido automaticamente (similaridade 1.0)

"Meu cachorro comeu meu carregador de notebook, o que eu faço?" → 📋 sem correspondência confiável, encaminhado para humano

A acurácia do classificador de urgência é impressa no console a cada execução (treinar(), em classificador.py), junto com o classification_report completo.

⚠️ Limitações conhecidas
O classificador de urgência é treinado com um dataset pequeno e sintético (DADOS_TREINO, em classificador.py) — bom para demonstrar a arquitetura, mas precisaria de dados históricos reais para uso em produção.

Sem uma lista de stopwords em português, palavras muito comuns (ex: "meu", "faço", "como") geravam falsos positivos de similaridade em chamados sem relação real com a FAQ — corrigido com uma lista de stopwords customizada em base_conhecimento.py.

A "geração" da resposta automática é feita por template (não por um LLM), o que limita a naturalidade da resposta, mas elimina custo de API.

💡 Possíveis evoluções

Trocar o Naive Bayes por modelos mais robustos (SVM, Random Forest) e comparar performance

Substituir o formatador de template por uma chamada real a um LLM, mantendo o mesmo retriever (RAG completo)

Expor o agente como API (FastAPI/Flask) para integração com um helpdesk real (Zendesk, Freshdesk etc.)

Retraining automático do classificador conforme novos chamados são rotulados

👤 Autor

Alexandre Ribeiro LinkedIn · GitHub
