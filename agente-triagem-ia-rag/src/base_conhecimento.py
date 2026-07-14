# =============================================================================
# Módulo: Base de Conhecimento (RAG local, sem custo de API)
#
# Implementa a etapa de "Retrieval" de um RAG (Retrieval-Augmented Generation)
# usando TF-IDF + similaridade de cosseno sobre uma base de FAQ. Não depende
# de nenhuma API paga: roda 100% local com scikit-learn.
#
# A "geração" da resposta é feita por template: a resposta do item da FAQ
# mais similar é formatada com uma introdução, simulando uma resposta
# gerada automaticamente pelo agente. Isso mantém o projeto sem custo,
# mas a arquitetura é a mesma de um RAG com LLM — o retriever é o mesmo;
# bastaria trocar o formatador de template por uma chamada a um LLM
# (ver seção "Possíveis evoluções" no README).
# =============================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stopwords básicas em português. Sem isso, palavras muito comuns (ex: "meu",
# "faço", "como") geram falsos positivos de similaridade em chamados que não
# têm nada a ver com a FAQ — foi um bug real encontrado ao testar este
# projeto (ver README, seção "Limitações conhecidas").
STOPWORDS_PT = [
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "faço", "faz", "fazer", "gostaria", "isso", "já", "meu",
    "meus", "minha", "minhas", "na", "nao", "não", "no", "nos", "o", "os",
    "para", "pode", "poderia", "poderiam", "por", "qual", "quero", "se",
    "sem", "seu", "seus", "sua", "suas", "tem", "um", "uma", "vocês",
    "que", "é", "são", "ser", "estou", "está", "estão", "ter", "há", "só",
    "mas", "ou", "todo", "toda", "todos", "todas", "outro", "outra", "vou",
    "sobre", "sou", "me", "minha", "seus", "algum", "alguma", "algo",
]


class BaseConhecimento:
    """Base de FAQ consultável por similaridade semântica (TF-IDF + cosseno)."""

    def __init__(self, caminho_csv, limiar_similaridade=0.30):
        self.df = pd.read_csv(caminho_csv)
        self.limiar_similaridade = limiar_similaridade

        self.vectorizer = TfidfVectorizer(stop_words=STOPWORDS_PT)
        self._matriz_perguntas = self.vectorizer.fit_transform(self.df["pergunta"])

    def buscar(self, texto_chamado):
        """
        Busca o item da FAQ mais similar ao texto do chamado.

        Retorna um dicionário com a pergunta/resposta encontradas e o score
        de similaridade, ou None se nenhum item ultrapassar o limiar mínimo
        (ou seja: o agente não tem confiança suficiente para responder sozinho).
        """
        vetor_chamado = self.vectorizer.transform([texto_chamado])
        similaridades = cosine_similarity(vetor_chamado, self._matriz_perguntas)[0]

        indice_melhor = similaridades.argmax()
        score = similaridades[indice_melhor]

        if score < self.limiar_similaridade:
            return None

        item = self.df.iloc[indice_melhor]
        return {
            "categoria": item["categoria"],
            "pergunta_faq": item["pergunta"],
            "resposta": item["resposta"],
            "similaridade": round(float(score), 3),
        }

    def gerar_resposta_automatica(self, texto_chamado):
        """
        Gera (via template) uma resposta automática para o chamado, com base
        no item mais similar da FAQ. Retorna None se não houver correspondência
        confiável — nesse caso, o chamado deve seguir para atendimento humano.
        """
        resultado = self.buscar(texto_chamado)
        if resultado is None:
            return None

        resposta_formatada = (
            f"Olá! Encontramos uma resposta que pode ajudar com sua dúvida:\n\n"
            f"{resultado['resposta']}\n\n"
            f"Se isso não resolver completamente, responda este chamado que um "
            f"atendente humano dará continuidade."
        )
        resultado["resposta_automatica"] = resposta_formatada
        return resultado
