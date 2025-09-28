import os
import re
import psycopg2
import unicodedata
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from llm_handler import LLMHandler

# Configuração de busca vetorial
KEY_VALUE = 30

load_dotenv()

# Templates de prompt melhorados
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS IMPORTANTES:
- Responda SOMENTE com base no CONTEXTO fornecido
- Para perguntas sobre valores (maior/menor faturamento), analise NUMERICAMENTE todos os valores em reais (R$)
- Compare os números após "R$" matematicamente
- Para "maior": encontre o MAIOR valor numérico
- Para "menor": encontre o MENOR valor numérico
- Use EXATAMENTE o nome da empresa conforme aparece no contexto
- Para perguntas sobre quantidade ("quantas", "quantos"), responda com o NÚMERO total
- Se a pergunta não puder ser respondida com base no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- NUNCA invente nomes de empresas, valores ou informações
- NUNCA use conhecimento externo ao contexto fornecido

PERGUNTA: {pergunta}

RESPONDA DE FORMA PRECISA E BASEADA APENAS NO CONTEXTO:
"""

PROMPT_TEMPLATE_GEMINI = """
TAREFA: Analisar dados financeiros e responder perguntas comparativas com precisão matemática.

Você receberá uma lista de empresas com valores em reais (R$).
Sua tarefa é encontrar e responder sobre os valores conforme solicitado na pergunta.

REGRAS IMPORTANTES:
1. Compare os NÚMEROS após "R$" matematicamente
2. Para "maior": encontre o MAIOR valor numérico
3. Para "menor": encontre o MENOR valor numérico  
4. Para pergunta sobre "qual empresa" (singular): responda apenas o nome de UMA empresa
5. Para pergunta sobre "liste as" ou "quais as" (plural): forneça lista numerada
6. Para "top N" ou "N empresas": liste exatamente N empresas
7. Para perguntas sobre quantidade ("quantas", "quantos"): responda com o NÚMERO total
8. Use EXATAMENTE o nome da empresa como aparece no contexto

EXEMPLOS DE FORMATO:
- "Qual empresa tem maior faturamento?" → "Empresa X"
- "Liste as 3 empresas com maior faturamento" → "1. Empresa A\n2. Empresa B\n3. Empresa C"
- "Quais as empresas com menor faturamento?" → Lista ordenada do menor para maior
- "Quantas empresas têm X no nome?" → "N empresas" (apenas o número)

DADOS PARA ANÁLISE:
{contexto}

PERGUNTA: {pergunta}

IMPORTANTE: 
- Leia a pergunta cuidadosamente
- Para "menor": ordene do MENOR para o MAIOR valor
- Para "maior": ordene do MAIOR para o MENOR valor
- Para contagem: forneça apenas o número total

RESPOSTA:
"""

def limpar_texto(texto):
    """Remove caracteres inválidos do texto preservando acentos"""
    if not texto:
        return ""
    
    try:
        if isinstance(texto, bytes):
            texto = texto.decode('utf-8', errors='replace')
        
        # Remover surrogates problemáticos
        texto_limpo = ''.join(c for c in texto if unicodedata.category(c) != 'Cs')
        
        # Normalizar unicode (mantém acentos)
        texto_limpo = unicodedata.normalize('NFC', texto_limpo)
        
        # Remover apenas caracteres de controle perigosos
        texto_limpo = ''.join(c for c in texto_limpo 
                             if unicodedata.category(c)[0] != 'C' or c in '\n\r\t ')
        
        return texto_limpo
        
    except Exception as e:
        print(f"⚠️ Erro na limpeza do texto: {e}")
        try:
            return texto.encode('utf-8', errors='replace').decode('utf-8')
        except:
            return str(texto)

def get_embeddings():
    """Retorna embeddings baseado no DEFAULT_EMBEDDING_MODEL"""
    try:
        embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL")
        
        if not embedding_model:
            embedding_model = "text-embedding-3-small"
        
        # Detectar se é modelo OpenAI ou Google
        if embedding_model.startswith("text-embedding") or embedding_model.startswith("ada"):
            # Modelo OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and openai_key.strip("'") != "coloque aqui":
                return OpenAIEmbeddings(
                    model=embedding_model,
                    api_key=openai_key.strip("'")
                )
        
        elif embedding_model.startswith("models/embedding"):
            # Modelo Google
            google_key = os.getenv("GOOGLE_API_KEY")
            if google_key and google_key.strip("'") != "coloque aqui":
                return GoogleGenerativeAIEmbeddings(
                    model=embedding_model,
                    google_api_key=google_key.strip("'")
                )
        
        # Fallback para OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.strip("'") != "coloque aqui":
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_key.strip("'")
            )
        
        raise Exception("Nenhuma chave de embedding válida encontrada")
        
    except Exception as e:
        print(f"⚠️ Erro nos embeddings: {e}")
        return None

class SimpleVectorStore:
    """Vector store simples usando psycopg2"""
    
    def __init__(self, connection_string, embeddings, table_name="langchain_pg_embedding"):
        self.connection_string = connection_string
        self.embeddings = embeddings
        self.table_name = table_name
    
    def similarity_search(self, query, k=5):
        """Busca por similaridade usando cosine distance"""
        try:
            query_limpa = limpar_texto(query)
            query_embedding = self.embeddings.embed_query(query_limpa)
            
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT document, embedding <=> %s::vector as distance
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, k))
            
            results = cursor.fetchall()
            
            docs = []
            for doc_content, distance in results:
                doc_content_limpo = limpar_texto(doc_content)
                docs.append(Document(
                    page_content=doc_content_limpo,
                    metadata={"distance": distance}
                ))
            
            cursor.close()
            conn.close()
            
            return docs
            
        except Exception as e:
            print(f"❌ Erro na busca vetorial: {e}")
            return []

def extrair_termos_busca(pergunta):
    """Extrai termos-chave da pergunta para busca lexical"""
    pergunta = limpar_texto(pergunta)
    
    # Extrair palavras entre aspas
    termos_aspas = re.findall(r"'([^']*)'|\"([^\"]*)\"", pergunta)
    termos_aspas = [t[0] or t[1] for t in termos_aspas if (t[0] or t[1]).strip()]
    
    # Extrair substantivos importantes (palavras > 3 chars)
    stop_words = {
        'qual', 'quais', 'quantas', 'quantos', 'empresa', 'empresas', 
        'nome', 'nomes', 'tem', 'têm', 'possui', 'possuem', 'lista',
        'liste', 'mostre', 'encontre', 'busque', 'procure'
    }
    
    palavras = re.findall(r'\b\w+\b', pergunta.lower())
    termos_relevantes = [
        p for p in palavras 
        if len(p) > 3 and p not in stop_words and p.isalpha()
    ]
    
    todos_termos = termos_aspas + termos_relevantes
    return list(dict.fromkeys(todos_termos))[:5]

def preprocessar_contexto_para_comparacao(contexto, pergunta):
    """Pré-processa o contexto para perguntas comparativas com ordenação correta"""
    
    # Verificar se é pergunta sobre maior/menor valor
    termos_valor = ['maior', 'menor', 'máximo', 'mínimo', 'faturamento', 'receita', 'valor']
    eh_pergunta_valor = any(termo in pergunta.lower() for termo in termos_valor)
    
    if not eh_pergunta_valor:
        return contexto
    
    # Detectar se é pergunta sobre MENOR valor
    eh_pergunta_menor = any(termo in pergunta.lower() for termo in ['menor', 'mínimo'])
    
    # Extrair e ordenar empresas por valor
    linhas = contexto.split('\n')
    empresas_valores = []
    
    for linha in linhas:
        if 'R$' in linha and linha.strip():
            try:
                parts = linha.split('R$')
                if len(parts) >= 2:
                    nome = parts[0].strip()
                    valor_str = parts[1].split()[0]
                    
                    # Tratar formato brasileiro de números
                    if ',' in valor_str:
                        valor_str = valor_str.replace('.', '').replace(',', '.')
                    else:
                        valor_str = valor_str.replace(',', '').replace('.', '')
                    
                    valor = float(valor_str)
                    empresas_valores.append((nome, valor, linha))
            except Exception:
                continue
    
    if empresas_valores:
        # Ordenar baseado no tipo de pergunta
        if eh_pergunta_menor:
            # Para perguntas sobre MENOR: ordenar do menor para o maior
            empresas_valores.sort(key=lambda x: x[1], reverse=False)
            contexto_ordenado = "=== EMPRESAS ORDENADAS POR MENOR FATURAMENTO ===\n\n"
        else:
            # Para perguntas sobre MAIOR: ordenar do maior para o menor
            empresas_valores.sort(key=lambda x: x[1], reverse=True)
            contexto_ordenado = "=== EMPRESAS ORDENADAS POR MAIOR FATURAMENTO ===\n\n"
        
        # Destacar os top 10 no início
        for i, (nome, valor, linha_original) in enumerate(empresas_valores[:10]):
            contexto_ordenado += f"#{i+1} - {linha_original}\n"
        
        contexto_ordenado += "\n=== CONTEXTO COMPLETO ===\n\n" + contexto
        
        return contexto_ordenado
    
    return contexto

def search_prompt_hibrido(question=None, llm_handler=None):
    """Busca híbrida: vetorial + lexical com otimizações para ambos os modelos"""
    try:
        # Configurar embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return "❌ Erro: Embeddings não configurados"
    
        database_url = os.getenv("DATABASE_URL")
        vectorstore = SimpleVectorStore(database_url, embeddings)
        
        # Configurar LLM Handler
        if llm_handler is None:
            llm_handler = LLMHandler()
        
        if not llm_handler.is_available():
            return "❌ Erro: Nenhum modelo LLM disponível"
        
        if question:
            question = limpar_texto(question)
            
            # Fase 1: Busca vetorial
            docs_vetorial = vectorstore.similarity_search(question, k=KEY_VALUE)
            contexto_vetorial = "\n\n".join([doc.page_content for doc in docs_vetorial])
            
            # Fase 2: Busca lexical
            termos = extrair_termos_busca(question)
            contextos_lexicais = []
            
            if termos:
                conn = psycopg2.connect(database_url)
                cursor = conn.cursor()
                
                for termo in termos:
                    cursor.execute("""
                        SELECT DISTINCT document 
                        FROM langchain_pg_embedding 
                        WHERE LOWER(document) LIKE %s
                    """, (f'%{termo.lower()}%',))
                    
                    docs_termo = cursor.fetchall()
                    for doc in docs_termo:
                        doc_limpo = limpar_texto(doc[0])
                        contextos_lexicais.append(doc_limpo)
                
                cursor.close()
                conn.close()
            
            # Fase 3: Combinar contextos
            todos_contextos = [contexto_vetorial]
            todos_contextos.extend(contextos_lexicais)
            
            contexto_final = "\n\n".join(dict.fromkeys(todos_contextos))
            contexto_final = limpar_texto(contexto_final)

            # Pré-processamento para AMBOS os modelos em perguntas comparativas
            termos_comparacao = ['maior', 'menor', 'máximo', 'mínimo', 'top', 'ranking', 'lista']
            eh_pergunta_comparacao = any(termo in question.lower() for termo in termos_comparacao)
            
            if eh_pergunta_comparacao:
                contexto_final = preprocessar_contexto_para_comparacao(contexto_final, question)
            
            # Gerar resposta com prompt apropriado
            if llm_handler and llm_handler.get_current_model() == "gemini" and eh_pergunta_comparacao:
                prompt = PROMPT_TEMPLATE_GEMINI.format(
                    contexto=contexto_final,
                    pergunta=question
                )
            else:
                prompt = PROMPT_TEMPLATE.format(
                    contexto=contexto_final,
                    pergunta=question
                )
            
            response = llm_handler.invoke(prompt)
            return response if response else "❌ Erro: Falha na geração de resposta"
        
        return {
            'vectorstore': vectorstore,
            'llm_handler': llm_handler,
            'embeddings': embeddings
        }
        
    except Exception as e:
        print(f"❌ Erro na busca híbrida: {e}")
        return None

def search_prompt(question=None, llm_handler=None):
    """Busca vetorial simples com otimizações para ambos os modelos"""
    try:
        embeddings = get_embeddings()
        if not embeddings:
            return "❌ Erro: Embeddings não configurados"
        
        database_url = os.getenv("DATABASE_URL")
        vectorstore = SimpleVectorStore(database_url, embeddings)
        
        if llm_handler is None:
            llm_handler = LLMHandler()
        
        if not llm_handler.is_available():
            return "❌ Erro: Nenhum modelo LLM disponível"
        
        if question:
            question = limpar_texto(question)
            
            docs = vectorstore.similarity_search(question, k=KEY_VALUE)
            contexto = "\n\n".join([doc.page_content for doc in docs])
            contexto = limpar_texto(contexto)
            
            # Verificar se é pergunta comparativa
            termos_comparacao = ['maior', 'menor', 'máximo', 'mínimo', 'top', 'ranking', 'lista']
            eh_pergunta_comparacao = any(termo in question.lower() for termo in termos_comparacao)
            
            # Pré-processamento para AMBOS os modelos
            if eh_pergunta_comparacao:
                contexto = preprocessar_contexto_para_comparacao(contexto, question)

            # Usar prompt específico para Gemini em perguntas comparativas
            if llm_handler and llm_handler.get_current_model() == "gemini" and eh_pergunta_comparacao:
                prompt = PROMPT_TEMPLATE_GEMINI.format(
                    contexto=contexto,
                    pergunta=question
                )
            else:
                prompt = PROMPT_TEMPLATE.format(
                    contexto=contexto,
                    pergunta=question
                )
            
            response = llm_handler.invoke(prompt)
            return response if response else "❌ Erro: Falha na geração de resposta"
        
        return {
            'vectorstore': vectorstore,
            'llm_handler': llm_handler,
            'embeddings': embeddings
        }
        
    except Exception as e:
        print(f"❌ Erro na busca: {e}")
        return None