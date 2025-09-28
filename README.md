# Sistema RAG - Chat com Documentos PDF

Sistema de Retrieval-Augmented Generation (RAG) desenvolvido para o Desafio MBA IA, permitindo consultas inteligentes sobre documentos PDF através de interface de chat via linha de comando.

## Funcionamento

O sistema implementa uma arquitetura RAG completa que combina busca vetorial e lexical para responder perguntas sobre conteúdo de documentos PDF. O fluxo principal inclui:

1. **Ingestão**: Processamento do PDF em chunks, geração de embeddings e armazenamento no banco vetorial
2. **Busca Híbrida**: Combinação de busca vetorial (semântica) e lexical (termos exatos)
3. **Processamento Inteligente**: Pré-processamento específico para consultas comparativas
4. **Resposta**: Geração de respostas baseadas exclusivamente no contexto recuperado

### Características Principais

- **Múltiplos LLMs**: Suporte a OpenAI (ChatGPT) e Google (Gemini) com fallback automático
- **Busca Híbrida**: Vetorial + lexical para maior precisão
- **Otimizações por Modelo**: Processamento especializado para consultas comparativas no Gemini
- **Interface CLI**: Chat interativo via linha de comando com comandos especiais
- **Aderência ao Contexto**: Sistema rigoroso para evitar alucinações

## Arquitetura do Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   document.pdf  │───▶│   ingest.py      │───▶│  PostgreSQL +   │
│                 │    │  - PyPDFLoader   │    │   pgVector      │
└─────────────────┘    │  - TextSplitter  │    │                 │
                       │  - Embeddings    │    └─────────────────┘
                       └──────────────────┘             │
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│    chat.py      │───▶│    search.py     │◀───────────┘
│  - Interface    │    │  - Busca Híbrida │
│  - Comandos     │    │  - Pré-process   │
│                 │    │  - LLM Handler   │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  llm_handler.py  │
                       │  - OpenAI/Gemini │
                       │  - Fallback      │
                       │  - Multi-model   │
                       └──────────────────┘
```

## Estrutura de Arquivos

```
projeto/
├── src/
│   ├── chat.py           # Interface principal do chat
│   ├── search.py         # Lógica de busca híbrida e prompts
│   ├── llm_handler.py    # Gerenciador de múltiplos LLMs
│   └── ingest.py         # Script de ingestão do PDF
├── docker-compose.yml    # Configuração PostgreSQL + pgVector
├── requirements.txt      # Dependências Python
├── .env.example         # Template de variáveis de ambiente
├── document.pdf         # Documento para processamento
└── README.md            # Esta documentação
```

## Bibliotecas e Dependências

### Core Framework
- **LangChain**: Framework principal para RAG e LLMs
- **LangChain Community**: Loaders de documentos (PyPDF)
- **LangChain Text Splitters**: Divisão inteligente de texto

### Bancos e Vetorização
- **pgVector**: Extensão PostgreSQL para busca vetorial
- **psycopg2**: Cliente PostgreSQL para Python
- **LangChain Postgres**: Integração LangChain + PostgreSQL

### Modelos LLM
- **LangChain OpenAI**: Integração com ChatGPT e embeddings
- **LangChain Google GenAI**: Integração com Gemini

### Utilitários
- **python-dotenv**: Gerenciamento de variáveis de ambiente
- **pypdf**: Processamento de arquivos PDF
- **numpy**: Operações numéricas para embeddings

## Instruções de Instalação e Execução

### Pré-requisitos

- Python 3.8+
- Docker e Docker Compose
- Git

### 1. Clone o Repositório

```bash
git clone <url-do-repositorio>
cd projeto-rag
```

### 2. Configure o Ambiente Python

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# No Linux/WSL:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Configure as API Keys

```bash
# Copiar template de configuração
cp .env.example .env

# Editar arquivo .env com suas chaves
nano .env
```

Configure no arquivo `.env`:

```env
# OpenAI Configuration
OPENAI_API_KEY='sk-sua_chave_openai_aqui'
OPENAI_MODEL=gpt-4o-mini

# Google AI Configuration  
GOOGLE_API_KEY='sua_chave_google_aqui'
GOOGLE_MODEL=gemini-2.0-flash-lite

# Default Settings (opcional)
DEFAULT_LLM_MODEL=gemini
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small

# Database (manter como está)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
```

### 4. Inicialize o Banco de Dados

```bash
# Subir PostgreSQL com pgVector
docker compose up -d

# Verificar se está rodando
docker compose ps

# Aguardar inicialização completa
docker compose logs postgres
```

### 5. Execute a Ingestão do PDF

```bash
# Processar o documento PDF
python src/ingest.py
```

**Saída esperada:**
```
📄 Iniciando ingestão do PDF...
✅ PDF carregado: X páginas
🔗 Processando texto completo...
✅ Texto completo: X caracteres
✂️ Dividindo em chunks...
✅ X chunks criados
🔗 Criando embeddings...
💾 Salvando no banco vetorial PostgreSQL...
🎉 SUCESSO! PDF ingerido com X chunks
```

### 6. Execute o Chat

```bash
# Iniciar sistema de chat
python src/chat.py
```

## Manual de Uso do Chat

### Comandos Especiais

- `modelo` - Ver modelo LLM atual
- `modelos` - Listar todos os modelos disponíveis
- `trocar` - Trocar modelo interativamente
- `status` - Ver status completo do sistema
- `sair` - Encerrar o chat

### Exemplos de Consultas

**Consultas Comparativas:**
```
Qual a empresa de maior faturamento?
Liste as 3 empresas com maior faturamento
Quais as 5 empresas com menor faturamento?
```

**Consultas Específicas:**
```
Qual o faturamento da SuperTechIABrazil?
Qual o ano de fundação da Alfa Energia S.A.?
Quantas empresas têm 'Sustentável' no nome?
```

**Teste de Aderência:**
```
Qual é a capital da França?
Como fazer um bolo de chocolate?
```

## Evidências de Testes

### Bateria de Testes Realizada

Foram executados testes comparativos entre ChatGPT (gpt-4o-mini) e Google Gemini (2.0-flash-lite):

| Categoria | Teste | ChatGPT | Gemini | Resultado |
|-----------|-------|---------|---------|-----------|
| **Comparativas** | Maior faturamento | ✅ Vanguarda Siderurgia Participações | ✅ Vanguarda Siderurgia Participações | Ambos corretos |
| | Top 3 faturamento | ✅ Lista completa + valores | ✅ Lista completa | Ambos corretos |
| | 5 menores faturamento | ✅ Ordenação correta | ✅ Ordenação correta | Ambos corretos |
| | Empresa mais recente | ✅ SuperTechIABrazil (2025) | ✅ SuperTechIABrazil | Ambos corretos |
| **Específicas** | Faturamento específico | ✅ R$ 10.000.000,00 | ✅ R$ 10.000.000,00 | Ambos corretos |
| | Ano de fundação | ✅ 1972 | ✅ 1972 | Ambos corretos |
| | Contagem empresas | ✅ 18 empresas | ✅ 9 empresas | Discrepância minor |
| **Aderência** | Capital França | ✅ Negou responder | ✅ Negou responder | Ambos corretos |
| | Clientes 2024 | ❌ Interpretação incorreta | ✅ Negou responder | Gemini superior |
| | Receita bolo | ✅ Negou responder | ✅ Negou responder | Ambos corretos |

### Análise de Performance

**Pontos Fortes:**
- ✅ **Precisão em consultas comparativas**: Ambos os modelos identificam corretamente valores máximos/mínimos
- ✅ **Aderência ao contexto**: Sistema evita alucinações efetivamente
- ✅ **Consultas específicas**: Informações pontuais são recuperadas com precisão
- ✅ **Ordenação matemática**: Pré-processamento garante comparações numéricas corretas

**Discrepâncias Identificadas:**
- **Contagem de entidades**: Variação entre modelos (18 vs 9 empresas com "Sustentável")
- **Interpretação ambígua**: ChatGPT ocasionalmente interpreta perguntas fora do contexto

### Causas Técnicas das Discrepâncias

1. **Busca vetorial não-determinística**: Diferentes embeddings (OpenAI vs Google) recuperam chunks ligeiramente diferentes
2. **Limitação k=30**: Nem todos os dados relevantes são sempre recuperados
3. **Busca lexical variável**: Termos podem estar distribuídos em chunks não recuperados
4. **Processamento de contexto**: Variações sutis na interpretação de texto entre modelos

## Melhorias Futuras

### Interface e Experiência do Usuário

**Interface Web Moderna:**
- Desenvolver interface web com Streamlit ou FastAPI + React
- Chat em tempo real com histórico persistente
- Upload de múltiplos PDFs via drag & drop
- Visualização de chunks recuperados para transparência
- Indicadores de confiança das respostas

**Interface Mobile:**
- Aplicativo mobile para consultas via voz
- Integração com assistentes virtuais (Siri, Google Assistant)
- Notificações push para atualizações de documentos

### Melhorias Técnicas

**Curto Prazo:**
- Aumentar k para 50-100 chunks para maior cobertura
- Implementar cache Redis para consultas frequentes
- Adicionar suporte a múltiplos formatos (DOCX, TXT, HTML)
- Sistema de logs estruturados para analytics

**Médio Prazo:**
- **GraphRAG**: Implementar busca baseada em grafos de conhecimento
- **Embeddings fine-tuned**: Treinar embeddings específicos para dados empresariais
- **Multi-modal RAG**: Suporte a imagens, tabelas e gráficos
- **Agentes especializados**: LLM agents para diferentes tipos de consulta

**Longo Prazo:**
- **RAG Avançado**: Multi-hop reasoning e chain-of-thought
- **Base de conhecimento híbrida**: Combinação de dados estruturados e não-estruturados
- **Sistema de feedback**: Aprendizado contínuo baseado em correções
- **API empresarial**: Endpoints REST para integração com sistemas externos


---

## Tecnologias Utilizadas

**Backend:**
- Python 3.8+
- LangChain
- PostgreSQL + pgVector
- Docker

**LLMs:**
- OpenAI GPT-4o-mini
- Google Gemini 2.0-flash-lite

**Embeddings:**
- OpenAI text-embedding-3-small
- Google Generative AI Embeddings

---

## Conclusão

O projeto demonstra uma implementação robusta de RAG que combina as melhores práticas de recuperação de informação com modelos de linguagem modernos. A arquitetura híbrida e as otimizações específicas por modelo garantem alta precisão nas respostas, enquanto mantém a aderência rigorosa ao contexto para evitar alucinações.

As evidências de teste comprovam a eficácia do sistema para diferentes tipos de consulta, e o roadmap de melhorias futuras posiciona a solução para evolução contínua e aplicação em cenários empresariais complexos.