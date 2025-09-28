
from search import search_prompt, search_prompt_hibrido
from llm_handler import LLMHandler

"""
DIFERENÇAS ENTRE AS FUNÇÕES DE BUSCA:

1. search_prompt() - BUSCA SIMPLES:
   - Usa apenas busca vetorial (similarity search)
   - Recupera k chunks mais similares semanticamente à pergunta
   - Aplica pré-processamento específico apenas para Gemini em perguntas comparativas
   - Mais rápida, mas pode perder informações relevantes se os chunks não forem recuperados

2. search_prompt_hibrido() - BUSCA HÍBRIDA:
   - FASE 1: Busca vetorial (como a função simples)
   - FASE 2: Busca lexical por termos-chave extraídos da pergunta
   - FASE 3: Combina ambos os resultados, removendo duplicatas
   - Extrai termos importantes da pergunta (empresas, valores, palavras-chave)
   - Faz consultas SQL LIKE no banco para encontrar documentos com esses termos
   - Mais completa, captura dados que a busca vetorial pode perder
   - Melhor para consultas comparativas complexas como "maior faturamento"

QUANDO USAR CADA UMA:
- search_prompt(): Para perguntas simples e específicas
- search_prompt_hibrido(): Para perguntas comparativas, listas, rankings
  
VANTAGEM DA HÍBRIDA:
Combina a precisão semântica (vetorial) com a precisão lexical (termos exatos),
garantindo que empresas com valores altos sejam encontradas mesmo se não 
estiverem nos top-k chunks da busca vetorial.
"""

def main():
    print("🤖 Sistema de Chat - Desafio MBA IA - RAG com Documentos PDF")
    print("Digite suas perguntas ou 'sair' para encerrar")
    print("=" * 50)
    
    # Inicializar LLMHandler uma vez
    print("🔧 Inicializando sistema...")
    llm_handler = LLMHandler()
    
    if not llm_handler.is_available():
        print("❌ Não foi possível iniciar o chat. Nenhum modelo LLM disponível.")
        print("Certifique-se de que:")
        print("- Pelo menos uma API key está configurada (OpenAI ou Google)")
        print("- As variáveis de ambiente estão corretas no .env")
        return
   
    chain = search_prompt_hibrido(llm_handler=llm_handler)
  
    if not chain:
        print("❌ Não foi possível iniciar o sistema de busca.")
        print("Certifique-se de que:")
        print("- O banco PostgreSQL está rodando")
        print("- A ingestão foi executada com sucesso")
        print("- As variáveis de ambiente estão configuradas")
        return
    
    print("✅ Sistema pronto! Faça suas perguntas sobre o PDF.")
    print(f"🤖 Modelo ativo: {llm_handler.get_model_display_name()}")
    print("\n💡 COMANDOS ESPECIAIS:")
    print("  'modelo' - Ver modelo atual")
    print("  'modelos' - Listar todos os modelos")
    print("  'trocar' - Trocar modelo interativamente")
    print("  'status' - Ver status completo")
    print("=" * 50)
    
    while True:
        try:
            # Solicitar pergunta  
            pergunta = input("\nFaça sua pergunta: ").strip()
            
            # Comandos para sair
            if pergunta.lower() in ['sair', 'exit', 'quit', 'bye']:
                print("\n👋 Chat encerrado!")
                break
                
            # Comandos especiais
            if pergunta.lower() == 'modelo':
                print(f"🤖 Modelo atual: {llm_handler.get_model_display_name()}")
                continue
                
            if pergunta.lower() in ['modelos', 'lista']:
                llm_handler.list_models()
                continue
                
            if pergunta.lower() in ['trocar', 'switch', 'mudar']:
                llm_handler.select_model_interactive()
                continue
                
            if pergunta.lower() == 'status':
                info = llm_handler.get_model_info()
                print(f"🤖 Modelo atual: {info['current_display']}")
                print(f"📊 Modelos disponíveis: {', '.join(info['available_display'])}")
                print(f"📈 Total de modelos: {info['total']}")
                continue
                
            # Ignorar entradas vazias
            if not pergunta:
                continue
                
            # Exibir pergunta  
            print(f"\nPERGUNTA: {pergunta}")
            print(f"🤖 Usando: {llm_handler.get_model_display_name()}")
            
            # Processar pergunta e obter resposta
            resposta = search_prompt_hibrido(pergunta, llm_handler=llm_handler)

            if resposta:
                # Exibir resposta  
                print(f"RESPOSTA: {resposta}")
            else:
                print("RESPOSTA: Erro ao processar sua pergunta.")
            
            # Separador  
            print("\n---")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat encerrado!")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
            print("Tente novamente ou digite 'sair' para encerrar.")

if __name__ == "__main__":
    main()