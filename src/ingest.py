from search import search_prompt_hibrido
from llm_handler import LLMHandler

def main():
    print("🤖 Sistema de Chat - Desafio MBA IA - RAG com Documentos PDF")
    print("Digite suas perguntas ou 'sair' para encerrar")
    print("=" * 50)
    
    # Inicializar LLMHandler
    print("🔧 Inicializando sistema...")
    llm_handler = LLMHandler()
    
    if not llm_handler.is_available():
        print("❌ Não foi possível iniciar o chat. Nenhum modelo LLM disponível.")
        print("Certifique-se de que:")
        print("- Pelo menos uma API key está configurada (OpenAI ou Google)")
        print("- As variáveis de ambiente estão corretas no .env")
        return
   
    # Testar inicialização do sistema de busca
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
                
            # Processar pergunta
            print(f"\nPERGUNTA: {pergunta}")
            print(f"🤖 Usando: {llm_handler.get_model_display_name()}")
            
            resposta = search_prompt_hibrido(pergunta, llm_handler=llm_handler)

            if resposta:
                print(f"RESPOSTA: {resposta}")
            else:
                print("RESPOSTA: Erro ao processar sua pergunta.")
            
            print("\n---")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat encerrado!")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
            print("Tente novamente ou digite 'sair' para encerrar.")

if __name__ == "__main__":
    main()