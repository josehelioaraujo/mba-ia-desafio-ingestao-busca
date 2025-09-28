import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class LLMHandler:
    """Orquestrador de múltiplos modelos LLM com fallback automático"""
    
    @property
    def MODELS(self):
        return {
            'openai': {
                'name': f'ChatGPT ({os.getenv("OPENAI_MODEL", "gpt-4o-mini")})',
                'class': ChatOpenAI,
                'config': {
                    'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    'temperature': 0
                },
                'api_key_env': 'OPENAI_API_KEY'
            },
            'gemini': {
                'name': f'Google Gemini ({os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite").replace("gemini-", "")})',
                'class': ChatGoogleGenerativeAI,
                'config': {
                    'model': os.getenv('GOOGLE_MODEL', 'gemini-2.0-flash-lite'),
                    'temperature': 0
                },
                'api_key_env': 'GOOGLE_API_KEY'
            }
        }
    
    def __init__(self):
        self.available_models = {}
        self.current_model = None
        self._initialize_models()
        self._set_default_model()
    
    def _initialize_models(self):
        """Inicializa os modelos disponíveis"""
        for model_key, model_info in self.MODELS.items():
            try:
                api_key = os.getenv(model_info['api_key_env'])
                
                if api_key and api_key.strip("'") != "coloque aqui":
                    config = model_info['config'].copy()
                    
                    if model_key == 'openai':
                        config['api_key'] = api_key.strip("'")
                        self.available_models[model_key] = model_info['class'](**config)
                    elif model_key == 'gemini':
                        config['google_api_key'] = api_key.strip("'")
                        self.available_models[model_key] = model_info['class'](**config)
                    
                    print(f"✅ {model_info['name']} inicializado")
                else:
                    print(f"⚠️ API key não encontrada para {model_info['name']}")
                    
            except Exception as e:
                print(f"❌ Erro ao inicializar {model_info['name']}: {e}")
    
    def _set_default_model(self):
        """Define o modelo padrão respeitando o .env"""
        env_model = os.getenv("DEFAULT_LLM_MODEL")
        
        if env_model and env_model in self.available_models:
            self.current_model = env_model
            print(f"🎯 Usando modelo padrão do .env: {self.get_model_display_name()}")
            return
        
        if env_model and env_model not in self.available_models:
            print(f"⚠️ Modelo '{env_model}' definido no .env não está disponível!")
        
        # Fallback com prioridade: gemini > openai
        priority_order = ['gemini', 'openai']
        
        for model in priority_order:
            if model in self.available_models:
                self.current_model = model
                print(f"🎯 Usando modelo padrão: {self.get_model_display_name()}")
                return
        
        if self.available_models:
            self.current_model = list(self.available_models.keys())[0]
            print(f"🔄 Usando primeiro modelo disponível: {self.get_model_display_name()}")
        else:
            print("❌ Nenhum modelo LLM disponível!")
    
    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis"""
        return list(self.available_models.keys())
    
    def list_models(self) -> None:
        """Lista todos os modelos com status"""
        print("\n📋 MODELOS DISPONÍVEIS:")
        print("=" * 50)
        
        for i, (key, info) in enumerate(self.MODELS.items(), 1):
            status = "✅ ATIVO" if key == self.current_model else "⚪ Disponível" if key in self.available_models else "❌ Indisponível"
            print(f"{i}. {info['name']} ({key}) - {status}")
        
        print("=" * 50)
    
    def set_model(self, model_name: str) -> bool:
        """Altera o modelo ativo"""
        if model_name in self.available_models:
            old_model = self.current_model
            self.current_model = model_name
            old_name = self.MODELS[old_model]['name'] if old_model in self.MODELS else old_model
            new_name = self.MODELS[model_name]['name']
            print(f"✅ Modelo alterado de {old_name} para {new_name}")
            return True
        else:
            available = ", ".join([f"{k} ({self.MODELS[k]['name']})" for k in self.available_models.keys()])
            print(f"❌ Modelo '{model_name}' não disponível.")
            if available:
                print(f"   Disponíveis: {available}")
            return False
    
    def select_model_interactive(self) -> bool:
        """Permite seleção interativa do modelo"""
        if not self.available_models:
            print("❌ Nenhum modelo disponível para seleção!")
            return False
        
        self.list_models()
        
        try:
            available_keys = list(self.available_models.keys())
            print(f"\n📢 Digite o número (1-{len(available_keys)}) ou nome do modelo:")
            
            choice = input("Escolha: ").strip().lower()
            
            # Tentar por número
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_keys):
                    return self.set_model(available_keys[idx])
            
            # Tentar por nome
            if choice in available_keys:
                return self.set_model(choice)
            
            print("❌ Opção inválida!")
            return False
            
        except Exception as e:
            print(f"❌ Erro na seleção: {e}")
            return False
    
    def get_current_model(self) -> str:
        """Retorna o modelo atual"""
        return self.current_model
    
    def get_model_display_name(self) -> str:
        """Retorna nome amigável do modelo atual"""
        if self.current_model in self.MODELS:
            return self.MODELS[self.current_model]['name']
        return self.current_model.upper() if self.current_model else "NENHUM"
    
    def invoke(self, prompt: str) -> Optional[str]:
        """Executa prompt no modelo atual com fallback"""
        if not self.available_models:
            print("❌ Nenhum modelo LLM disponível")
            return None
            
        try:
            if self.current_model in self.available_models:
                response = self.available_models[self.current_model].invoke(prompt)
                return response.content
        except Exception as e:
            print(f"❌ Erro no modelo {self.get_model_display_name()}: {e}")
            
            # Fallback para outro modelo
            for model_name, model in self.available_models.items():
                if model_name != self.current_model:
                    try:
                        print(f"🔄 Tentando fallback para {self.MODELS[model_name]['name']}...")
                        response = model.invoke(prompt)
                        print(f"✅ Fallback bem-sucedido com {self.MODELS[model_name]['name']}")
                        return response.content
                    except Exception as fallback_error:
                        print(f"❌ Fallback {self.MODELS[model_name]['name']} falhou: {fallback_error}")
        
        print("❌ Todos os modelos falharam")
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre os modelos"""
        info = {
            "current": self.current_model,
            "current_display": self.get_model_display_name(),
            "available": self.get_available_models(),
            "available_display": [self.MODELS[k]['name'] for k in self.available_models.keys()],
            "total": len(self.available_models),
            "all_models": {key: info['name'] for key, info in self.MODELS.items()}
        }
        return info
    
    def is_available(self) -> bool:
        """Verifica se há pelo menos um modelo disponível"""
        return len(self.available_models) > 0