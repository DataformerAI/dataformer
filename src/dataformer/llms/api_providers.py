model_dict = {
    "https://api.openai.com/v1/chat/completions": "gpt-3.5-turbo",
    "https://api.openai.com/v1/completions": "gpt-3.5-turbo-instruct",
    "https://api.groq.com/openai/v1/chat/completions": "mixtral-8x7b-32768",    
    "https://llm.monsterapi.ai/v1/chat/completions": "google/gemma-2-9b-it",
    "https://llm.monsterapi.ai/v1/completions": "google/gemma-2-9b-it",
    "https://api.together.xyz/v1/chat/completions": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "https://api.together.xyz/v1/completions": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "https://api.deepinfra.com/v1/openai/chat/completions": "mistralai/Mistral-7B-Instruct-v0.3",
    "https://api.deepinfra.com/v1/openai/completions": "mistralai/Mistral-7B-Instruct-v0.1",
    "https://openrouter.ai/api/v1/chat/completions": "openai/gpt-3.5-turbo",
    "https://openrouter.ai/api/v1/completions": "openai/gpt-3.5-turbo",
    "https://api.anthropic.com/v1/messages": "claude-3-5-sonnet-20240620",
    
}

url_dict = {
    "openai": {
        "chat": "https://api.openai.com/v1/chat/completions",
        "text": "https://api.openai.com/v1/completions",
        "models":"https://api.openai.com/v1/models"
    },
    "groq": {
        "chat": "https://api.groq.com/openai/v1/chat/completions",
        "text": "https://api.groq.com/openai/v1/completions",
        "models":"https://api.groq.com/openai/v1/models"
    },
    "monsterapi": {
        "chat": "https://llm.monsterapi.ai/v1/chat/completions",
        "text": "https://llm.monsterapi.ai/v1/completions",
        "models":"https://llm.monsterapi.ai/v1/models"
    },
    "together": {
        "chat": "https://api.together.xyz/v1/chat/completions",
        "text": "https://api.together.xyz/v1/completions",
        "models":"https://api.together.xyz/v1/models"
    },
    "deepinfra": {
        "chat": "https://api.deepinfra.com/v1/openai/chat/completions",
        "text": "https://api.deepinfra.com/v1/openai/completions",
        "models":"https://api.deepinfra.com/v1/openai/models"
    },
    "openrouter": {
        "chat": "https://openrouter.ai/api/v1/chat/completions",
        "text": "https://openrouter.ai/api/v1/completions",
        "models":"https://openrouter.ai/api/v1/models"
    },
    "anthropic": {
        "chat": "https://api.anthropic.com/v1/messages",
        
    },
}

api_key_dict = {
    "api.openai.com": "OPENAI_API_KEY",
    "api.groq.com": "GROQ_API_KEY",
    "llm.monsterapi.ai": "MONSTER_API_KEY",
    "api.anthropic.com": "ANTHROPIC_API_KEY",
    "api.together.xyz": "TOGETHER_API_KEY",
    "api.deepinfra.com": "DEEPINFRA_API_KEY",
    "openrouter.ai": "OPENROUTER_API_KEY",
}
