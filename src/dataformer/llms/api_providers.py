model_dict = {
    "https://api.openai.com/v1/chat/completions": "gpt-3.5-turbo",
    "https://api.groq.com/openai/v1/chat/completions": "mixtral-8x7b-32768",
    "https://llm.monsterapi.ai/v1/chat/completions": "google/gemma-2-9b-it",
    "https://api.together.xyz/v1/chat/completions": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "https://api.endpoints.anyscale.com/v1/chat/completions": "mistralai/Mistral-7B-Instruct-v0.1",
    "https://api.deepinfra.com/v1/openai/chat/completions": "meta-llama/Meta-Llama-3-8B-Instruct",
    "https://openrouter.ai/api/v1/chat/completions": "openai/gpt-3.5-turbo",
    "https://api.anthropic.com/v1/messages": "claude-3-5-sonnet-20240620",
}

base_url_dict = {
    "openai": {
        "chat": "https://api.openai.com/v1/chat/completions",
        "text": "https://api.openai.com/v1/completions",
    },
    "groq": {
        "chat": "https://api.groq.com/openai/v1/chat/completions",
        "text": "https://api.groq.com/openai/v1/completions",
    },
    "monsterapi": {
        "chat": "https://llm.monsterapi.ai/v1/chat/completions",
        "text": "https://llm.monsterapi.ai/v1/completions",
    },
    "together": {
        "chat": "https://api.together.xyz/v1/chat/completions",
        "text": "https://api.together.xyz/v1/completions",
    },
    "anyscale": {
        "chat": "https://api.endpoints.anyscale.com/v1/chat/completions",
        "text": "https://api.endpoints.anyscale.com/v1/completions",
    },
    "deepinfra": {
        "chat": "https://api.deepinfra.com/v1/openai/chat/completions",
        "text": "https://api.deepinfra.com/v1/openai/completions",
    },
    "openrouter": {
        "chat": "https://openrouter.ai/api/v1/chat/completions",
        "text": "https://openrouter.ai/api/v1/completions",
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
    "api.endpoints.anyscale.com": "ANYSCALE_API_KEY",
    "api.deepinfra.com": "DEEPINFRA_API_KEY",
    "openrouter.ai": "OPENROUTER_API_KEY",
}
