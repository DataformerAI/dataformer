# Examples

Setup .env file for the provider you want to use.
```
OPENAI_API_KEY=
GROQ_API_KEY=
TOGETHER_API_KEY= 
ANYSCALE_API_KEY=
DEEPINFRA_API_KEY=
OPENROUTER_API_KEY=
MONSTER_API_KEY=
ANTHROPIC_API_KEY=
```

Also, set max rate limits available for you by your api provider in .env file.    
Default values are 20 RPM & 10K TPM.    
```
MAX_REQUESTS_PER_MINUTE=
MAX_TOKENS_PER_MINUTE=
```
Set project name, default is dataformer
```
PROJECT_NAME=
``s`
## Chat
`python examples/chat.py`

## Evol Instruct
`python examples/evol_instruct.py`

## Evol Quality
`python examples/evol_quality.py`