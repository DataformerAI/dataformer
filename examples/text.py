
from dotenv import load_dotenv
from dataformer.llms import AsyncLLM
# Load environment variables from .env file
load_dotenv()
request_list = [{"prompt": "Complete the paragraph.\n She lived in nashville"},{"prompt": "Write a story on 'Honesty is the best Policy'"}]

llm=AsyncLLM(api_provider="together",gen_type="text")
# Assuming llm.generate returns a response
response = llm.generate(request_list)
print(response)