#tests\test_quality_scorer.py
#The test is created to check if the quality scorer component works perfectly.
from dataformer.components import QualityScorer
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
def test_quality_scorer():
    input = [{"instruction": "What are the key features of Python programming language?",
            "responses": ["Python is known for its simplicity, readability, and versatility. It supports multiple programming paradigms, has a rich standard library, and is widely used in various domains such as web development, data science, and automation.",
                            "Python is a language. It is used for coding. Some people like it. It can do things. There are libraries. It is not the only language. Some say it's good, others not so much.",
                            "Python is a popular programming language with easy syntax and extensive libraries. It is used for tasks like scripting, web development, and scientific computing. Its dynamic typing can be both a strength and a weakness depending on the context."
                            ]}]

    llm = AsyncLLM(
        model="gpt-4o", api_provider="openai"
    )

    scorer = QualityScorer(
        llm=llm
    )

    results = scorer.score(
        input, use_cache=False
        ) # By default cache is True.


    assert results is not None,"Response is None"
    for result in results:
        assert results is not None,"A response is missing"
        assert all(i in list(result.keys()) for i in ['instruction','responses','scores']),'Some keys missing in response'
        assert result['instruction'] is not None,"instruction value not found in Response"
        assert result['responses'] is not None,"response value not found in Resposne"
        assert result['scores'] is not None,"scores not found in resposne"
   