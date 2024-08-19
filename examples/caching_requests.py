from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
#Initialize object
llm = AsyncLLM(api_provider="together",project_name="generate_Data")

#Create a request list and generae responses with default project name: dataformer
#Results stored in cache for dataformer project, to avoid overhead of cache when working in diferent project, 
#Can delete these cache files with just project name
request_list=[
    {"messages":[{"role":"user","content":"Why should people read books?"}]},
    {"messages":[{"role":"user","content":"What is the importance of clouds? "}],"api_provider":"together"}
]

llm.generate(request_list)

#Generate new requests for new project
request_list=[
    {"messages":[{"role":"user","content":"Why should people read books?"}]},
    {"messages":[{"role":"user","content":"Name people who have won medals at olympics 2024."}],"api_provider":"together"}
]
llm.generate(request_list,project_name="Questions")

#Generate new requests for new project
request_list=[
    {"messages":[{"role":"user","content":"What is 2+10/2?"}]},
    {"messages":[{"role":"user","content":"Solve 5x+2x=0"}],"api_provider":"together"}
]
llm.generate(request_list,project_name="Maths")
#Delete cache for old projects
request_list=[
    {"messages":[{"role":"user","content":"Why should people read books?"}]}, #Request will be skipped
    {"messages":[{"role":"user","content":"Name people who have won medals at olympics 2024."}],"api_provider":"together"}
]

#Delete specific project's cache 
llm.generate(request_list,project_name="NewProject",clear_project_cache="Questions")

#Delete Multiple project caches
llm.generate(request_list,project_name="New",clear_project_cache=["Maths","Generate_data"])

#Delete entire cache, all the projects are included
llm.generate(request_list,project_name="New",clear_project_cache="full")