# tests\test_caching_requests.py
#The Test is created to check if chaching operations takes place properly and the cached requests are skipped,also if a project's cache or entire cache is cleared, the required files are deleted.
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
import os
import json
# Load environment variables from .env file
load_dotenv()

cache_dir=".cache/project"
def check_ifexists(project_name,cache_dir):
    project_name=project_name.lower()
    #assert cache directory is created properly 
    assert os.path.exists(cache_dir), "Cache dir doesn't exist"
    #assert association file is created properly
    assert os.path.exists(os.path.join(cache_dir,"association.jsonl")),"Association File doesn't exist"

    files = os.listdir(cache_dir)
    
    with open(os.path.join(cache_dir,"association.jsonl"),"r") as f:
        data = json.load(f)

    #Assert project cache association is marked properly
    assert project_name in list(data['project_requests'].keys())
    #assert only the files of association dictionary exists
    for i in data['project_requests'][project_name]:
            assert i+".jsonl" in files, "Some Files of Project are not deleted."


def get_files_tobe_deleted(project_name,cache_dir):
    files_tobe_deleted=[]
    if os.path.exists(cache_dir) and os.path.exists(cache_dir+"/association.jsonl"):
         with open(cache_dir+"/association.jsonl","r") as f:
              data= json.load(f)
              if isinstance(project_name,str):
                   if project_name.lower()=="full":
                        for key in data['project_requests']:
                             files_tobe_deleted.extend(data['project_requests'][key])
                   else:
                        files_tobe_deleted = list(data['project_requests'][project_name.lower()])
              elif isinstance(project_name,list):
                   for project in project_name:
                        files_tobe_deleted.extend(data['project_requests'][project.lower()])
    return files_tobe_deleted
     
def delete_cache_check(project_name_deleted,cache_dir,new_project_name,deleted_files):
     new_project_name=new_project_name.lower()
     #assert the association in the association file is deleted
     if os.path.exists(os.path.join(cache_dir,"association.jsonl")):
        with open(os.path.join(cache_dir,"association.jsonl"),"r") as f:
               data = json.load(f)
               if "project_requests" in list(data.keys()):
                    if isinstance(project_name_deleted,str):
                        project_name_deleted=project_name_deleted.lower()
                        if project_name_deleted=="full":
                             assert all(i==new_project_name for i in list(data['project_requests'].keys())),"All cache deleted, but no entry found for new project of requests in association file"
                        assert project_name_deleted not in list(data['project_requests'].keys()),"Project name not deleted from the association file"
                    elif isinstance(project_name_deleted,list):
                         for i in project_name_deleted:
                              assert i.lower() not in list(data['project_requests'].keys()),"Project name not deleted from the association file"
        #assert all files in association are aslso deleted
        files = os.listdir(cache_dir)
        for i in files:
                assert i+".jsonl" not in deleted_files,"Some Files from cache not deleted"

def test_caching_requests():
     #Initialize object
    llm = AsyncLLM(api_provider="together",project_name="generate_Data",cache_dir=cache_dir)


    request_list=[
        {"messages":[{"role":"user","content":"Why should people read books?"}]},
        {"messages":[{"role":"user","content":"What is the importance of clouds? "}],"api_provider":"together"}
    ]

    llm.generate(request_list)
    check_ifexists("generate_Data",cache_dir)

    #Generate new requests for new project
    request_list=[
        {"messages":[{"role":"user","content":"Why should people read books?"}]},
        {"messages":[{"role":"user","content":"Name people who have won medals at olympics 2024."}],"api_provider":"together"}
    ]
    llm.generate(request_list,project_name="Questions")
    check_ifexists("Questions",cache_dir)

    #Generate new requests for new project
    request_list=[
        {"messages":[{"role":"user","content":"What is 2+10/2?"}]},
        {"messages":[{"role":"user","content":"Solve 5x+2x=0"}],"api_provider":"together"}
    ]
    llm.generate(request_list,project_name="Maths")
    check_ifexists("Maths",cache_dir)
    #Delete cache for old projects
    request_list=[
        {"messages":[{"role":"user","content":"Why should people read books?"}]}, #Request will be skipped
        {"messages":[{"role":"user","content":"Name people who have won medals at olympics 2024."}],"api_provider":"together"}
    ]

    #Delete specific project's cache 
    files_deleted =get_files_tobe_deleted("Questions",cache_dir)
    llm.generate(request_list,project_name="NewProject",clear_project_cache="Questions")
    check_ifexists("NewProject",cache_dir)
    delete_cache_check("questions",cache_dir,"NewProject",files_deleted)

    #Delete Multiple project caches
    files_deleted =get_files_tobe_deleted(["Maths","Generate_data"],cache_dir)
    llm.generate(request_list,project_name="New",clear_project_cache=["Maths","Generate_data"])
    check_ifexists("New",cache_dir)
    delete_cache_check(["Maths","Generate_data"],cache_dir,"New",files_deleted)

    #Delete entire cache, all the projects are included
    files_deleted =get_files_tobe_deleted("full",cache_dir)
    llm.generate(request_list,project_name="New",clear_project_cache="full",)
    check_ifexists("New",cache_dir)
    delete_cache_check("full",cache_dir,"New",files_deleted)
