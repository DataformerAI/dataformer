import hashlib
import shutil
import os
import json

def get_cache_vars(input, ignore_keys=None, additional_data=None,vars_use=True):
    if vars_use:
        cache_vars = vars(input)
    else:
        cache_vars=input
    if ignore_keys is None:
        ignore_keys = []

    if additional_data:
        cache_vars.update(additional_data)

    filtered_cache_vars = {}

    for key, value in cache_vars.items():
        if key not in ignore_keys:
            if isinstance(value, (list, int, float, dict, str)):
                filtered_cache_vars[key] = value

    return filtered_cache_vars


def get_request_details(request_list):
    indices = [0, len(request_list) // 2, len(request_list) - 1]
    indices = list(set(indices))  # In case len(request_list) < 3
    request_details = [request_list[i] for i in indices]
    return request_details


def create_hash(instance_vars):
    hash_object = hashlib.md5()
    hash_object.update(str(instance_vars).encode())
    input_hash = hash_object.hexdigest()
    return input_hash


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1
        
def delete_cache(project_or_full="dataformer",dir_path=".cache/dataformer"):
    
    #Delete full cache directory
    if project_or_full=="full":
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Directory {dir_path} has been deleted successfully.")
            except Exception as e:
                print(f"Error: {e}")
    else:
        #Delete specfic project files mentioned
        with open(os.path.join(dir_path,"association.jsonl"),"r") as file:
            data = json.load(file)
        
        #Search for project if exists delete all request_ .jsonl files belonging to project
        
        if project_or_full in list(data['project_requests'].keys()):
            for request_name in data["project_requests"][project_or_full]:
                path = os.path.join(dir_path,request_name+".jsonl")
                print(path)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"File {path} has been deleted successfully.")
                        
                        #Change in association.jsonl file
                        association_path = os.path.join(dir_path,"association.jsonl")
                        with open(association_path,"r") as file:
                            data=json.load(file)
                        
                        for i in data['project_requests'][project_or_full]:
                            del data[i]
                        del data['project_requests'][project_or_full]
                        if len(data['project_requests'])==0:
                            os.remove(association_path)
                        else:
                            with open(association_path,"w") as file:
                                json.dump(data,file)
                    except Exception as e:
                        print(f"Error: {e}")
        else:
            raise("Inappropriate argument value provided")
