import hashlib
import shutil
import os

def get_cache_vars(input, ignore_keys=None, additional_data=None):
    cache_vars = vars(input)

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
        
def delete_cache(dir_path=".cache/dataformer"):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} has been deleted successfully.")
        except Exception as e:
            print(f"Error: {e}")