import json


def import_json_configuration(file_name):
    with open(file_name) as f:
        data = json.load(f)
    strings_list = []
    for key, value in data.items():
        strings_list.append(str(key))
        strings_list.append(str(value))
    return strings_list
