import os


def create_folder_if_not_exists(folder_name):
    if os.path.exists(folder_name):
        if not os.path.isdir(folder_name):
            raise ValueError("Something with the name " + folder_name + " already exists but it's not a folder.")
    else:
        os.mkdir(folder_name)


def append_separator_if_needed(folder_name):
    if folder_name[-1] == os.sep:
        return folder_name
    else:
        return folder_name + os.sep
