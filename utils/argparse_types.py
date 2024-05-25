import os


def existing_dir(path):
    if os.path.isdir(path):
        return path
    else:
        raise ValueError(f'{path} does not exists')


def existing_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise ValueError(f'{path} does not exists')


def new_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

