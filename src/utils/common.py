import yaml


def read_yml(path: str):
    with open(path, 'r') as file:
        text = yaml.safe_load(file)
    return text
