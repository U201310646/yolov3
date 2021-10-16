import os


def parse_data_config(file_path):
    os.chdir('../')
    with open(file_path, 'r') as f:
        config = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue
                key, value = line.split('=')
                config[key.strip()] = value.strip()


def parse_model_config(model_config):
    """
    :param model_config: model.cfg file path
    :return: [{"type:value,"param1":value,..}]
    """
    with open(model_config, 'r') as f:
        model_defs = []
        lines = f.read().split('\n')
        for line in lines:
            line.strip()
            if line.startswith('#') or line == '':
                continue
            if line.startswith('[') and line.endswith(']'):
                model_defs.append({})
                model_defs[-1]['type'] = line[1:-1]
            else:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                model_defs[-1][key] = value
    return model_defs
