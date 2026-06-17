import json
import os
import copy


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'default_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config = _validate_config(config)
    config = _compute_derived_config(config)
    return config


def _validate_config(config):
    required = ['project', 'data', 'preprocessing', 'featureExtraction',
                'deepLearning', 'federatedLearning', 'evaluation']
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    if config['data']['samplingRate'] <= 0:
        raise ValueError("samplingRate must be positive")
    if not (1 <= config['data']['windowSize'] <= 10000):
        raise ValueError("windowSize must be between 1 and 10000")
    if not (0 < config['data']['trainRatio'] < 1):
        raise ValueError("trainRatio must be between 0 and 1")

    dl = config['deepLearning']
    if dl['numClasses'] < 2:
        raise ValueError("numClasses must be at least 2")
    if dl['latentDim'] < 1:
        raise ValueError("latentDim must be positive")

    fl = config['federatedLearning']
    if fl['numClients'] < 2:
        raise ValueError("numClients must be at least 2")
    if not (0 <= fl['byzantineFraction'] <= 0.5):
        raise ValueError("byzantineFraction must be between 0 and 0.5")

    return config


def _compute_derived_config(config):
    config['derived'] = {}
    config['derived']['nTrainSamples'] = int(
        config['data']['windowSize'] * config['data']['trainRatio'])
    config['derived']['featureDim'] = config['data']['windowSize']
    config['derived']['inputShape'] = [config['data']['windowSize'], config['data'].get('inputChannels', 1)]
    config['derived']['nByzantine'] = int(
        config['federatedLearning']['numClients'] * config['federatedLearning']['byzantineFraction'])

    base = os.path.dirname(os.path.dirname(__file__))
    config['derived']['paths'] = {
        'results': os.path.join(base, 'Results'),
        'exports': os.path.join(base, 'Export'),
    }

    for path in config['derived']['paths'].values():
        os.makedirs(path, exist_ok=True)

    return config


def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
