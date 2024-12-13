import json

from custom_dataset import create_pytorch_custom_dataset
from model import train, eval

if __name__ == "__main__":
    config_file_path = 'config.json'
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    train_file_path = config['train_file_path']
    test_file_path = config['test_file_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    model_lr = config['learning_rate']

    train_dataset, test_dataset = create_pytorch_custom_dataset(train_file_path, test_file_path)

    model = train(train_dataset, num_epochs, batch_size, model_lr)
    eval(model, test_dataset, batch_size)
