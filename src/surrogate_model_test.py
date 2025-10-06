import os
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, os.path.join(os.getcwd(), '../ciceroscm/', 'src'))

from src.utils import load_processed_data, prepare_dataloaders
from src.utils.config_utils import load_yaml_config
from src.utils.model_utils import instantiate_model, parse_model_config
from src.utils.train_utils import validation_metrics

def main():
    surrogate_test_cfg = load_yaml_config("surrogate_model_test.yaml", "surrogate_test")
    data_dir = surrogate_test_cfg["data_dir"]
    data_dir = os.path.join(data_dir, "processed")
    data_dir = Path(data_dir).expanduser()
    
    model_dir = surrogate_test_cfg["model_dir"]
    actual_model_dir = os.path.join(model_dir, "model.pth")
    actual_model_dir = Path(actual_model_dir).expanduser()

    training_config_path = os.path.join(model_dir, "train_config.yaml")
    training_config = load_yaml_config(training_config_path)
    model_config = training_config["model"]
    device = training_config['general']['device']
    model_type, hidden_size, num_layers, kernel_size = parse_model_config(model_config)

    # Load and format data
    X_train, y_train, X_val, y_val, X_test, y_test, mu, std = load_processed_data(data_dir)
    train_loader, val_loader, test_loader, G = prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, mu, std,
                                                                batch_size=512, device=device, num_workers=2)

    # Instantiate model
    model = instantiate_model(model_type, G, hidden_size, num_layers, kernel_size=kernel_size, device=device, freeze=False)
    model.load_state_dict(torch.load(actual_model_dir, map_location=device, weights_only=False))
    model.eval()
    # Evaluate model
    metrics = validation_metrics(test_loader, model, device)

    # Save results in model dir
    results_path = os.path.join(model_dir, "test_results.txt")
    with open(results_path, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    

if __name__ == "__main__":
    main()
