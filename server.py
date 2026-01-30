import os
import sys
import csv
import torch
import flwr as fl

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.eval import load_model
from utils.param import get_model_params, set_model_params
from fl_config import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HISTORY_CSV = os.path.join(RESULT_DIR, "training_history.csv")
os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
global_model = load_model(
    path=ECGCNN_MODEL_PATH,
    input_channels=1, 
    num_classes=5,
    input_length=29,
    device=DEVICE,
)
def append_history(round_num: int, metrics: dict, filename: str = HISTORY_CSV):
    """Simpan hasil training ke CSV."""
    fieldnames = ["round", "num_samples", "accuracy", "loss"]
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {"round": round_num}
        if metrics:
            row.update(metrics)
        writer.writerow(row)

class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, patience_rounds=10, monitor="accuracy"):
        super().__init__()
        self.model = model
        self.patience_rounds = patience_rounds 
        self.monitor = monitor
        self.best_metric = -float("inf")
        self.rounds_no_improve = 0

    def initialize_parameters(self, client_manager):
        print("🔹 Broadcasting pretrained model to clients")
        return fl.common.ndarrays_to_parameters(get_model_params(self.model))

    @staticmethod
    def fit_metrics_aggregation_fn(metrics):
        metrics = [m for _, m in metrics]
        
        train_losses = [m["train_loss"] for m in metrics if "train_loss" in m]
        val_losses = [m["val_loss"] for m in metrics if "val_loss" in m]
        train_accs = [m["train_acc"] for m in metrics if "train_acc" in m]
        val_accs = [m["val_acc"] for m in metrics if "val_acc" in m]

        return {
            "train_loss": sum(train_losses) / len(train_losses) if train_losses else None,
            "train_acc": sum(train_accs) / len(train_accs) if train_accs else None,
            "val_loss": sum(val_losses) / len(val_losses) if val_losses else None,
            "val_acc": sum(val_accs) / len(val_accs) if val_accs else None,
        }

    @staticmethod
    def evaluate_metrics_aggregation_fn(metrics):
        metrics = [m for _, m in metrics]
        accs = [m["test_acc"] for m in metrics]
        losses = [m["test_loss"] for m in metrics]

        return {
            "test_acc": sum(accs) / len(accs),
            "test_loss": sum(losses) / len(losses),
        }

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None

        if isinstance(aggregated, tuple):
            parameters, _ = aggregated
        else:
            parameters = aggregated

        metrics = self._aggregate_client_metrics(results)
        params_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        set_model_params(self.model, params_ndarrays)
        model_path = os.path.join(GLOBAL_MODEL_DIR, f"model_global_round{rnd}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"💾 Global model updated and saved after round {rnd}")
        if metrics:
            append_history(rnd, metrics)
        current_metric = metrics.get(self.monitor, 0.0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.rounds_no_improve = 0
        else:
            self.rounds_no_improve += 1
            print(f"⚠️ No improvement in {self.rounds_no_improve} round(s)")

        return aggregated
    
    @staticmethod
    def _aggregate_client_metrics(results):
        if not results:
            return None

        total_samples = 0
        weighted_acc = 0.0

        for _, fit_res in results:
            num_examples = getattr(fit_res, "num_examples", 0)
            acc = fit_res.metrics.get("final_val_acc", 0.0)

            total_samples += num_examples
            weighted_acc += acc * num_examples

        accuracy = weighted_acc / total_samples if total_samples else 0.0

        return {
            "num_samples": total_samples,
            "accuracy": accuracy,
            "loss": 1 - accuracy,
        }


# =============================
# Main Entry Point
# =============================
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=MyStrategy(global_model),
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
    )