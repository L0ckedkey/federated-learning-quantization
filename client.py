import torch
import numpy as np
import os
import sys
import time
import csv
import argparse
import torch
import flwr as fl

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fl_config import *
from utils.param import get_model_params, set_model_params
from utils.data import load_and_preprocess_new, create_dataloaders, split_data, load_and_preprocess
from models.base import ECGCNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--client", type=str, required=True, help="Client ID")
parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name")
args = parser.parse_args()

BASE_RESULT_DIR = os.path.join("fl", "results")
CLIENT_ID = args.client

DATASET_PATHS = {
    "MITDB_PATH": MITDB_PATH,
    "SVDB_PATH": SVDB_PATH,
    "LINUX_MITDB_PATH": LINUX_MITDB_PATH,
    "LINUX_SVDB_PATH": LINUX_SVDB_PATH,
    "LINUX_RAW_SVDB": LINUX_RAW_SVDB,
    "LINUX_RAW_MITDB": LINUX_RAW_MITDB,
}

DATASET = DATASET_PATHS.get(args.dataset)
if DATASET is None:
    raise ValueError(f"Dataset {args.dataset} tidak dikenali!")

def get_latest_result_dir():
    if not os.path.exists(BASE_RESULT_DIR):
        raise FileNotFoundError(f"Base result dir {BASE_RESULT_DIR} not found")

    subdirs = [
        os.path.join(BASE_RESULT_DIR, d)
        for d in os.listdir(BASE_RESULT_DIR)
        if os.path.isdir(os.path.join(BASE_RESULT_DIR, d))
    ]
    if not subdirs:
        raise FileNotFoundError("No result subdir found in fl/results")

    return max(subdirs, key=os.path.getmtime)


RESULT_DIR = get_latest_result_dir()
CLIENT_MODEL_DIR = os.path.join(RESULT_DIR, "clients")
os.makedirs(CLIENT_MODEL_DIR, exist_ok=True)
X_np, y_np, encoder = load_and_preprocess_new(DATASET)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_np, y_np)
train_loader, val_loader, test_loader = create_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
)
print(DEVICE)
model = ECGCNN(
    input_channels=X_np.shape[2],
    num_classes=MODEL_CLASS,
    input_length=X_np.shape[1],
).to(DEVICE)

def get_next_round_dir(client_model_dir, client_id):
    client_dir = os.path.join(client_model_dir, client_id)
    os.makedirs(client_dir, exist_ok=True)

    round_folders = [
        d for d in os.listdir(client_dir)
        if os.path.isdir(os.path.join(client_dir, d)) and d.startswith("round_")
    ]
    last_round = max([int(d.split("_")[1]) for d in round_folders], default=0)
    next_round = last_round + 1

    round_dir = os.path.join(client_dir, f"round_{next_round}")
    os.makedirs(round_dir, exist_ok=True)
    return round_dir, next_round

def train(model, train_loader, val_loader, epochs, device, lr=0.0005, patience=10, save_dir=None):
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.tolist())

    num_classes = 5
    class_counts = np.bincount(all_labels, minlength=num_classes)
    total_samples = len(all_labels)

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(class_counts > 0,
                           total_samples / (num_classes * class_counts),
                           0.0)

    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        infer_times = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                start_time = time.perf_counter()
                outputs = model(X_batch)
                infer_times.append((time.perf_counter() - start_time) * 1000)

                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        avg_infer_time = sum(infer_times) / len(infer_times)
        throughput = val_total / (sum(infer_times) / 1000)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "infer_time_ms": avg_infer_time,
            "throughput": throughput,
        })

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Infer Time: {avg_infer_time:.2f} ms/batch | Throughput: {throughput:.2f} samples/s"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if save_dir:
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    return history


def evaluate_model(model, loader, device, round_dir, round_number, mode="val"):
    model.eval()
    all_labels = []
    for _, y in loader:
        all_labels.extend(y.tolist())
    num_classes = 5
    class_counts = np.bincount(all_labels, minlength=num_classes)
    total_samples = len(all_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(class_counts > 0,
                           total_samples / (num_classes * class_counts),
                           0.0)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    log_data = [{"loss": avg_loss, "accuracy": acc}]
    log_path = os.path.join(round_dir, f"{mode}_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
        writer.writeheader()
        writer.writerows(log_data)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {mode.capitalize()} - Round {round_number}")
    cm_path = os.path.join(round_dir, f"{mode}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    cm_path_csv = os.path.join(round_dir, f"{mode}_confusion_matrix.csv")
    with open(cm_path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"Pred_{i}" for i in range(cm.shape[1])])
        for i, row in enumerate(cm):
            writer.writerow([f"True_{i}"] + row.tolist())

    return avg_loss, acc

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_model_params(model)

    def fit(self, parameters, config):
        set_model_params(model, parameters)

        history = train(model, train_loader, val_loader, epochs=EPOCH, device=DEVICE, lr=LR)
        round_dir, round_number = get_next_round_dir(CLIENT_MODEL_DIR, CLIENT_ID)

        torch.save(model.state_dict(), os.path.join(CLIENT_MODEL_DIR,CLIENT_ID, "model_from_fit.pth"))
        with open(os.path.join(round_dir, "log.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        avg_metrics = {
            "round": round_number,
            "train_loss": sum(h["train_loss"] for h in history) / len(history),
            "train_acc": sum(h["train_acc"] for h in history) / len(history),
            "val_loss": sum(h["val_loss"] for h in history) / len(history),
            "val_acc": sum(h["val_acc"] for h in history) / len(history),
            "infer_time_ms": sum(h["infer_time_ms"] for h in history) / len(history),
            "throughput": sum(h["throughput"] for h in history) / len(history),
        }

        summary_path = os.path.join(CLIENT_MODEL_DIR, CLIENT_ID, "avg_round.csv")
        file_exists = os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=avg_metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(avg_metrics)

        print(f"Saved results in {round_dir}")
        print(f"Round summary saved at {summary_path}")

        return get_model_params(model), len(train_loader.dataset), {
            "final_val_acc": history[-1]["val_acc"],
            "round": round_number,
        }

    def evaluate(self, parameters, config):
        set_model_params(model, parameters)
        client_dir = os.path.join(CLIENT_MODEL_DIR, CLIENT_ID)
        round_folders = sorted(
            [d for d in os.listdir(client_dir) if d.startswith("round_")],
            key=lambda x: int(x.split("_")[1])
        )
        if round_folders:
            round_dir = os.path.join(client_dir, round_folders[-1])
            round_number = int(round_folders[-1].split("_")[1])
        else:
            round_dir, round_number = get_next_round_dir(CLIENT_MODEL_DIR, CLIENT_ID)

        test_loss, test_acc = evaluate_model(model, test_loader, DEVICE, round_dir, round_number, mode="test")
        return float(test_loss), len(test_loader.dataset), {"accuracy": test_acc, "loss": test_loss}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="192.168.1.12:8080", client=FlowerClient())
