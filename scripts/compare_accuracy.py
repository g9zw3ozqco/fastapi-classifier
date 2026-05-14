import json
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from app.model_loader import SimpleNN


def load_best_accuracy(path: str) -> Decimal:
    if not os.path.exists(path):
        return Decimal("0")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = str(payload.get("best_accuracy", "0"))
    return Decimal(value)


def save_best_accuracy(path: str, accuracy: Decimal) -> None:
    payload = {
        "best_accuracy": format(accuracy, "0.4f"),
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def evaluate_model(model_path: str, subset_path: str) -> Decimal:
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    subset_payload = torch.load(subset_path, map_location=torch.device("cpu"))
    images = subset_payload["images"]
    labels = subset_payload["labels"]
    dataset = torch.utils.data.TensorDataset(images, labels)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_images, batch_labels in testloader:
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = Decimal(correct) / Decimal(total)
    return accuracy.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def main() -> None:
    model_path = os.path.join(REPO_ROOT, "model", "model.pth")
    subset_path = os.path.join(REPO_ROOT, "data", "ci_eval_subset.pt")
    baseline_path = os.path.join(REPO_ROOT, "model", "best_accuracy.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Missing eval subset: {subset_path}")

    current = evaluate_model(model_path, subset_path)
    best = load_best_accuracy(baseline_path)

    print(f"Current accuracy: {current}")
    print(f"Best accuracy: {best}")

    if current > best:
        print("New best accuracy found. Updating baseline.")
        save_best_accuracy(baseline_path, current)
    else:
        print("No improvement. Baseline stays the same.")


if __name__ == "__main__":
    main()
