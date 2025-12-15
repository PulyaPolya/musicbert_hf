import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F


def topk_threshold_accuracy(
    logits: np.ndarray,      # shape [B, T, C] or [N, C]
    labels: np.ndarray,      # shape [B, T] or [N]
    k: int = 3,
    min_prob: float = 0.10,
    ignore_index: int = -100,
):
    # Ensure 3D/2D consistency
    if logits.ndim == 3:   # [B, T, C] -> flatten valid positions
        B, T, C = logits.shape
        labels_flat = labels.reshape(-1)
        logits_flat = logits.reshape(-1, C)

    # mask valid tokens
    valid = labels_flat != ignore_index
    if not np.any(valid):
        return 0.0, 0, 0  # accuracy, correct, total

    labels_v = labels_flat[valid]
    logits_v = logits_flat[valid]  # [M, C]
    M, C = logits_v.shape

    # softmax
    logits_v = logits_v - logits_v.max(axis=1, keepdims=True)
    probs = np.exp(logits_v)
    probs /= probs.sum(axis=1, keepdims=True)  # [M, C]

    # top-k membership
    kk = min(k, C)
    # argpartition gives indices of top-k (unordered within the top set)
    topk_idx = np.argpartition(probs, -kk, axis=1)[:, -kk:]  # [M, kk] [1,3,0, -4, 3,5,6,-2] -> [4,5,6]

    # check if true label is inside top-k
    in_topk = (topk_idx == labels_v[:, None]).any(axis=1)  # [M]

    # probability of the true label
    row_idx = np.arange(M)
    true_prob = probs[row_idx, labels_v]  # [M]

    meets_threshold = true_prob >= min_prob

    correct = in_topk & meets_threshold
    acc = float(correct.mean())
    return acc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens)

    true_predictions = np.concatenate(
        [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    )
    true_labels = np.concatenate(
        [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    )
    # TODO: (Malcolm 2025-01-22) should consider best averaging strategy
    precision = precision_score(true_labels, true_predictions, average="macro")
    recall = recall_score(true_labels, true_predictions, average="macro")
    accuracy = accuracy_score(true_labels, true_predictions)
    acc_new = topk_threshold_accuracy(logits, labels, k=3, min_prob=0.10)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "top3_accuracy": acc_new,
    }


def compute_metrics_multitask(eval_pred, *, task_names: list[str]):
    # In multitask case:
    # - logits is a lost of np arrays, one per task, with expected
    # shape (batch_size, seq_len, num_labels_for_task)
    # - labels is a list of np arrays, one per task, with expected
    # shape (batch_size, seq_len)
    logits_list, labels_list = eval_pred
    assert len(logits_list) == len(labels_list) == len(task_names)

    metrics = {}
    precisions = []
    recalls = []
    accuracies = []
    accuracies_top3 = []
    for task_name, logits, labels in zip(task_names, logits_list, labels_list):
        task_metrics = compute_metrics((logits, labels))
        precisions.append(task_metrics["precision"])
        recalls.append(task_metrics["recall"])
        accuracies.append(task_metrics["accuracy"])
        accuracies_top3.append(task_metrics["top3_accuracy"])
        for metric_name, metric_value in task_metrics.items():
            metrics[f"{task_name}_{metric_name}"] = metric_value

    metrics["precision"] = np.mean(precisions)
    metrics["recall"] = np.mean(recalls)
    metrics["accuracy"] = np.mean(accuracies)
    metrics[ "top3_accuracy"] = np.mean(accuracies_top3)

    return metrics


if __name__ == "__main__":
    batch_size = 2
    seq_len = 12
    num_labels = 8
    logits = np.random.randn(batch_size, seq_len, num_labels)
    labels = np.random.randint(0, num_labels, (batch_size, seq_len))

    print(compute_metrics((logits, labels)))
