import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
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
    for task_name, logits, labels in zip(task_names, logits_list, labels_list):
        task_metrics = compute_metrics((logits, labels))
        precisions.append(task_metrics["precision"])
        recalls.append(task_metrics["recall"])
        accuracies.append(task_metrics["accuracy"])
        for metric_name, metric_value in task_metrics.items():
            metrics[f"{task_name}_{metric_name}"] = metric_value

    metrics["precision"] = np.mean(precisions)
    metrics["recall"] = np.mean(recalls)
    metrics["accuracy"] = np.mean(accuracies)

    return metrics


if __name__ == "__main__":
    batch_size = 2
    seq_len = 12
    num_labels = 8
    logits = np.random.randn(batch_size, seq_len, num_labels)
    labels = np.random.randint(0, num_labels, (batch_size, seq_len))

    print(compute_metrics((logits, labels)))
