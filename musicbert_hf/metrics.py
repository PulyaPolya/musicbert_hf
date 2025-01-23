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

    # TODO: (Malcolm 2025-01-22) consider averaging
    precision = precision_score(true_labels, true_predictions, average="macro")
    recall = recall_score(true_labels, true_predictions, average="macro")
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    batch_size = 2
    seq_len = 12
    num_labels = 8
    logits = np.random.randn(batch_size, seq_len, num_labels)
    labels = np.random.randint(0, num_labels, (batch_size, seq_len))

    print(compute_metrics((logits, labels)))
