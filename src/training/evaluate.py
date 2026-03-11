# evaluate.py
import os
import json
import argparse
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report

from src.model import model_def
from src.dataset.dataset_utils import sample_filepaths, make_dataset

def safe_load_model(path):
    """
    Try to load model normally, then fallback to custom_objects if needed.
    """
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e1:
        try:
            return tf.keras.models.load_model(path, compile=False,
                                              custom_objects={'PreprocessLayer': model_def.PreprocessLayer})
        except Exception as e2:
            raise RuntimeError(f"Failed to load model.\nPrimary error: {e1}\nFallback error: {e2}")

def predict_probs(model, ds):
    probs = []
    labels = []
    for x, y in ds:
        p = model.predict(x)
        probs.append(p.ravel())
        labels.append(y.numpy().ravel())
    if len(probs) == 0:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(labels)

def save_roc_pr_curves(y_true, y_score, out_dir):
    """
    Saves roc_curve.png and pr_curve.png if computable.
    Returns (roc_auc or None, pr_auc or None, note)
    """
    note = None
    roc_auc = None
    pr_auc = None

    # Need at least two classes present to compute ROC/PR properly
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        note = "Only one class present in y_true; ROC/PR curves cannot be computed."
        return roc_auc, pr_auc, note

    # safe numeric arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # compute ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = float(auc(fpr, tpr))
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
        plt.close()
    except Exception as e:
        note = (note + " | " if note else "") + f"ROC curve generation failed: {e}"

    # compute PR
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(recall, precision))
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'pr_curve.png'))
        plt.close()
    except Exception as e:
        note = (note + " | " if note else "") + f"PR curve generation failed: {e}"

    return roc_auc, pr_auc, note

def main(args):
    t0 = time.time()

    # load model
    model = safe_load_model(args.model_path)

    # sample test files
    classes = ['Fake', 'Real']
    test_files = sample_filepaths(args.base_dir, 'Test', classes, samples_per_class=args.test_per_class)
    test_labels = [1 if (os.path.sep + 'Real' + os.path.sep) in p or '/Real/' in p else 0 for p in test_files]

    test_ds = make_dataset(test_files, test_labels, batch=args.batch, shuffle=False, augment=False)

    y_score, y_true = predict_probs(model, test_ds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if y_score.size == 0:
        raise RuntimeError("No predictions produced. Check dataset paths, make_dataset, and batch size.")

    # ensure 1d arrays
    y_score = np.asarray(y_score).ravel()
    y_true = np.asarray(y_true).ravel()

    # save curves and compute aucs if possible
    roc_auc, pr_auc, note = save_roc_pr_curves(y_true, y_score, str(out_dir))

    # classification at threshold
    thresh = args.threshold
    y_pred = (y_score >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'], output_dict=True)

    # build report
    report_dict = {
        'model_path': args.model_path,
        'num_test_samples': int(y_true.size),
        'threshold': float(thresh),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'note': note,
        'runtime_seconds': round(time.time() - t0, 3)
    }

    # write json
    with open(out_dir / 'eval_report.json', 'w') as f:
        json.dump(report_dict, f, indent=2)

    print("Evaluation saved to", str(out_dir))
    if note:
        print("Note:", note)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--base_dir', default='./Dataset')
    parser.add_argument('--out_dir', default='./runs/exp1/eval')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--test_per_class', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
