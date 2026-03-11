# finetune.py
"""
Fine-tune a saved ensemble model (ResNet50+VGG16) by unfreezing the last N layers
of each backbone and continuing training with a lower learning rate.

Usage example:
python finetune.py \
  --base_dir ./Dataset \
  --model_path ./runs/exp1/best_model.keras \
  --out_dir ./runs/exp1_finetune \
  --epochs 10 \
  --batch 16 \
  --lr 1e-5 \
  --train_per_class 10000 \
  --val_per_class 3000 \
  --test_per_class 1000 \
  --unfreeze_last 50
"""
import os
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report

from src.model import model_def
from src.dataset.dataset_utils import sample_filepaths, make_dataset, IMG_SIZE

# fix reproducibility seed if desired
SEED = 12345
tf.random.set_seed(SEED)


def build_labels(files):
    return [1 if (os.path.sep + 'Real' + os.path.sep) in p or '/Real/' in p else 0 for p in files]


def find_backbone_models(model):
    """
    Return list of (name, submodel) for nested submodels that look like backbones.
    We search for instances of tf.keras.Model inside the main Functional model.
    """
    backbones = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            lname = layer.name.lower()
            if 'resnet' in lname or 'vgg' in lname or 'mobilenet' in lname or 'efficient' in lname:
                backbones.append((layer.name, layer))
    # as a fallback, search by known layer names inside model (older saves embed models differently)
    if not backbones:
        for layer in model.layers:
            lname = layer.name.lower()
            if 'resnet50' in lname:
                backbones.append((layer.name, layer))
            if 'vgg16' in lname:
                backbones.append((layer.name, layer))
    return backbones


def unfreeze_last_n_layers_of_backbones(model, n_last):
    """
    Unfreeze the last n_last layers of each backbone-like submodel found in the model.
    Returns the number of layers changed.
    """
    changed = 0
    backbones = find_backbone_models(model)
    if not backbones:
        warnings.warn("No nested backbone tf.keras.Model objects found — attempting global unfreeze")
        # fallback: unfreeze last n_last layers of the whole model
        all_layers = [l for l in model.layers if hasattr(l, 'trainable')]
        for l in all_layers[-n_last:]:
            if not l.trainable:
                l.trainable = True
                changed += 1
        return changed

    for name, sub in backbones:
        layers_list = [l for l in sub.layers if hasattr(l, 'trainable')]
        # guard: if n_last > total, unfreeze all
        n_to_unfreeze = min(n_last, len(layers_list))
        for l in layers_list[-n_to_unfreeze:]:
            if not l.trainable:
                l.trainable = True
                changed += 1
    return changed


def save_roc_pr(y_true, y_score, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(loc='lower right')
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(loc='lower left')
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'))
    plt.close()
    return {'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)}


def evaluate_and_save(model, test_ds, out_dir, thresh=0.5):
    # predict all
    y_scores = []
    y_trues = []
    for x, y in test_ds:
        preds = model.predict(x)
        y_scores.append(preds.ravel())
        y_trues.append(y.numpy().ravel())
    y_scores = np.concatenate(y_scores)
    y_trues = np.concatenate(y_trues)

    os.makedirs(out_dir, exist_ok=True)
    aucs = save_roc_pr(y_trues, y_scores, out_dir)

    y_pred = (y_scores >= thresh).astype(int)
    cm = confusion_matrix(y_trues, y_pred).tolist()
    report = classification_report(y_trues, y_pred, target_names=['Fake', 'Real'], output_dict=True)

    eval_dict = {
        'roc_auc': aucs['roc_auc'],
        'pr_auc': aucs['pr_auc'],
        'confusion_matrix': cm,
        'classification_report': report
    }

    with open(os.path.join(out_dir, 'eval_report.json'), 'w') as f:
        json.dump(eval_dict, f, indent=2)

    return eval_dict


def main(args):
    # sample files
    classes = ['Fake', 'Real']
    print("Sampling filepaths (this will error if insufficient files exist)...")
    train_files = sample_filepaths(args.base_dir, 'Train', classes, samples_per_class=args.train_per_class)
    val_files = sample_filepaths(args.base_dir, 'Validation', classes, samples_per_class=args.val_per_class)
    test_files = sample_filepaths(args.base_dir, 'Test', classes, samples_per_class=args.test_per_class)

    train_labels = build_labels(train_files)
    val_labels = build_labels(val_files)
    test_labels = build_labels(test_files)

    train_ds = make_dataset(train_files, train_labels, batch=args.batch, shuffle=True, augment=True)
    val_ds = make_dataset(val_files, val_labels, batch=args.batch, shuffle=False, augment=False)
    test_ds = make_dataset(test_files, test_labels, batch=args.batch, shuffle=False, augment=False)

    # load model (supports custom layer)
    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path, custom_objects={'PreprocessLayer': model_def.PreprocessLayer})

    # Unfreeze last n layers of each backbone
    print(f"Unfreezing last {args.unfreeze_last} layers of backbone submodels...")
    changed = unfreeze_last_n_layers_of_backbones(model, args.unfreeze_last)
    print(f"Total layers set trainable: {changed}")

    # Recompile with low LR
    opt = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

    # Prepare out_dir and callbacks
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_dir = os.path.join(args.out_dir, 'tb_logs', timestamp)
    os.makedirs(tb_dir, exist_ok=True)

    ckpt_path = os.path.join(args.out_dir, 'best_model_finetuned.keras')
    cb_list = [
        callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_auc', mode='max', save_best_only=True),
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-8),
        callbacks.CSVLogger(os.path.join(args.out_dir, 'finetune_log.csv')),
        callbacks.TensorBoard(log_dir=tb_dir)
    ]

    # Fit
    print("Starting fine-tuning...")
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=cb_list)

    # Save final model
    final_path = os.path.join(args.out_dir, 'final_finetuned.keras')
    model.save(final_path)

    # Save history
    with open(os.path.join(args.out_dir, 'history_finetune.json'), 'w') as f:
        json.dump(history.history, f, indent=2)

    # Quick evaluate and save results
    print("Evaluating fine-tuned model on test set...")
    eval_outdir = os.path.join(args.out_dir, 'eval')
    eval_report = evaluate_and_save(model, test_ds, eval_outdir, thresh=args.threshold)
    print("Fine-tune evaluation saved to", eval_outdir)
    print("Eval summary:", json.dumps({'roc_auc': eval_report['roc_auc'], 'pr_auc': eval_report['pr_auc']}, indent=2))

    # done
    print("Fine-tuning complete. Models and logs are in:", args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./Dataset', help='dataset root containing Train/Validation/Test')
    parser.add_argument('--model_path', required=True, help='path to the initial best_model.keras')
    parser.add_argument('--out_dir', default='./runs/exp1_finetune', help='where to save finetune outputs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for fine-tuning')
    parser.add_argument('--train_per_class', type=int, default=10000)
    parser.add_argument('--val_per_class', type=int, default=3000)
    parser.add_argument('--test_per_class', type=int, default=1000)
    parser.add_argument('--unfreeze_last', type=int, default=50, help='number of last layers to unfreeze per backbone')
    parser.add_argument('--threshold', type=float, default=0.5, help='classification threshold for evaluation')
    args = parser.parse_args()
    main(args)
