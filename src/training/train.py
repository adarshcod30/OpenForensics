import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from src.dataset.dataset_utils import sample_filepaths, make_dataset, IMG_SIZE
from src.model.model_def import build_ensemble

SEED = 12345
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def build_labels(files):
    labels = []
    for p in files:
        if os.path.sep + 'Real' + os.path.sep in p or '/Real/' in p:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def main(args):
    base_dir = args.base_dir
    classes = ['Fake', 'Real']

    train_files = sample_filepaths(base_dir, 'Train', classes, samples_per_class=args.train_per_class)
    val_files   = sample_filepaths(base_dir, 'Validation', classes, samples_per_class=args.val_per_class)
    test_files  = sample_filepaths(base_dir, 'Test', classes, samples_per_class=args.test_per_class)

    train_labels = build_labels(train_files)
    val_labels   = build_labels(val_files)
    test_labels  = build_labels(test_files)

    train_ds = make_dataset(train_files, train_labels, batch=args.batch, shuffle=True, augment=True)
    val_ds   = make_dataset(val_files, val_labels, batch=args.batch, shuffle=False, augment=False)
    test_ds  = make_dataset(test_files, test_labels, batch=args.batch, shuffle=False, augment=False)

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = build_ensemble(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        opt = optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.AUC(name='auc'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, 'best_model.keras')

    cb = [
        callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_auc', mode='max', save_best_only=True),
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=7, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, min_lr=1e-7),
        callbacks.CSVLogger(os.path.join(args.out_dir, 'training_log.csv')),
        callbacks.TensorBoard(log_dir=os.path.join(args.out_dir, 'tb_logs'))
    ]

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=cb)

    final_path = os.path.join(args.out_dir, 'final_model.keras')
    model.save(final_path)  # Keras infers format from extension

    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)

    res = model.evaluate(test_ds, return_dict=True)
    with open(os.path.join(args.out_dir, 'quick_test_eval.json'), 'w') as f:
        json.dump(res, f)

    print("Training complete. Models and logs saved to:", args.out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./Dataset')
    parser.add_argument('--out_dir', default='./runs/exp1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train_per_class', type=int, default=10000)
    parser.add_argument('--val_per_class', type=int, default=3000)
    parser.add_argument('--test_per_class', type=int, default=1000)
    args = parser.parse_args()
    main(args)
