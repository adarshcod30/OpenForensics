import os
import random
import tensorflow as tf

SEED = 12345
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)

def sample_filepaths(base_dir, split, classes, samples_per_class):
    out = []
    for cls in classes:
        cls_dir = os.path.join(base_dir, split, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) < samples_per_class:
            raise ValueError(f"Not enough images for {split}/{cls}: found {len(files)}, need {samples_per_class}")
        random.Random(SEED).shuffle(files)
        out += files[:samples_per_class]
    random.Random(SEED).shuffle(out)
    return out

def preprocess_image(path, label, img_size=IMG_SIZE):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_fn(image, label):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_brightness(image, 0.08, seed=SEED)
    image = tf.image.random_contrast(image, 0.95, 1.05, seed=SEED)
    return image, label

def make_dataset(filepaths, labels, batch=32, shuffle=True, augment=False):
    paths = tf.constant(filepaths)
    labs = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labs))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED)
    ds = ds.map(lambda p, l: preprocess_image(p, l), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds
