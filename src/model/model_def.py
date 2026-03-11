import warnings
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable   # ← works on macOS builds


@register_keras_serializable(package="OpenForensics")
class PreprocessLayer(layers.Layer):
    """
    Serializable layer that wraps keras.applications preprocess_input.
    Receives input in [0,1], scales to [0,255], then applies model-specific preprocessing.
    """
    def __init__(self, mode='resnet', **kwargs):
        super().__init__(**kwargs)
        if mode not in ('resnet', 'vgg16'):
            raise ValueError("mode must be 'resnet' or 'vgg16'")
        self.mode = mode

    def call(self, inputs):
        x = inputs * 255.0
        if self.mode == 'resnet':
            return tf.keras.applications.resnet.preprocess_input(x)
        else:
            return tf.keras.applications.vgg16.preprocess_input(x)

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def safe_load_model(app_fn, *args, **kwargs):
    try:
        return app_fn(*args, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to load pretrained weights for {app_fn.__name__}: {e}. Falling back to weights=None")
        kwargs2 = kwargs.copy()
        kwargs2['weights'] = None
        return app_fn(*args, **kwargs2)


def build_ensemble(input_shape=(224, 224, 3)):
    inp = layers.Input(shape=input_shape, name='input_image')

    pre_r = PreprocessLayer(mode='resnet', name='preprocess_resnet')(inp)
    pre_v = PreprocessLayer(mode='vgg16', name='preprocess_vgg')(inp)

    base_r = safe_load_model(tf.keras.applications.ResNet50,
                             weights='imagenet', include_top=False, input_tensor=pre_r)

    base_v = safe_load_model(tf.keras.applications.VGG16,
                             weights='imagenet', include_top=False, input_tensor=pre_v)

    for m in (base_r, base_v):
        for layer in m.layers:
            layer.trainable = False

    def head(x):
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        return x

    h1 = head(base_r.output)
    h2 = head(base_v.output)

    merged = layers.Concatenate()([h1, h2])
    merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(256, activation='relu')(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.3)(merged)
    out = layers.Dense(1, activation='sigmoid', name='pred')(merged)

    model = Model(inputs=inp, outputs=out, name="resnet_vgg_ensemble")
    return model
