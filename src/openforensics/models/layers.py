"""Custom layers.

Kept in their own module so that loading a saved model only needs to import
this, not the whole training stack.
"""
from __future__ import annotations

import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable

# Each Keras application expects its own input convention. Getting this wrong
# is silent: the model still trains, just from a worse starting point.
#   resnet50 / vgg16      -> caffe-style, [0,255] then mean-subtracted
#   efficientnetv2        -> raw [0,255]; the network rescales internally,
#                            so its preprocess_input is a documented no-op
_MODES = ("resnet50", "vgg16", "efficientnetv2")


@register_keras_serializable(package="OpenForensics")
class PreprocessLayer(layers.Layer):
    """Adapt a shared [0,1] input tensor to one backbone's expected scaling."""

    def __init__(self, mode: str = "resnet50", **kwargs):
        super().__init__(**kwargs)
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode

    def call(self, inputs):
        x = inputs * 255.0
        if self.mode == "resnet50":
            return tf.keras.applications.resnet.preprocess_input(x)
        if self.mode == "vgg16":
            return tf.keras.applications.vgg16.preprocess_input(x)
        return x  # efficientnetv2 normalises inside the backbone

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


# The original model_def.py used mode='resnet'/'vgg16'. Old checkpoints carry
# those strings, so accept them when deserialising rather than breaking loads.
_LEGACY = {"resnet": "resnet50", "vgg": "vgg16"}


@register_keras_serializable(package="OpenForensics")
class LegacyPreprocessLayer(PreprocessLayer):
    def __init__(self, mode: str = "resnet", **kwargs):
        super().__init__(mode=_LEGACY.get(mode, mode), **kwargs)
