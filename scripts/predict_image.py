import argparse
import json
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

# Add project root to path so we can import src
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import model_def so that any custom layers (PreprocessLayer) are registered.
from src.model import model_def

def load_image_array(path, img_size=(224,224)):
    img = image.load_img(path, target_size=img_size)
    arr = image.img_to_array(img).astype('float32') / 255.0
    return np.expand_dims(arr, 0)

def load_model_safe(model_path):
    """
    Try to load model normally. If deserialization fails due to custom layer,
    retry with custom_objects pointing to model_def.PreprocessLayer.
    """
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        # Attempt safe fallback using custom_objects
        try:
            return tf.keras.models.load_model(model_path, compile=False,
                                              custom_objects={'PreprocessLayer': model_def.PreprocessLayer})
        except Exception as e2:
            raise RuntimeError(f"Failed to load model normally ({e}). "
                               f"Fallback with custom_objects also failed ({e2}).") from e2

def predict_single(model, img_path, thresh=0.5):
    x = load_image_array(img_path)
    prob = float(model.predict(x).ravel()[0])
    label = "Real" if prob >= thresh else "Fake"
    return {"image": img_path, "probability_real": prob, "predicted_label": label}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('img', help='path to image')
    p.add_argument('--model', required=True, help='path to .keras model')
    p.add_argument('--threshold', type=float, default=0.5, help='decision threshold for class label')
    args = p.parse_args()

    try:
        model = load_model_safe(args.model)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        out = predict_single(model, args.img, thresh=args.threshold)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()
