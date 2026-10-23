"""Classical baseline in Keras on the TensorFlow backend: TF-IDF bigrams + dense net.

Gives the fine-tuned transformer something honest to beat.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.data import load_banking77  # noqa: E402


def build(num_features, num_labels):
    return keras.Sequential([
        keras.Input(shape=(num_features,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_labels, activation="softmax"),
    ])


def main():
    keras.utils.set_random_seed(42)
    ds, labels = load_banking77()
    x_tr, y_tr = np.array(ds["train"]["text"]), np.array(ds["train"]["label"])
    # the CSV is grouped by intent, so shuffle before Keras takes its last-10% validation split
    order = np.random.default_rng(42).permutation(len(x_tr))
    x_tr, y_tr = x_tr[order], y_tr[order]
    x_te, y_te = np.array(ds["test"]["text"]), np.array(ds["test"]["label"])
    vec = keras.layers.TextVectorization(max_tokens=20000, ngrams=2, output_mode="tf_idf")
    vec.adapt(tf.constant(x_tr))
    f_tr, f_te = vec(tf.constant(x_tr)).numpy(), vec(tf.constant(x_te)).numpy()
    model = build(f_tr.shape[1], len(labels))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    model.fit(f_tr, y_tr, epochs=15, batch_size=64, validation_split=0.1, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    _, acc = model.evaluate(f_te, y_te, verbose=0)
    out = {"test_accuracy": round(float(acc), 4), "train_minutes": round((time.time() - t0) / 60, 1),
           "tensorflow": tf.__version__, "keras": keras.__version__}
    Path("results/keras_baseline.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
