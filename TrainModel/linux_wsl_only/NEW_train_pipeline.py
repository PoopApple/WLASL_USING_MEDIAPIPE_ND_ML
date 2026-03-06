"""activate venv =>


source /home/aryan/opensource_lab_proj/venv/bin/activate
"""

"""
installed using this guide
https://www.tensorflow.org/install/pip#windows-wsl2_1
https://developer.nvidia.com/cuda-12-3-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
"""


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import glob
import numpy as np
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("float32")
from tensorflow.keras import layers, models, callbacks, losses, optimizers, metrics

from datetime import datetime


modelnum = 0
with open("../models/max_model_num.txt", "r") as f:
    modelnum = int(f.read().strip())

modelpath = "../models"
landmarkpath = "../gte9_landmarks"

model_name = (
    modelpath
    + f"/{modelnum}_{datetime.now().strftime("%H-%M_%d-%m-%y")}_signlang_model.keras"
)


SEQUENCE_LENGTH = 70
NUM_LANDMARKS = 63
FEATURE_DIM = 63 * 4
NUM_FEATURES = 4
INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES)

BATCH_SIZE = 16  # INCREASED from 16 → GPU needs larger batches for efficiency (still safe for 5000 samples)
TRAIN_SPLIT = 0.25
LR = 1e-3
EPOCHS = 100
RANDOM_SEED = 1024
AUTOTUNE = tf.data.AUTOTUNE

NUM_CLASSES = 204


tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

modelnum = 0
with open("../models/max_model_num.txt", "r") as f:
    modelnum = int(f.read().strip())


print(modelnum)
# GPU CONFIG
gpus = tf.config.list_physical_devices("GPU")
print(gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU detected: {gpus[0].name}")

        # ENABLE MIXED PRECISION for faster GPU training (RTX 4050 supports FP16)
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✅ Mixed precision enabled: {policy.name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected, running on CPU.")


"""
BHAIIIII 57 ko 63 krna hi bhool gaya
sochra hu ye 1% kyu aari h aur loss itta saara
crazy
"""
# Use 85% for training, 15% for validation (with small data, need more training)





def load_dataset(LM_path=landmarkpath):
    """
    Scans root_folder/<word>/*.npy and returns (X, y, labels_map)
    X shape: (N, SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES)
    y: integer labels (N,)
    """
    word_dirs = [
        d for d in os.listdir(LM_path) if os.path.isdir(os.path.join(LM_path, d))
    ]
    word_dirs.sort()
    # print(word_dirs)
    # print(len(word_dirs))

    label_to_idx = {w: i for i, w in enumerate(word_dirs)}
    xs = []
    ys = []
    filenames = []
    for w in word_dirs:
        wordfolder = os.path.join(LM_path, w)
        # print(wordfolder)
        files = glob.glob(os.path.join(wordfolder, "*.npy"))
        # print(files)
        for f in files:
            arr = np.load(f)  # expect shape (70,63,4)
            # print(arr.shape)
            if arr.shape != (SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES):
                raise ValueError(f"Unexpected shape {arr.shape} for file {f}")
            xs.append(arr.astype(np.float32))
            ys.append(label_to_idx[w])
            filenames.append(f)
    X = np.stack(xs, axis=0)

    y = np.array(ys, dtype=np.int32)
    # print(X.shape)
    # print(y)
    # print(label_to_idx)
    # print(len(filenames))
    return X, y, label_to_idx, filenames


def random_frame_dropout(seq, drop_rate=0.05):
    # randomly drop frames and pad with zeros (simple)
    seq = seq.copy()
    if np.random.rand() < drop_rate:
        k = np.random.randint(1, max(2, int(SEQUENCE_LENGTH * 0.1)))
        idx = np.random.randint(0, SEQUENCE_LENGTH - k + 1)
        seq[idx : idx + k] = 0.0
    return seq


def random_noise(seq, sigma=0.01):
    # add small gaussian noise to x,y,z only (not visibility)
    seq = seq.copy()
    noise = np.random.normal(scale=sigma, size=seq[..., :3].shape)
    # print(noise.shape)
    seq[..., :3] += noise
    return seq


"not using"


def random_scale_rotate(seq, scale_range=(0.95, 1.05), rot_angle_deg=5.0):
    # apply scale and small rotation about origin (shoulder-centered preproc)
    seq = seq.copy()
    angle = np.deg2rad(np.random.uniform(-rot_angle_deg, rot_angle_deg))
    s = np.random.uniform(*scale_range)
    c, s_ang = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s_ang], [s_ang, c]], dtype=np.float32)
    # apply rotation+scale to x,y only
    xy = seq[..., :2].reshape(-1, 2)
    xy = (xy @ R.T) * s
    seq[..., :2] = xy.reshape(seq[..., :2].shape)
    return seq


def make_tf_dataset(X, y, batch_size=BATCH_SIZE, training=True, augment=True):
    num_classes = np.max(y) + 1
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    ds = tf.data.Dataset.from_tensor_slices((X, y_onehot))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=RANDOM_SEED)

    def _augment(x, y):
        # numpy augmentation inside tf.py_function
        def aug_np(x_np):
            if np.random.rand() < 0.7 and augment:
                x_np = random_frame_dropout(x_np, drop_rate=0.05)
                x_np = random_noise(x_np, sigma=0.01)
                x_np = random_scale_rotate(
                    x_np, scale_range=(0.96, 1.04), rot_angle_deg=7.0
                )
            return x_np

        x_aug = tf.py_function(func=aug_np, inp=[x], Tout=tf.float32)
        x_aug.set_shape((SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES))
        return x_aug, y

    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


class FrameAttention(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)

    def call(self, inputs, mask=None):
        # inputs shape: (batch, time, features)
        score = self.v(self.W(inputs))  # (batch, time, 1)
        score = tf.squeeze(score, axis=-1)  # (batch, time)
        if mask is not None:
            score += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        attn = tf.nn.softmax(score, axis=1)  # (batch, time)
        attn = tf.expand_dims(attn, axis=-1)  # (batch, time, 1)
        weighted = inputs * attn
        return tf.reduce_sum(weighted, axis=1)  # (batch, features)


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, dropout_rate=0.35):
    inp = layers.Input(shape=input_shape)  # (70,63,4)

    # Per-frame spatial processing: flatten landmarks and project
    x = layers.Reshape((input_shape[0], input_shape[1] * input_shape[2]))(
        inp
    )  # (time, 63*4=252)
    x = layers.LayerNormalization()(x)
    x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dropout(0.1))(x)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)

    # BiLSTM stack
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, recurrent_dropout=0.1)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(192, return_sequences=True, recurrent_dropout=0.1)
    )(x)

    # Self-attention / frame attention pooling
    attn_out = FrameAttention(units=128)(x)  # (batch, features)

    # Dense head
    h = layers.LayerNormalization()(attn_out)
    h = layers.Dense(256, activation="relu")(h)
    h = layers.Dropout(dropout_rate)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(dropout_rate / 2)(h)

    out = layers.Dense(num_classes, activation="softmax")(h)

    model = models.Model(inputs=inp, outputs=out, name="ASL_BiLSTM_Attn")
    return model


def make_callbacks(save_path=model_name):
    cb = []
    cb.append(
        callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
        )
    )
    cb.append(
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )
    )
    cb.append(
        callbacks.ModelCheckpoint(
            save_path, monitor="val_loss", save_best_only=True, verbose=1
        )
    )
    cb.append(callbacks.CSVLogger("training_log.csv"))
    cb.append(callbacks.TensorBoard(log_dir="tb_logs"))
    return cb


if __name__ == "__main__":

    print("Loading data...")
    X, y, label_map, files = load_dataset(landmarkpath)
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}, classes: {len(label_map)}")
    NUM_CLASSES = len(label_map)

    # simple stratified split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )

    train_ds = make_tf_dataset(
        X_train, y_train, batch_size=BATCH_SIZE, training=True, augment=True
    )
    val_ds = make_tf_dataset(
        X_val, y_val, batch_size=BATCH_SIZE, training=False, augment=False
    )

    model = build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    model.summary()

    # compile
    opt = optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=opt,
        loss=losses.CategoricalCrossentropy(label_smoothing=0.01),
        metrics=[
            metrics.CategoricalAccuracy(name="acc"),
            TopKFloat32(k=3, name="top3_acc"),
        ],
    )

    cbs = make_callbacks()
    print("Starting training...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=2
    )

    # Save label map for inference
    import json

    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    with open("../models/max_model_num.txt", "w") as f:
        f.write(str(modelnum + 1))
