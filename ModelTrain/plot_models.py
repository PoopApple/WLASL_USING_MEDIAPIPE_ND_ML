import os
import tensorflow as tf

# Import your model zoo
from model import MODEL_REGISTRY

# =========================================================
# SETTINGS
# =========================================================

OUTPUT_DIR = "model_diagrams_vertical"
NUM_CLASSES = 500  # change if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# GENERATE DIAGRAMS
# =========================================================

for model_name, build_fn in MODEL_REGISTRY.items():
    print(f"\nBuilding: {model_name}")

    try:
        model = build_fn(NUM_CLASSES)

        # Save PNG diagram
        output_path = os.path.join(OUTPUT_DIR, f"{model_name}.png")

        # tf.keras.utils.plot_model(
        #     model,
        #     to_file=output_path,
        #     show_shapes=True,
        #     show_dtype=False,
        #     show_layer_names=True,
        #     rankdir="TB",  # TB = top-bottom
        #     expand_nested=True,
        #     dpi=200,
        # )
        tf.keras.utils.plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
            expand_nested=True,
            dpi=600,
            rankdir="TB"
        )

        print(f"Saved: {output_path}")

        # Print summary info
        print(f"Parameters: {model.count_params():,}")

    except Exception as e:
        print(f"FAILED: {model_name}")
        print(e)

print("\nDone.")
