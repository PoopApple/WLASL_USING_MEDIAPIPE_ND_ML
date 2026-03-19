import tensorflow as tf

print("--- TensorFlow Hardware Report ---")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Devices found: {tf.config.list_physical_devices()}")
gpu_devices = tf.config.list_physical_devices("GPU")
if gpu_devices:
    print(f"✅ Success! GPU found: {gpu_devices}")
else:
    print("❌ No GPU found. Still falling back to CPU.")


"""LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${PWD}/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:${PWD}/.venv/lib/python3.12/site-packages/nvidia/cudart/lib:${LD_LIBRARY_PATH}"""
