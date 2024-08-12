import tensorflow as tf

# Check if TensorFlow is using GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print(f"- {gpu.name}")
else:
    print("No GPU found. TensorFlow is using CPU.")