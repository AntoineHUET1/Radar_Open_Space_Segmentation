import tensorflow as tf

# Vérifier la version de TensorFlow
print("Version de TensorFlow :", tf.__version__)

# Vérifier si un GPU est disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU détecté :", gpus)
else:
    print("Aucun GPU détecté")
