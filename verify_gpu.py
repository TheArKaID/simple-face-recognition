import tensorflow as tf

# ...existing code...
print("GPUs Available:", tf.config.list_physical_devices("GPU"))