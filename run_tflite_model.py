import tensorflow as tf
import numpy as np

# Load the TensorFlow Lite model.
interpreter = tf.lite.Interpreter(model_path="model-1.tflite")
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data (sample input shape: [1, 224, 224, 3]).
input_shape = input_details[0]['shape']
input_data = np.zeros(input_shape, dtype=np.float32)  # Set input data accordingly.

# Set input tensor.
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference.
interpreter.invoke()

# Get output tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

# Process output data.
# ...

# Example: Print the predicted class label (if the model is a classifier).
predicted_class = np.argmax(output_data)
print("Predicted class:", predicted_class)
