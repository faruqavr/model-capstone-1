import os
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the TensorFlow Lite model
model_path = os.path.join(os.path.dirname(__file__), "model-1.tflite")

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=model_path)

# Allocate tensors
interpreter.allocate_tensors()

interpreter.allocate_tensors()
# Define input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Preprocess input data
    input_shape = input_details[0]['shape']
    input_data = np.zeros(input_shape, dtype=np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process output data
    predicted_result = process_output_data(output_data)

    # Return the predicted results
    return jsonify({'prediction': predicted_result})


if __name__ == '__main__':
    app.run()
