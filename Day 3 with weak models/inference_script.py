import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt

# Paths to the models
tflite_model_path = 'models/custom_tinyml128.tflite'
quantized_model_path = 'models/TINYml_best_quantized_int8.tflite'
h5_model_path = 'models/TinyML128Best.h5'  # Path to the H5 model

# Load TFLite model (regular and quantized)
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load H5 model
def load_h5_model(model_path):
    return tf.keras.models.load_model(model_path)

# Preprocess the image for the TFLite model (expects FLOAT32 input)
def preprocess_image_for_tflite(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1] for FLOAT32 input
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Preprocess the image for Quantized TFLite model (expects UINT8 input)
def preprocess_image_for_quant_tflite(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    img = cv2.resize(img, img_size)
    img = img.astype(np.uint8)  # Keep in [0, 255] range for UINT8 input
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Preprocess the image for H5 model
def preprocess_image_for_h5(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to get predictions from TFLite model
def predict_tflite(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)
    return output_data

# Function to get predictions from H5 model
def predict_h5(model, input_data):
    return model.predict(input_data)

# Load both TFLite models
tflite_interpreter = load_tflite_model(tflite_model_path)
quant_tflite_interpreter = load_tflite_model(quantized_model_path)

# Load H5 model
h5_model = load_h5_model(h5_model_path)

# Run inference based on user choice
def run_inference(choice, image_path):
    # Preprocess and predict based on the user's choice
    if choice == 1:
        # TFLite model (FLOAT32)
        image_tflite = preprocess_image_for_tflite(image_path, img_size=(128, 128))
        if image_tflite is not None:
            tflite_prediction = predict_tflite(tflite_interpreter, image_tflite)
            predicted_label_tflite = np.argmax(tflite_prediction)
            print(f"Predicted label: {predicted_label_tflite}")
    elif choice == 2:
        # Quantized TFLite model (UINT8)
        image_quant_tflite = preprocess_image_for_quant_tflite(image_path, img_size=(128, 128))
        if image_quant_tflite is not None:
            quant_tflite_prediction = predict_tflite(quant_tflite_interpreter, image_quant_tflite)
            predicted_label_quant_tflite = np.argmax(quant_tflite_prediction)
            print(f"Predicted label: {predicted_label_quant_tflite}")
    elif choice == 3:
        # H5 model
        image_h5 = preprocess_image_for_h5(image_path, img_size=(128, 128))
        if image_h5 is not None:
            h5_prediction = predict_h5(h5_model, image_h5)
            predicted_label_h5 = np.argmax(h5_prediction)
            print(f"Predicted label: {predicted_label_h5}")
    else:
        print("Invalid choice! Please choose 1, 2, or 3.")

# Main function to parse arguments and run the program
def main():
    # Ensure correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python3 inference.py path/to/image model_choice (1, 2, or 3)")
        return

    # Get image path and model choice from command line arguments
    image_path = sys.argv[1]
    try:
        choice = int(sys.argv[2])
    except ValueError:
        print("Invalid model choice. Please enter 1, 2, or 3.")
        return

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: The image path '{image_path}' does not exist.")
        return

    # Display the image
    img_to_display = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img_to_display, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Run inference on chosen model
    run_inference(choice, image_path)

if __name__ == "__main__":
    main()
