from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import torch
import os
from models import DeepLSTMModel, HybridKhmerRecognizer  # Ensure this model is correctly defined in your models directory
from utils import (
    extract_coordinates, scale_coordinates, split_to_substroke, english_to_khmer_digit,
    prediction_correctness_to_word, preprocess_drawing, KHMER_CHARACTER_MAP
)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes (allow requests from any origin, you can be more specific here)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Global model variables
digit_model = None
char_model = None

def load_digit_model():
    """Load the digit recognition model."""
    try:
        model_path = os.path.join("models", "best_lstm_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Digit model file '{model_path}' not found!")
        input_size = 16
        hidden_size = 256
        num_layers = 4
        label_output_size = 10
        correctness_output_size = 2
        dropout_rate = 0.3
        model = DeepLSTMModel(input_size, hidden_size, num_layers, label_output_size, correctness_output_size, dropout_rate)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading digit model: {e}")
        raise

def load_char_model():
    """Load the character recognition model."""
    try:
        model_path = os.path.join("models", "best-checkpoint.ckpt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Character model file '{model_path}' not found!")

        # Define model parameters
        INPUT_DIM = 16
        CNN_OUT_CHANNELS = 128
        RNN_HIDDEN_DIM = 256
        NUM_LAYERS = 2
        NUM_CLASSES = 119  # Total number of Khmer characters in the model
        DROPOUT_PROB = 0.4

        # Initialize model
        model = HybridKhmerRecognizer(
            input_dim=INPUT_DIM, 
            cnn_out_channels=CNN_OUT_CHANNELS, 
            rnn_hidden_dim=RNN_HIDDEN_DIM,
            num_layers=NUM_LAYERS, 
            num_classes=NUM_CLASSES, 
            dropout_prob=DROPOUT_PROB
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()  # Set to evaluation mode
        return model

    except Exception as e:
        print(f"Error loading character model: {e}")
        raise

@app.route('/recognize-digit', methods=['POST'])
def recognize_digit():
    """API endpoint for recognizing Khmer digits."""
    try:
        # Get JSON data from the request
        canvas_data = request.get_json()

        # Extract the coordinates from the canvas data
        list_coords = extract_coordinates(canvas_data)
        if not list_coords:
            return jsonify({"error": "No valid drawing data provided"}), 400

        # Normalize the data
        normalized_data = scale_coordinates(list_coords)
        if not normalized_data:
            return jsonify({"error": "Failed to normalize coordinates"}), 400

        # Split data into substrokes
        nested_coords = split_to_substroke(normalized_data)
        if not nested_coords:
            return jsonify({"error": "Failed to split coordinates into substrokes"}), 400

        # Convert data to tensor
        tensor_data = torch.tensor(nested_coords, dtype=torch.float32)
        tensor_data = tensor_data.unsqueeze(0)  # Add batch dimension

        # Perform inference with the digit model
        with torch.no_grad():
            label_out, correctness_out = digit_model(tensor_data)
            label_preds = torch.argmax(label_out, dim=1)
            correct_preds = torch.argmax(correctness_out, dim=1)

            khmer_number_prediction = english_to_khmer_digit(label_preds)
            correctness = prediction_correctness_to_word(correct_preds)

        # Return the prediction result
        return jsonify({
            "predicted_digit": khmer_number_prediction,
            "writing_quality": correctness
        })
    
    except Exception as e:
        print(f"Error in recognize_digit: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/recognize-character', methods=['POST'])
def recognize_character():
    """API endpoint for recognizing Khmer characters."""
    try:
        # Get JSON data from the request
        request_data = request.get_json()

        # Extract 'objects' from the JSON body
        objects = request_data.get('objects', [])

        # Process the drawing data
        processed_input = preprocess_drawing({"objects": objects}, input_dim=16)
        if processed_input is None:
            return jsonify({"error": "Not enough drawing data to make a prediction"}), 400

        # Perform inference with the character model
        with torch.no_grad():
            output = char_model(processed_input)
            _, predicted_index = torch.max(output.data, 1)
            predicted_char = KHMER_CHARACTER_MAP[predicted_index.item()]

        return jsonify({
            "predicted_character": predicted_char
        })
    
    except Exception as e:
        print(f"Error in recognize_character: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # Load the models before starting the server
    digit_model = load_digit_model()
    char_model = load_char_model()

    # Start Flask server
    app.run(debug=True, host="0.0.0.0", port=8000)
