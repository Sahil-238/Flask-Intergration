import os
import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import torchvision

# Import your model from the existing code
from Text2FaceImageGeneration import VAE  # Adjust this import based on your model class location

# Create Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)  # Assuming VAE is your model class in Text2FaceImageGeneration.py
model.load_state_dict(torch.load('vae_model_epoch_final.pth'))  # Load the trained model file
model.eval()  # Set model to evaluation mode

# Image transformation to match input format for the model
transform = transforms.Compose([
    transforms.Resize((56, 56)),  # Resize to match input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image from the request
        image = Image.open(file.stream).convert('RGB')

        # Apply transformations to the image
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device (CPU or GPU)

        # Run the image through the model to get the reconstruction
        with torch.no_grad():
            reconstructed, _ = model(image)

        # Convert the output to a usable image format
        reconstructed_image = reconstructed.squeeze(0).cpu()  # Remove batch dimension and move back to CPU
        reconstructed_image = (reconstructed_image * 0.5) + 0.5  # Rescale to [0, 1] range

        # Save the reconstructed image (optional)
        save_path = 'reconstructed_image.png'
        torchvision.utils.save_image(reconstructed_image, save_path)

        # Return the path of the saved image in response
        return jsonify({'message': 'Prediction successful', 'reconstructed_image': save_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Health check route (just to check if the server is running)
@app.route('/')
def index():
    return "Flask app is running!"


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
