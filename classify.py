# classify.py
#
# This script runs a pre-trained PyTorch AI model (ResNet18) to classify
# an image. It is designed for the RDK X5 AI Vision Challenge.
#
# Steps it performs:
# 1. Loads a pre-trained image classification model.
# 2. Downloads the list of human-readable labels (ImageNet classes).
# 3. Loads and pre-processes the 'sample_image.png'.
# 4. Performs inference (makes a prediction).
# 5. Prints the top prediction and its confidence score.

import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

# --- Configuration ---
MODEL_NAME = "ResNet18"
IMAGE_PATH = "sample_image.png"
# URL to a raw JSON file containing the 1000 ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def get_model():
    """
    Loads and returns a pre-trained ResNet18 model in evaluation mode.
    The first time this runs, it will download the model weights.
    """
    print(f"Loading pre-trained model: {MODEL_NAME}...")
    # Load a model pre-trained on the ImageNet dataset
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Set the model to evaluation mode. This is important for inference.
    model.eval()
    print("Model loaded successfully.")
    return model

def get_labels():
    """
    Downloads and returns the list of ImageNet class labels.
    """
    print(f"Downloading class labels from {LABELS_URL}...")
    with urllib.request.urlopen(LABELS_URL) as url:
        labels = json.loads(url.read().decode())
    print("Labels downloaded successfully.")
    return labels

def process_image(image_path):
    """
    Loads an image and applies the necessary transformations for the model.
    """
    print(f"Processing image: {image_path}...")
    # Transformations must match what the model was trained on.
    # 1. Resize to 256x256
    # 2. Center crop to 224x224
    # 3. Convert to a PyTorch Tensor
    # 4. Normalize with ImageNet's mean and standard deviation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Open the image file
    img = Image.open(image_path).convert('RGB')
    
    # Apply the transformations
    img_t = preprocess(img)
    
    # The model expects a batch of images, so we add a "batch" dimension of 1.
    # [3, 224, 224] -> [1, 3, 224, 224]
    batch_t = torch.unsqueeze(img_t, 0)
    print("Image processed.")
    return batch_t


def predict(model, image_tensor, labels):
    """
    Performs inference and returns the top prediction.
    """
    print("Running AI inference...")
    # Perform inference without calculating gradients
    with torch.no_grad():
        output = model(image_tensor)

    # The output contains raw scores (logits). We apply a softmax function
    # to convert these scores into probabilities.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 1 prediction
    top1_prob, top1_cat_id = torch.topk(probabilities, 1)
    
    # Look up the category name from the labels list
    category_name = labels[top1_cat_id.item()]
    confidence_score = top1_prob.item() * 100
    
    print("Inference complete.")
    return category_name, confidence_score


if __name__ == "__main__":
    try:
        # Execute the main steps
        model = get_model()
        labels = get_labels()
        image_tensor = process_image(IMAGE_PATH)
        category, confidence = predict(model, image_tensor, labels)

        # Print the final, formatted result
        print("\n--- AI Vision Result ---")
        print(f"Prediction: {category}, with {confidence:.2f}% confidence.")
        print("------------------------")

    except FileNotFoundError:
        print(f"\n[ERROR] The file '{IMAGE_PATH}' was not found.")
        print("Please make sure the image is in the same directory as the script.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("Please check your internet connection and that all libraries are installed correctly.")