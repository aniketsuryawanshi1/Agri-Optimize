from django.shortcuts import render
from .models import Photo
from .CNN import CNN
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import os
import logging
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
logger = logging.getLogger(__name__)

# Load CSV files
def load_csv_files():
    current_directory = os.path.dirname(__file__)
    c_disease_info_path = os.path.join(current_directory, 'c_disease_info.csv')
    supplement_info_path = os.path.join(current_directory, 'supplement_info.csv')

    try:
        c_disease_info = pd.read_csv(c_disease_info_path, encoding='cp1252')
        supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')
        return c_disease_info, supplement_info
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: {e.filename} not found") from e

# Load CNN model
model = CNN(39)
model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_model_1_latest.pt')
try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("CNN model loaded successfully")
    else:
        raise FileNotFoundError(f"Error: CNN model file {model_path} not found")
except Exception as e:
    logger.exception("Error loading CNN model: %s", str(e))

# Load CSV files
c_disease_info, supplement_info = load_csv_files()

# Function to predict disease
def predict_disease(image):
    try:
        # Resize image to 224x224 pixels
        image = image.resize((224, 224))
        
        # Convert image to tensor
        input_data = TF.to_tensor(image)
        
        # Add batch dimension
        input_data = input_data.unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(input_data)
            output = output.detach().numpy()
            index = np.argmax(output)
            return index
    except Exception as e:
        logger.exception("Error in predict_disease function: %s", str(e))
        return None

@login_required(login_url="log_in")
def camera(request):

    if request.method == 'POST':

        if 'image_data' in request.FILES:
            try:
                image_data = request.FILES.get('image_data')
                photo = Photo.objects.create(image_data=image_data)
                image_path = photo.image_data.path

                # Open the image using PIL
                image = Image.open(image_path).convert("RGB")

                # Function to predict disease
                def predict_disease(image):
                    try:
                        # Resize image to 224x224 pixels
                        image = image.resize((224, 224))
                        
                        # Convert image to tensor
                        input_data = TF.to_tensor(image)
                        
                        # Add batch dimension
                        input_data = input_data.unsqueeze(0)

                        # Perform prediction
                        with torch.no_grad():
                            output = model(input_data)
                            output = output.detach().numpy()
                            index = np.argmax(output)
                            return index
                    except Exception as e:
                        print("Error in predict_disease:", e)
                        return None

                pred_index = predict_disease(image)
                if pred_index is not None and pred_index < 39:

                    title = c_disease_info.iloc[pred_index]['disease_name']
                    description = c_disease_info.iloc[pred_index]['description']
                    prevent = c_disease_info.iloc[pred_index]['Possible Steps']
                    image_url = c_disease_info.iloc[pred_index]['image_url']
                    supplement_name = supplement_info.iloc[pred_index]['supplement name']
                    supplement_image_url = supplement_info.iloc[pred_index]['supplement image']
                    supplement_buy_link = supplement_info.iloc[pred_index]['buy link']

                    # Check for NaN values and handle them
                    if pd.isna(supplement_name):
                        supplement_name = "Unknown"
                    if pd.isna(supplement_image_url):
                        supplement_image_url = ""
                    if pd.isna(supplement_buy_link):
                        supplement_buy_link = ""

                    predicted_result = {
                'title': title,
                'description': description,
                'prevent': prevent,
                'image_url': image_url,
                'supplement_name': supplement_name,
                'supplement_image_url': supplement_image_url,
                'supplement_buy_link': supplement_buy_link,
                    }

                    return JsonResponse(predicted_result)
                else:
                    return JsonResponse({'error': 'The model could not predict the image. Please try again.'}, status=400)
            except Exception as e:
                return JsonResponse({'error': 'An error occurred while processing the image. Please try again.'}, status=500)
        else:
            return JsonResponse({'error': 'Image data not found in request.'}, status=400)
    else:
        return render(request, 'camera.html')


