from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .auth import authentication
from .forms import UploadImageForm
from .C_CNN import CNN
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
from .fertilizer import fertilizer_dic
from . import config
import requests
from django.utils.safestring import mark_safe
import logging
from django.http import JsonResponse
# Create your views here.
logger = logging.getLogger(__name__)
# Load CSV files
def load_csv_files():
    current_directory = os.path.dirname(__file__)
    disease_info_path = os.path.join(current_directory, 'disease_info.csv')
    supplement_info_path = os.path.join(current_directory, 'supplement_info.csv')

    

    try:
       
        disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
        supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')

        return disease_info, supplement_info
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
    print(e)


# Loading crop recommendation model

crop_recommendation_model_path =  os.path.join(os.path.dirname(__file__), 'RandomForest.pkl')
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


def home(request):
    return render(request,'home.html')

def register(request):
    if request.method == "POST":
        name = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        repassword = request.POST['repassword']
        
        # Verify password and register user
        verify = authentication(name, password, repassword)
        if verify == "success":
            # Check if the user with the given email already exists
            if User.objects.filter(email=email).exists():
                messages.error(request, "Email is already in use.")
            else:
                # Create a new user
                user = User.objects.create_user(username=email, email=email, password=password)
                user.first_name = name
                user.save()
                
                # Display success message
                messages.success(request, "Your Account has been Created.")
                print("User has been created")
                # Redirect to login page after successful registration
                return redirect("log_in")
        else:
            # Display error message
            messages.error(request, verify)
    
    # Render the registration form template
    return render(request, "register.html")

def log_in(request):
    if request.method == "POST":
        username = request.POST['username']  # Assuming email is used as username
        password = request.POST['password']

        # Authenticate user
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            print("User logged in successfully")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    
    # Render the login form template
    return render(request, "log_in.html")

@login_required(login_url="log_in")
def dashboard(request):
    user_email = request.user.email
    username = request.user.first_name

    context = {
        'user_email': user_email,
        'username': username
        }
    
    return render(request, "dashboard.html", context)

@login_required(login_url="log_in")
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfully...!")
    return redirect("/")



def predict_disease(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Load CSV files
disease_info, supplement_info = load_csv_files()
@login_required(login_url="log_in")
def accept(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST,request.FILES)
        if form.is_valid():
            image_instance =  form.save()
            image_path = image_instance.image.path
            try:
                pred_index = predict_disease(image_path)
            except:
                error_message = "Model could not predict the image. Please try again."
                return render(request, 'error.html', {'error_message': error_message})

            # Add handling for unknown or untrained images
            if pred_index is None or pred_index >= 39:
                error_message = "The model is not trained for the given image."
                return render(request, 'error.html', {'error_message': error_message})            
            if disease_info is not None and supplement_info is not None:
                title = disease_info.iloc[pred_index]['disease_name']
                description = disease_info.iloc[pred_index]['description']
                prevent = disease_info.iloc[pred_index]['Possible Steps']
                image_url = disease_info.iloc[pred_index]['image_url']
                supplement_name = supplement_info.iloc[pred_index]['supplement name']
                supplement_image_url = supplement_info.iloc[pred_index]['supplement image']
                supplement_buy_link = supplement_info.iloc[pred_index]['buy link']
            else:
                title = description = prevent = image_url = supplement_name = supplement_image_url = supplement_buy_link = "Data not available"

            context = {
                'title': title,
                'description': description,
                'prevent': prevent,
                'image_url': image_url,
                'supplement_name': supplement_name,
                'supplement_image_url': supplement_image_url,
                'supplement_buy_link': supplement_buy_link,
            }

            return render(request, 'predict.html', context)
    else:
        form = UploadImageForm()
    return render(request, 'accept.html', {'form': form})


#****************************************************Fertilization and Soil Module ***********************************************************

def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "q=" + city_name + "&appid=" + api_key
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] != "404":
        main_data = data["main"]
        temperature = main_data["temp"]
        humidity = main_data["humidity"]
        return temperature, humidity
    else:
        return None


@login_required(login_url="log_in")
def crop(request):
    title = 'AgriOptimize - Crop Recommendation'
    if request.method == 'POST':
        # Extract data from the POST request
        N = int(request.POST.get('nitrogen'))
        P = int(request.POST.get('phosphorous'))
        K = int(request.POST.get('pottasium'))
        ph = float(request.POST.get('ph'))
        rainfall = float(request.POST.get('rainfall'))
        city = request.POST.get("city")

        # Call weather_fetch function with the city parameter
        print(city)
        weather_data = weather_fetch(city)
        print(weather_data)
        print(city)

        if weather_data is not None:
            temperature, humidity = weather_data
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            # Render the template with the prediction result
            return render(request, 'crop_r.html', {'final_prediction': final_prediction,'title': title })
        else:
            # Render the template with an error message if weather data is not available
            error_message = "Weather data not available for the selected city. Please try again."
            return render(request, 'error.html', {'error_message': error_message})
    else:
        # Render the template for the initial form view
        return render(request, 'crop.html')
    
@login_required(login_url="log_in")
def fertilizer(request):
    title = 'AgriOptimize - Fertilizer Suggestion'

    if request.method == 'POST':
        crop_name = request.POST.get('cropname')
        nitrogen_str = request.POST.get('nitrogen')
        phosphorous_str = request.POST.get('phosphorous')
        potassium_str = request.POST.get('pottasium')
        print(crop_name,nitrogen_str,phosphorous_str,potassium_str)
        
        # Check if any of the input values are empty or None
        if not all([crop_name, nitrogen_str, phosphorous_str, potassium_str]):
            # Handle the case where inputs are missing
            error_message = "Please fill in all the fields."
            return render(request, 'fertilizer.html', {'title': title, 'error_message': error_message})
        
        # Convert input values to integers
        try:
            N = int(nitrogen_str)
            P = int(phosphorous_str)
            K = int(potassium_str)
        except ValueError:
            # Handle the case where input values are not valid integers
            error_message = "Invalid input values. Please enter valid integers."
            return render(request, 'fertilizer.html', {'title': title, 'error_message': error_message})
        
        # Load fertilizer data
        fertilizer_path = os.path.join(os.path.dirname(__file__), 'fertilizer.csv')  # Corrected variable name
        df = pd.read_csv(fertilizer_path)
        
        # Calculate fertilizer recommendation
        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "Nlow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "Plow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "Klow"

        recommendation = mark_safe(str(fertilizer_dic[key]))
        print(recommendation)
        return render(request, 'fertilizer_r.html', {'recommendation': recommendation, 'title': title})

    else:
        return render(request, 'fertilizer.html', {'title': title})
    

