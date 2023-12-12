import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
from model import load_model
from model import load_fashion_model
from model import get_random_image
from model import is_canvas_blank
from PIL import Image

final_model = load_model()
final_model_fashion = load_fashion_model()
st.set_page_config(layout="wide")
st.header("MNIST App")

dataset_option = st.selectbox(
    'Select Dataset',
    ('MNIST', 'Fashion MNIST'))

transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_fashion = datasets.FashionMNIST(root='./data', train = False, download = True, transform = transform)

option = st.selectbox(
    'Select Prediction Option',
    ('Generate Random Image', 'Draw your own'))

if dataset_option == 'MNIST':
    selected_dataset = mnist_trainset
    model = final_model
else:
    selected_dataset = mnist_fashion
    model = final_model_fashion

if option == 'Generate Random Image':
    if 'random_class' not in st.session_state:
        st.session_state.random_class = random.randint(0, 9)
    if st.button("Generate New Image", key = 'gen'):
        st.session_state.random_class = random.randint(0, 9)

    if 'random_class' in st.session_state:
        st.session_state.random_image = get_random_image(selected_dataset, st.session_state.random_class)
        
    # Display the chosen/generated image using st.image
    pil_image = Image.fromarray(st.session_state.random_image, mode='L')
    st.image(pil_image, caption=f'Class Label: {st.session_state.random_class}', width=150)

    if st.button("Predict"):
        # Convert the numpy array back to PyTorch tensor
        tensor_image = torch.tensor(st.session_state.random_image / 255.).unsqueeze(0).unsqueeze(0).float()
        y_pred = model(tensor_image)
        predictions = torch.argmax(y_pred, dim=1).numpy().tolist()
        if dataset_option == 'MNIST':
            st.write(f"Predicted Class: {predictions[0]}")
        else:
            clothing_items = ['T-shirt/top', 'Trouser', 'Pullover', 
                              'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            st.write(f"Predicted Class: {clothing_items[predictions[0]]}")


else:
    if dataset_option == 'MNIST':
        st.write("Draw an integer between 0 and 9")
    else:
        st.write("Draw a clothing item, either T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot")
    drawing_mode = 'freedraw'
    stroke_color = '#FFFFFF'
    bg_color = '#000000'
    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=16,
        stroke_color=stroke_color,
        background_color=bg_color,
        # background_image=Image.open(bg_image) if bg_image else None,
        background_image = None,
        update_streamlit=True,
        width=168,
        height=168,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert the canvas image to an OpenCV image (Grayscale format)
        opencv_image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        # resized_image = cv2.resize(opencv_image, (28, 28))
        resized_image = cv2.resize(opencv_image, (28, 28), interpolation=cv2.INTER_NEAREST)
        st.image(resized_image, channels='GRAY', width = 150)

    if st.button("Predict"):
        if is_canvas_blank(canvas_result.image_data):
            st.write("If you selected 'MNIST', please draw a number between 0-9 before clicking 'Predict'. If you selected Fashion MNIST, please draw a clothing item before clicking 'Predict'")
        else:
            feed_image = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float()
            feed_image = feed_image / 255
            # st.write(feed_image)

            y_pred = model(feed_image)
            # st.write(y_pred)
            predictions = torch.argmax(y_pred, dim=1).numpy().tolist()
            if dataset_option == 'MNIST':
                st.write(f"Predicted Class: {predictions[0]}")
            else:
                clothing_items = ['T-shirt/top', 'Trouser', 'Pullover', 
                              'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                st.write(f"Predicted Class: {clothing_items[predictions[0]]}")
                


   
    
