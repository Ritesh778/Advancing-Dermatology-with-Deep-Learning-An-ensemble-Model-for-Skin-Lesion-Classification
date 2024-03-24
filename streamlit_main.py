
##RITESH

import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.utils import custom_object_scope
import torch
import plotly.graph_objects as go  # Import Plotly

from torchvision import transforms

# Load the U-Net model with custom object scope
def load_segmentation_model():
    with tf.device('/CPU:0'):
        with custom_object_scope({'jaccard_distance': jaccard_distance, "iou": iou, "dice_coe": dice_coe, "precision": precision, "recall": recall}):
            model = tf.keras.models.load_model("/home/karthik/Desktop/project/unet_100_epoch.h5")
        return model

# Function to perform segmentation on an input image
def perform_segmentation(img_array, segmentation_model):
    with tf.device('/CPU:0'):
        resized_image = cv2.resize(img_array, (224, 224))
        input_image = np.expand_dims(resized_image, axis=0)
        prediction = segmentation_model.predict(input_image)
        # print(prediction.shape)
        return prediction

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Dermatofibroma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def majority_voting(output1, output2, output3):
    # Implement your majority voting logic here
    # For simplicity, let's assume a simple majority voting where the class with the most votes wins
    votes = [output1, output2, output3]
    majority_vote = max(set(votes), key=votes.count)
    return majority_vote

# Your custom metrics functions
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
    sum_ = np.sum(np.square(y_true), axis=-1) + np.sum(np.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

def iou(y_true, y_pred, smooth=100):
    intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
    sum_ = np.sum(np.square(y_true), axis=-1) + np.sum(np.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def dice_coe(y_true, y_pred, smooth=100):
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.finfo(np.float32).eps)
    return precision

def recall(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + np.finfo(np.float32).eps)
    return recall

# Load models
segmentation_model = load_segmentation_model()
# vgg16_model, resnet101_model, densenet_model = load_pytorch_models()

def theme_page():
    st.title('Skin Lesion Detection and Classification')
    st.write('Welcome to our application!')
    image = Image.open("./skin.jpg")
    resized_image = image
    st.image(resized_image, caption='Skin Lesion Image', use_column_width=True)
    st.write('Our application is designed to detect and classify different types of skin lesions using state-of-the-art deep learning models. By leveraging advanced segmentation and classification techniques, we aim to provide accurate and reliable results to aid in the early detection and diagnosis of various skin conditions.')
    st.write('The project aims to develop a robust skin lesion classification system capable of accurately identifying and classifying lesions into seven different diseases. Leveraging the power of both semantic segmentation and image classification, we propose an ensembling approach that combines the U-Net architecture for precise lesion segmentation with the VGG network for disease classification. The U-Net model will be utilized to perform pixel-level segmentation of skin lesions, providing detailed information about lesion boundaries. Subsequently, the segmented regions will be fed into VGG and Res-Net for classification in an ensemble can further enhance the model performance for disease classification, enabling the model to make predictions based on both localized features and overall lesion characteristics.')

# File upload page
def file_upload_page():
    st.title('Upload an Image')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform segmentation
        img_array = np.array(image)
        segmentation_result = perform_segmentation(img_array, segmentation_model)
        segmentation_result = np.squeeze(segmentation_result, axis=0)  # Remove the channel dimension
        st.image(segmentation_result, caption='Segmentation Result', use_column_width=True)

        # Convert image to PyTorch tensor and normalize
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.76304483, 0.54564637, 0.5700451], [0.14092779, 0.15261324, 0.16997057])
        ])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_resized = image.resize((224, 224))
        input_tensor = test_transform(img_resized).unsqueeze(0).to(device)
        # print(input_tensor)
        # Classify image using PyTorch models
        with torch.no_grad():
            vgg16_model = torch.jit.load("/home/karthik/Desktop/final year project/Vgg16.pt")
            vgg16_model.eval()
            vgg16_output = vgg16_model(input_tensor)

            resnet101_model = torch.jit.load("/home/karthik/Desktop/final year project/resnet.pt")
            resnet101_model.eval()
            resnet101_output = resnet101_model(input_tensor)

            densenet_model = torch.jit.load("/home/karthik/Desktop/final year project/densenet.pt")
            densenet_model.eval()
            densenet_output = densenet_model(input_tensor)

        class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

        probabilities_1 = torch.softmax(vgg16_output, dim=1)
        probabilities_2 = torch.softmax(resnet101_output, dim=1)
        probabilities_3 = torch.softmax(densenet_output, dim=1)

        epsilon = 1e-7  # A small epsilon value to prevent division by zero

        # Normalize probabilities for all models
        normalized_probs = []
        for probabilities in [probabilities_1, probabilities_2, probabilities_3]:
            min_prob, _ = torch.min(probabilities, dim=1, keepdim=True)
            max_prob, _ = torch.max(probabilities, dim=1, keepdim=True)
            normalized_probs.append((probabilities - min_prob) / (max_prob - min_prob + epsilon))

        # Scale normalized probabilities for better visualization (optional)
        scaled_probs = []
        for prob in normalized_probs:
            scaled_prob = prob.cpu().numpy() * 100  # Assuming you want to scale by 100
            scaled_probs.append(scaled_prob.tolist()[0])  # Convert to list and extract first element

        print("Normalized and Scaled Probabilities:")
        print(f"VGG16: {scaled_probs[0]}")
        print(f"ResNet101: {scaled_probs[1]}")
        print(f"DenseNet: {scaled_probs[2]}")

        vgg16_prediction = class_labels[torch.argmax(vgg16_output, 1).item()]
        resnet101_prediction = class_labels[torch.argmax(resnet101_output, 1).item()]
        densenet_prediction = class_labels[torch.argmax(densenet_output, 1).item()]

        vgg16_accuracy = torch.max(probabilities_1, 1)[0].item() * 100
        resnet101_accuracy = torch.max(probabilities_2, 1)[0].item() * 100
        densenet_accuracy = torch.max(probabilities_3, 1)[0].item() * 100

        st.write('VGG16 Prediction:', lesion_type_dict[vgg16_prediction], f'(Accuracy: {vgg16_accuracy:.2f}%)')
        st.write('ResNet101 Prediction:', lesion_type_dict[resnet101_prediction], f'(Accuracy: {resnet101_accuracy:.2f}%)')
        st.write('DenseNet Prediction:', lesion_type_dict[densenet_prediction], f'(Accuracy: {densenet_accuracy:.2f}%)')

        ensemble_prediction = majority_voting(vgg16_prediction, resnet101_prediction, densenet_prediction)
        ensemble_accuracy = (vgg16_accuracy + resnet101_accuracy + densenet_accuracy) / 3
        st.write('Ensemble Prediction:', lesion_type_dict[ensemble_prediction], f'(Accuracy: {ensemble_accuracy:.2f}%)')

        # Display individual line graphs for each model
        fig1 = go.Figure(data=[go.Scatter(x=class_labels, y=scaled_probs[0], mode='lines', name='VGG16')])
        fig1.update_layout(title='VGG16 Probabilities', xaxis_title='Class Labels', yaxis_title='Probability')
        st.plotly_chart(fig1)

        fig2 = go.Figure(data=[go.Scatter(x=class_labels, y=scaled_probs[1], mode='lines', name='ResNet101')])
        fig2.update_layout(title='ResNet101 Probabilities', xaxis_title='Class Labels', yaxis_title='Probability')
        st.plotly_chart(fig2)

        fig3 = go.Figure(data=[go.Scatter(x=class_labels, y=scaled_probs[2], mode='lines', name='DenseNet')])
        fig3.update_layout(title='DenseNet Probabilities', xaxis_title='Class Labels', yaxis_title='Probability')
        st.plotly_chart(fig3)

        # Display combined line graph with different colors
        combined_fig = go.Figure()
        combined_fig.add_trace(go.Scatter(x=class_labels, y=scaled_probs[0], mode='lines', name='VGG16', line=dict(color='blue')))
        combined_fig.add_trace(go.Scatter(x=class_labels, y=scaled_probs[1], mode='lines', name='ResNet101', line=dict(color='red')))
        combined_fig.add_trace(go.Scatter(x=class_labels, y=scaled_probs[2], mode='lines', name='DenseNet', line=dict(color='green')))
        combined_fig.update_layout(title='Combined Probabilities', xaxis_title='Class Labels', yaxis_title='Probability')
        st.plotly_chart(combined_fig)

# About us page
# About us page with images resized using interpolation for better clarity
def about_us_page():
    st.title('About Us')
    st.write('This application is developed by Ritesh and Karthik as part of our final year project.')

    # Ritesh's section
    col1, col2 = st.columns(2)
    with col1:
        ritesh_image_path = "./Ritesh.jpg"
        ritesh_image = Image.open(ritesh_image_path)
        # resized_ritesh_image = ImageOps.fit(ritesh_image, (100, 100), method=Image.LANCZOS)
        st.image(ritesh_image, caption='Ritesh', use_column_width=True)
    with col2:
        st.write('### Ritesh')
        st.write('GitHub: [https://github.com/Ritesh778](https://github.com/Ritesh778)')
        st.write('LinkedIn: [https://www.linkedin.com/in/ritesh-j-2b1331214/](https://www.linkedin.com/in/ritesh-j-2b1331214/)')

    # Karthik's section
    col3, col4 = st.columns(2)
    with col3:
        karthik_image_path = "./karthik.jpeg"
        karthik_image = Image.open(karthik_image_path)
        # resized_karthik_image = ImageOps.fit(karthik_image, (100, 100), method=Image.LANCZOS)
        st.image(karthik_image, caption='Karthik', use_column_width=True)
    with col4:
        st.write('### Karthik')
        st.write('GitHub: [https://github.com/karthik-username](https://github.com/karthik-username)')
        st.write('LinkedIn: [https://www.linkedin.com/in/karthik-profile](https://www.linkedin.com/in/karthik-profile)')

# App navigation
def main():
    pages = {
        "Theme": theme_page,
        "File Upload": file_upload_page,
        "About Us": about_us_page
    }

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == "__main__":
    main()