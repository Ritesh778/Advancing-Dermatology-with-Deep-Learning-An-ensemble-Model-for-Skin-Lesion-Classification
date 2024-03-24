import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, filters, MessageHandler, CallbackContext, CallbackQueryHandler
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import custom_object_scope
import asyncio
from datetime import datetime
import os
import requests
import logging
from telegram import Update, Bot
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import keras.backend as K
import keras.layers as kl
tf.config.set_visible_devices([], 'GPU')
import torch
from torchvision import models, transforms

#Telegram bot token
TOKEN = '6559447701:AAF7CcOzTwbEv9QGB5ST22x37t891OKa77M'

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Dermatofibroma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

## used in test_transform 
norm_mean = [0.76304483, 0.54564637, 0.5700451]
norm_std = [0.14092779, 0.15261324, 0.16997057]


input_size = 224
test_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# Your custom metrics functions
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def dice_coe(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Load the U-Net model with custom object scope
def load_segmentation_model():
    with tf.device('/CPU:0'):
        with custom_object_scope({'jaccard_distance': jaccard_distance, "iou": iou, "dice_coe": dice_coe, "precision": precision, "recall": recall}):
            model = tf.keras.models.load_model("/home/karthik/Desktop/project/unet_100_epoch.h5")
        return model

# Function to perform segmentation on an input image
def perform_segmentation(image_path, segmentation_model):
    with tf.device('/CPU:0'):
        img = cv2.imread(image_path)
        resized_image = cv2.resize(img, (224, 224))
        input_image = np.expand_dims(resized_image, axis=0)
        prediction = segmentation_model.predict(input_image)
        return prediction

# Function to save segmented image
def save_segmented_image(segmentation_result, output_path):
    plt.imsave(output_path, segmentation_result.reshape(224, 224), cmap='binary_r')



# Your existing process_image function
async def process_image_vgg16(image_data):
    try:
        # Load the PyTorch model
        model_path = "./Vgg16.pt"
        model = torch.jit.load(model_path)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load your input data asynchronously using PILAsync
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, lambda: Image.open(image_data))

        # Resize the image asynchronously
        img_resized = await loop.run_in_executor(None, lambda: img.resize((224, 224)))

        # Convert the image to PyTorch tensor and normalize
        image_tensor = test_transform(img_resized).unsqueeze(0).to(device)

        # Predict asynchronously using PyTorch model
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, 1).item()

        class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
        predicted_label = class_labels[predicted_class]

        # Now, you can use the predicted_label in your further processing

        return predicted_label
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error processing image"
    
async def process_image_resnet101(image_data):
    try:
        # Load the PyTorch model
        model_path = "./resnet.pt"
        model = torch.jit.load(model_path)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load your input data asynchronously using PILAsync
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, lambda: Image.open(image_data))

        # Resize the image asynchronously
        img_resized = await loop.run_in_executor(None, lambda: img.resize((224, 224)))

        # Convert the image to PyTorch tensor and normalize
        image_tensor = test_transform(img_resized).unsqueeze(0).to(device)

        # Predict asynchronously using PyTorch model
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, 1).item()

        class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
        predicted_label = class_labels[predicted_class]

        # Now, you can use the predicted_label in your further processing

        return predicted_label
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error processing image"
async def process_image_densenet(image_data):
    try:
        # Load the PyTorch model
        model_path = "/home/karthik/Desktop/project/densenet.pt"
        model = torch.jit.load(model_path)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load your input data asynchronously using PILAsync
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, lambda: Image.open(image_data))

        # Resize the image asynchronously
        img_resized = await loop.run_in_executor(None, lambda: img.resize((224, 224)))

        # Convert the image to PyTorch tensor and normalize
        image_tensor = test_transform(img_resized).unsqueeze(0).to(device)

        # Predict asynchronously using PyTorch model
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, 1).item()

        class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
        predicted_label = class_labels[predicted_class]

        # Now, you can use the predicted_label in your further processing

        return predicted_label
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error processing image"

# Other parts of your code remain unchanged...
def download_telegram_file( file_id, directory="./photos", filename_prefix=""):
  """Downloads a Telegram file from its ID and saves it to a dynamic path.

  Args:
    api_key (str): Your Telegram bot API key.
    file_id (str): The ID of the Telegram file.
    directory (str, optional): The directory to save the file. Defaults to current directory.
    filename_prefix (str, optional): A prefix to add to the filename. Defaults to "".

  Returns:
    str: The downloaded file path on success, None on failure.
  """

  url = f"https://api.telegram.org/bot{TOKEN}/getFile"
  payload = {"file_id": file_id}
  headers = {
      "accept": "application/json",
      "User-Agent": "My Telegram Bot",
      "content-type": "application/json"
  }

  try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for non-200 status codes

    data = response.json()
    if data["ok"]:
      download_url = f"https://api.telegram.org/file/bot{TOKEN}/{data['result']['file_path']}"
      response = requests.get(download_url)
      response.raise_for_status()

      # Generate unique filename based on file details and timestamp
      file_extension = os.path.splitext(data['result']['file_path'])[1]
      timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
      filename = f"{filename_prefix}_{timestamp}{file_extension}"

      # Ensure directory exists, create if needed
      os.makedirs(directory, exist_ok=True)
      full_path = os.path.join(directory, filename)

      with open(full_path, 'wb') as f:
        f.write(response.content)
      print(f"File downloaded successfully: {full_path}")
      return full_path
    else:
      print(f"Error retrieving file info: {data}")
      return None

  except requests.exceptions.RequestException as e:
    print(f"Download failed: {e}")
    return None
def apply_mask_and_enhance(input_image, mask):
    # Resize the mask to match the input image if necessary
    input_image=cv2.imread(input_image)
    mask=cv2.imread(mask,cv2.IMREAD_GRAYSCALE)

    if input_image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (input_image.shape[1], input_image.shape[0]))

    # Ensure the mask is binary (black and white)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    mask= mask
    # Convert the mask to a 3-channel image to match the input image
    mask_3_channel = cv2.merge([mask] * 3)

    # Combine the input image and the mask
    result = cv2.bitwise_and(input_image, mask_3_channel)

    return result


# Your existing handle_photo function
async def handle_photo(update: Update, context: CallbackContext, segmentation_model):
    photo = update.message.photo[-1]  # Get the highest-resolution photo
    file_id = photo.file_id
    print(photo)
    logger.info(f"Received photo with file_id: {file_id}")

    try:
        # Download the photo using the `telegram.File` object
        download_path = download_telegram_file(file_id)

        # Perform segmentation
        segmentation_result = perform_segmentation(download_path, segmentation_model)
        # Save the segmented image

        segmented_image_path = "./segmented_output.png"
        save_segmented_image(segmentation_result, segmented_image_path)
        # Apply mask and enhance

        enhanced_image = apply_mask_and_enhance(download_path, segmented_image_path)

        # Save the enhanced image
        enhanced_image_path = "./enhanced_output.png"
        cv2.imwrite(enhanced_image_path, enhanced_image)


        await context.bot.send_message(chat_id=update.effective_chat.id, text="Hang on ................")

        # Send the segmented image back to the user
        await context.bot.send_photo(
            chat_id=update.effective_chat.id, photo=open(enhanced_image_path, 'rb'), caption = "Check out the Segmentation Result! 🕵️‍♂️ Keep an eye on the highlighted areas!"
        )
        output1 = await process_image_vgg16(download_path)
        output2 = await process_image_resnet101(download_path)
        output3 = await process_image_densenet(download_path)
        # output4 = await process_image_vgg16(enhanced_image_path)
        # output5 = await process_image_resnet101(enhanced_image_path)
        # output6 = await process_image_densenet(enhanced_image_path)
        ensemble_result = ensemble_report(output1, output2, output3, lesion_type_dict)
        if not any("error" in result for result in [output1, output2, output3]) and output1 and output1 and output3:
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=ensemble_result
            )

        os.remove(download_path)
        os.remove(segmented_image_path)
        os.remove(enhanced_image_path)
    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text="An error occurred while processing your image."
        )

async def start(update: Update, context: CallbackContext):
    greeting = "Hello Mate , First time edo try chesanu pic pampu."
    await update.message.reply_text(greeting)


async def handle_error(update: Update, context: CallbackContext):
    """Logs errors."""
    logger.error(f"Update {update} caused error: {context.error}")

def ensemble_report(output1, output2, output3,lesion_type_dict):
    # Assuming output1, output2, and output3 are the predicted class labels

    # Get the ensemble prediction based on majority voting
    ensemble_prediction = majority_voting(output1, output2, output3)

    # ensemble_prediction_enhanced = majority_voting(output6, output5, output6)
    # Create the ensemble report
    report = f"Ensemble Report:\n"
    report += f"RESNET101: {lesion_type_dict.get(output2, 'Unknown')}\n"
    report += f"Densenet: {lesion_type_dict.get(output3, 'Unknown')}\n"
    report += f"Vgg16: {lesion_type_dict.get(output1, 'Unknown')}\n"
    report += f"Ensemble Prediction: {lesion_type_dict.get(ensemble_prediction, 'Unknown')} based on majority voting."


    # report +=f"Ensemble Report for enhaned:\n"
    # report += f"RESNET101: {lesion_type_dict.get(output5, 'Unknown')}\n"
    # report += f"Densenet: {lesion_type_dict.get(output6, 'Unknown')}\n"
    # report += f"Vgg16: {lesion_type_dict.get(output4, 'Unknown')}\n"
    # report += f"Ensemble Prediction: {lesion_type_dict.get(ensemble_prediction_enhanced, 'Unknown')} based on majority voting."
    return report

def majority_voting(output1, output2, output3):
    # Implement your majority voting logic here
    # For simplicity, let's assume a simple majority voting where the class with the most votes wins
    votes = [output1, output2, output3]
    majority_vote = max(set(votes), key=votes.count)
    return majority_vote



def main():
    # Load the segmentation model
    segmentation_model = load_segmentation_model()
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, lambda update, context: handle_photo(update, context, segmentation_model)))
    application.add_handler(CallbackQueryHandler(handle_error))
    application.run_polling()
    
    
    # ...

    # Handle photo messages
    application.add_handler(MessageHandler(filters.PHOTO, lambda update, context: handle_photo(update, context, segmentation_model)))

    # ...

if __name__ == '__main__':
    main()
