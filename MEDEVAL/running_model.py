import os
import threading
import numpy as np
import smtplib, ssl
import pandas as pd
import foolbox as fb
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from vit_keras import vit
from flask import request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# BATCH_SIZE = 32 
IMG_SIZE = (224, 224)

def get_loader_data(data_loader):   # For getting images and labels from dataset_generator
    for images, labels in data_loader:
        images = images
        labels = labels
        break
    return images, labels

def generator(path_to_folder, batch_size):      # For getting data_generator
    dataset_datagen = ImageDataGenerator(rescale=1./255)
    dataset_generator = dataset_datagen.flow_from_directory(
        path_to_folder,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )
    print('Image shape:',dataset_generator.image_shape)
    print("Number of images in dataset_generator:", dataset_generator.n)
    return dataset_generator

def evaluate_dataset(model, dataset_path, accuracy):   # For evaluation on default dataset
    print('Defualt dataset')
    dataset_generator = generator(dataset_path, 32)
    dataset_loss, dataset_accuracy = model.evaluate(dataset_generator)
    print("Dataset Loss:", dataset_loss)
    print("Dataset Accuracy:", dataset_accuracy)
    accuracy.update({"dataset":[round(dataset_accuracy, 3), round(dataset_loss, 3)]})

def run_fgsm_attack(model, fgsm_attack_dataset, accuracy):    # For evaluation on fgsm attack
    print('FGSM Attack')
    dataset_generator = generator(fgsm_attack_dataset, 64)    # keep batch size less for faster since it chooses only first batch
    loaded_model = model

    # Check if the previous layer is a Dense layer with 1 unit
    if isinstance(loaded_model.layers[-1], Dense) and loaded_model.layers[-1].output_shape[-1] == 1:
        x = loaded_model.layers[-2].output
        new_output = Dense(2, activation='sigmoid', name='new_output')(x)
        modified_model = Model(inputs=loaded_model.input, outputs=new_output)
        modified_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        modified_model = loaded_model
    
    preprocessing = dict(flip_axis=-1, mean=[103.939, 116.779, 123.68])
    bounds = (0, 255)
    fmodel = fb.TensorFlowModel(modified_model, bounds=bounds, preprocessing=preprocessing)
    fmodel = fmodel.transform_bounds((0, 1))
    assert fmodel.bounds == (0, 1)
    
    images, labels = get_loader_data(dataset_generator)
    images_tf = tf.constant(images)
    labels_tf = tf.constant(labels)
    attack = fb.attacks.LinfDeepFoolAttack()
    # epsilons = np.linspace(0.03, 1, num=20)
    epsilon = 0.05

    raw, clipped, is_adv = attack(fmodel, images_tf, labels_tf, epsilons=epsilon)
    is_adv_float32 = tf.cast(is_adv, tf.float32)
    mean_adv = tf.reduce_mean(is_adv_float32, axis=-1)
    robust_accuracy = 1 - mean_adv
    print("FGSM Attack Dataset Accuracy:", robust_accuracy.numpy())

    accuracy.update({"fgsm":[round(robust_accuracy.numpy(), 3), 'NA (epsilon = 0.05)']})

def run_red_attack(model, red_attack_dataset,accuracy):     # For evaluation on RED attack
    print('RED Attack')
    red_attack_dataset_generator = generator(red_attack_dataset, 32)
    print(red_attack_dataset_generator.image_shape)
    print("Number of images in red_attack_dataset_generator:", red_attack_dataset_generator.n)

    # Evaluate on dataset
    red_attack_dataset_loss, red_attack_dataset_accuracy = model.evaluate(red_attack_dataset_generator)
    print("RED Attack Dataset Loss:", red_attack_dataset_loss)
    print("RED Attack Dataset Accuracy:", red_attack_dataset_accuracy)

    accuracy.update({"red":[round(red_attack_dataset_accuracy, 3), round(red_attack_dataset_loss, 3)]})

def sendmail(accuracy, profile):    # For sending email
    base_accuracy = accuracy['dataset'][0]
    change_in_accuracy = {key: base_accuracy - value[0] for key, value in accuracy.items() if key != 'dataset'}
    change_values = list(change_in_accuracy.values())

    mail_message = "\n".join([f"<tr><td><h3>{'Xray Pneumonia Dataset' if key=='dataset' else 'RED Attack' if key=='red' else 'FGSM Attack'}</h3></td><td>{value[0]}</td><td>{value[1]}</td></tr>" for key, value in accuracy.items()])
    mail_message = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Output</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                }}
                .container {{
                    width: 80%;
                    margin: 20px auto;
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #909090ba;
                    color: #fff;
                }}
                .augmentation-list {{
                    margin-top: 20px;
                }}
                .augmentation-list ol {{
                    list-style-type: decimal;
                    padding-left: 20px;
                }}
                .augmentation-list li {{
                    margin-bottom: 10px;
                }}
                p {{
                    text-align: center;
                }}

            </style>
        </head>
        <body>
            <div class="container">
                Hello {profile['name']},
                <h1>Model Report</h1>
                <p>Here is report of model you uploaded at MEDEVAL</p>
                <table>
                    <thead>
                        <tr>
                            <th>Key</th>
                            <th>Accuracy</th>
                            <th>Loss</th>
                        </tr>
                    </thead>
                    <tbody>
                        {mail_message}
                    </tbody>
                </table>
                There is {'decrease' if change_values[0]>0 else 'increase'} in accuracy of about {abs(change_values[0])*100}% after the attack.
                <h1>Augmentation</h1>
                <div class="augmentation-list">
                    <p>Here are some possible augmentations you can perform on X-ray images for pneumonia detection to increase the robustness and accuracy of your model:</p>
                    <ol>
                        <li><b>Rotation:</b> Rotate the images by a certain degree to simulate different angles of X-ray capture.</li>
                        <li><b>Horizontal and Vertical Shift:</b> Shift the images horizontally and vertically to simulate variations in positioning during X-ray capture.</li>
                        <li><b>Zoom:</b> Zoom in or out on the images to simulate different scales of X-ray capture.</li>
                        <li><b>Brightness Adjustment:</b> Adjust the brightness of the images to simulate variations in exposure during X-ray capture.</li>
                        <li><b>Contrast Adjustment:</b> Adjust the contrast of the images to simulate variations in contrast during X-ray capture.</li>
                        <li><b>Shear Transformation:</b> Apply shearing to the images to simulate distortion that may occur during X-ray capture.</li>
                        <li><b>Horizontal and Vertical Flip:</b> Flip the images horizontally or vertically to simulate different orientations of X-ray capture.</li>
                        <li><b>Noise Injection:</b> Add random noise to the images to simulate imperfections in X-ray capture.</li>
                        <li><b>Elastic Transformation:</b> Apply elastic deformation to the images to simulate stretching and deformation that may occur during X-ray capture.</li>
                        <li><b>Color Jitter:</b> Randomly change the color of the images to simulate variations in color balance during X-ray capture.</li>
                        <li><b>Cutout:</b> Randomly remove rectangular patches from the images to simulate occlusions or artifacts in X-ray capture.</li>
                        <li><b>Grid Distortion:</b> Apply grid distortion to the images to simulate grid-like distortions that may occur during X-ray capture.</li>
                        <li><b>Random Erasing:</b> Randomly erase patches from the images to simulate missing or obscured areas in X-ray capture.</li>
                        <li><b>Histogram Equalization:</b> Apply histogram equalization to enhance the contrast and details in the images.</li>
                    </ol>
                    <p>These augmentations can help increase the variability in your training data and make your model more robust to different conditions and variations in X-ray images. Experiment with different combinations of augmentations to find the ones that work best for your specific dataset and model architecture.</p>
                </div>
                Thanks,<br>MEDEVAL
            </div>
        </body>
        </html>
    """

    email_from = os.environ.get('institute_email')
    password = os.environ.get('institute_email_password')
    email_to = profile['email']
    
    date_str = pd.Timestamp.today().strftime('%Y-%m-%d')

    email_message = MIMEMultipart()
    email_message['From'] = email_from
    email_message['To'] = email_to
    email_message['Subject'] = f'Model Evaluation: {date_str}'

    email_message.attach(MIMEText(mail_message, "html"))

    email_string = email_message.as_string()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(email_from, password)
        server.sendmail(email_from, email_to, email_string)

    print('Mail Sent')

def evaluate_and_mail(model_path, attack_name, profile):
    # Load the model from the specified folder
    model = tf.keras.models.load_model(model_path)

    # Directory paths
    dataset = 'path_to_dataset'
    red_attack_dataset = 'path_to_dataset_having_red_attacked_images'

    accuracy = {}   # For storing accuracy of each attack performed

    threads = []    # Applied threading for evaluating model on different attacks parallelly
    t1 = threading.Thread(target=evaluate_dataset, args=(model, dataset, accuracy))       # Thread 1: Evaluate dataset
    threads.append(t1)
    
    if attack_name=='fgsm':
        t2 = threading.Thread(target=run_fgsm_attack, args=(model, dataset, accuracy))    # Thread 2: Run FGSM attack
        threads.append(t2)
    
    if attack_name=='red':
        t3 = threading.Thread(target=run_red_attack, args=(model, red_attack_dataset, accuracy))    # Thread 3: Run RED attack 
        threads.append(t3)

    for t in threads:   # Start all threads
        t.start()

    for t in threads:   # Wait for all threads to complete
        t.join()
    
    print(accuracy)
    sendmail(accuracy, profile)

    return 
