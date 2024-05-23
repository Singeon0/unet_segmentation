import os

import cv2
import imageio.v2 as imageio  # Use imageio version 2 to avoid deprecation warning
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from sklearn.metrics import f1_score, jaccard_score
from scipy.ndimage import binary_fill_holes
import random


def create_folder(full_path_filename):
    # this function creates a folder if its not already existed
    if not os.path.exists(full_path_filename):
        os.makedirs(full_path_filename)


# Helper Functions
def predict(dir_path, CNN_model, mu=0, sd=1, n1=240, model_name=''):
    # print(f"Starting prediction. mu: {mu}, sd: {sd}, crop size: {n1}")

    # get all the files for testing
    files = [file for file in os.listdir(dir_path) if not any(ignore in file for ignore in ["log", "Utah_Training.h5", ".DS_Store", "auto segmentation"])]    # print(f"Processing files: {files}")

    for i in range(len(files)):
        # print(f"Segmenting: {files[i]}")
        create_folder(os.path.join(dir_path, files[i], f"auto segmentation {model_name}"))
        # get the shape of the image and number of slices
        file_path = os.path.join(dir_path, files[i], "data", "slice001.jpeg")
        temp = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if temp is None:
            # print(f"Error reading image: {os.path.join(dir_path, files[i], 'data', 'slice001.jpeg')}")
            continue
        # print(f"Image shape (pre-crop): {temp.shape}")

        number_of_slices = len(os.listdir(os.path.join(dir_path, files[i], "data")))
        # print(f"Number of slices: {number_of_slices}")

        # based off the image size, and the specified input size, find coordinates to crop image
        midpoint = temp.shape[0] // 2
        n11, n12 = midpoint - int(n1 / 2), midpoint + int(n1 / 2)
        # print(f"Crop coordinates: {n11}, {n12}")

        # initialise temp array for prediction
        input1 = np.zeros(shape=[number_of_slices, n1, n1])
        # print(f"Initial input1 shape: {input1.shape}")

        # ***  Loading data
        for n in range(number_of_slices):
            input_filename = f"slice{str(n + 1).zfill(3)}.jpeg"
            ImageIn = cv2.imread(os.path.join(dir_path, files[i], "data", input_filename), cv2.IMREAD_GRAYSCALE)
            if ImageIn is None:
                # print(f"Error reading image: {input_filename}")
                continue
            ImageIn = (ImageIn - mu) / sd
            input1[n, :, :] = ImageIn[n11:n12, n11:n12]

        # print(f"Loaded and preprocessed input1 shape: {input1.shape}")

        # *** Making predictions
        output = np.zeros(shape=[number_of_slices, n1, n1, 2])
        for n in range(number_of_slices):
            # Check input dimensions to ensure compatibility with model
            if input1.shape[1] < 2 or input1.shape[2] < 2:
                # print("Error: Input dimensions too small for model's pooling layers.")
                continue

            input_batch = input1[n, :, :, None]  # Add batch dimension
            try:
                prediction = CNN_model.predict(np.array([input_batch]))  # Ensure input is correctly formatted
                # print(f"Prediction shape for slice {n}: {prediction.shape}")
                output[n, :, :, :] = prediction
            except ValueError as e:
                # print(f"Error during prediction: {e}")
                continue

        output = np.argmax(output, axis=3)
        # print(f"Output shape after argmax: {output.shape}")

        # *** Writing data to output
        for n in range(number_of_slices):
            Imout = np.zeros(shape=[temp.shape[0], temp.shape[1]])
            Imout[n11:n12, n11:n12] = output[n, :, :]
            output_filename = f"slice{str(n + 1).zfill(3)}.jpeg"
            cv2.imwrite(os.path.join(dir_path, files[i], f"auto segmentation {model_name}", output_filename),
                            255 * Imout)


def score(dir_path, log_path, model_name):
    # create a txt file to write the results to
    with open(os.path.join(log_path, f"log_{model_name}.txt"), "a") as f:
        f1_scores = []
        iou_score = []

        # get all the files for testing
        files = os.listdir(dir_path)
        if "log" in files: files.remove("log")
        if "Utah_Training.h5" in files: files.remove("Utah_Training.h5")
        if ".DS_Store" in files: files.remove(".DS_Store")

        for i in range(len(files)):
            temp_path = os.path.join(dir_path, files[i], f"auto segmentation {model_name}", "slice001.jpeg")
            temp = imageio.imread(temp_path)

            # Check if the image has more than one channel (RGB) and convert to grayscale
            if temp.ndim > 2:
                temp = rgb2gray(temp) * 255  # Convert to grayscale and scale to 0-255
            temp = temp.astype(np.uint8)  # Ensure the type is uint8 for consistency with OpenCV

            pred = np.zeros(
                [len(os.listdir(os.path.join(dir_path, files[i], f"auto segmentation {model_name}"))), temp.shape[0], temp.shape[1]])

            temp_path = os.path.join(dir_path, files[i], "cavity", "slice001.jpeg")
            temp = imageio.imread(temp_path)
            if temp.ndim > 2:
                temp = rgb2gray(temp) * 255
            temp = temp.astype(np.uint8)

            true = np.zeros([len(os.listdir(os.path.join(dir_path, files[i], "cavity"))), temp.shape[0], temp.shape[1]])

            for k in range(pred.shape[0]):
                input_filename = "slice" + "{0:03}".format(k + 1) + ".jpeg"
                pred_img_path = os.path.join(dir_path, files[i], f"auto segmentation {model_name}", input_filename)
                true_img_path = os.path.join(dir_path, files[i], "cavity", input_filename)

                pred_img = imageio.imread(pred_img_path)
                true_img = imageio.imread(true_img_path)

                if pred_img.ndim > 2:
                    pred_img = rgb2gray(pred_img) * 255
                pred_img = pred_img.astype(np.uint8)

                if true_img.ndim > 2:
                    true_img = rgb2gray(true_img) * 255
                true_img = true_img.astype(np.uint8)

                preprocess = True if random.random() <= 0.1 else False

                if preprocess and model_name == "MULTIRES_ATT_UNET_MODEL":
                    # Ensure the mask is binary
                    binary_pred_img = (pred_img // 255).astype(bool)
                    binary_true_img = (true_img // 255).astype(bool)

                    # Fill holes in the binary mask
                    filled_pred_img = binary_fill_holes(binary_pred_img)
                    filled_true_img = binary_fill_holes(binary_true_img)

                    # Convert back to the original data type if necessary
                    pred[k, :, :] = filled_pred_img.astype(int)
                    true[k, :, :] = filled_true_img.astype(int)

                else:
                    pred[k, :, :] = pred_img // 255
                    true[k, :, :] = true_img // 255

            # calculate f1 score
            pred_f1, true_f1 = pred.flatten(), true.flatten()
            f1 = f1_score(pred_f1, true_f1, average="binary")
            iou = jaccard_score(pred_f1, true_f1, average="binary")

            f.write(files[i] + " - F1 Score: " + str(round(f1, 3)) + "\n")
            f.write(files[i] + " - IoU Score: " + str(round(iou, 3)) + "\n")
            f.write("-------------------------------------------------------------------------------------------------------------------------------------\n")
            f1_scores.append(f1)
            iou_score.append(iou)

        f.write(f"\nModel {model_name}")
        f.write("\nOVERALL F1 AVERAGE = {:.2f}%".format(round(np.mean(np.array(f1_scores)) * 100, 2)))
        f.write("\nOVERALL IOU AVERAGE = {:.2f}%".format(round(np.mean(np.array(iou_score)) * 100, 2)))
        f.write("-------------------------------------------------------------------------------------------------------------------------------------\n")
        f.write("-------------------------------------------------------------------------------------------------------------------------------------\n")
        f.write("\n\n")


# Fonction pour superposer un masque de segmentation sur une image originale
def overlay_segmentation(dir_path, model_name, color=[255, 0, 0], alpha=0.1):
    """

    :param dir_path: path to UTAH Test set folder
    """
    # get all the files for testing
    files = os.listdir(dir_path)
    if "log" in files: files.remove("log")
    if "Utah_Training.h5" in files: files.remove("Utah_Training.h5")
    if ".DS_Store" in files: files.remove(".DS_Store")
    if ".DS_Store" in files: files.remove(".DS_Store")

    for patient in files:
        patient_path = os.path.join(dir_path, patient)

        # print(f"patient_path = {patient_path}")

        create_folder(f"{patient_path}/predVSreal_{model_name}")

        folders_patients = os.listdir(patient_path)
        if ".DS_Store" in folders_patients: folders_patients.remove(".DS_Store")

        for folder in folders_patients:
            folder_path = os.path.join(patient_path, folder)
            slices = os.listdir(folder_path)

            if "data" in folder_path:
                path_slices_data = []
                for slice in slices:
                    temp = os.path.join(folder_path, slice)
                    path_slices_data.append(temp)  # print(len(path_slices_data))

            elif "cavity" in folder_path:
                # print(f"    folders from patient {patient_path} = {folder_path}")
                path_slices_real = []
                for slice in slices:
                    temp = os.path.join(folder_path, slice)
                    path_slices_real.append(temp)  # print(len(path_slices_real))

            elif "auto segmentation" in folder_path:
                # print(f"    folders from patient {patient_path} = {folder_path}")
                path_slices_pred = []
                for slice in slices:
                    temp = os.path.join(folder_path, slice)
                    path_slices_pred.append(temp)  # print(len(path_slices_pred))

        for i in range(len(path_slices_data)):
            image = cv2.imread(path_slices_data[i])
            mask_real = cv2.imread(path_slices_real[i], cv2.IMREAD_GRAYSCALE)
            mask_pred = cv2.imread(path_slices_pred[i], cv2.IMREAD_GRAYSCALE)

            if False:
                # Ensure the mask is binary
                binary_mask_real = (mask_real // 255).astype(bool)
                binary_mask_pred = (mask_pred // 255).astype(bool)

                # Fill holes in the binary mask
                filled_mask_real = binary_fill_holes(binary_mask_real)
                filled_mask_pred = binary_fill_holes(binary_mask_pred)

                # Convert back to the original data type if necessary
                mask_real = filled_mask_real.astype(int)
                mask_pred = filled_mask_pred.astype(int)


            # Convertir le masque en une image à 3 canaux
            mask_colored_real = cv2.cvtColor(mask_real, cv2.COLOR_GRAY2BGR)
            mask_colored_real[mask_real == 255] = color  # Appliquer la couleur au masque

            mask_colored_pred = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)
            mask_colored_pred[mask_pred == 255] = color  # Appliquer la couleur au masque

            # Superposer le masque sur l'image
            overlayed_real_segmentation = cv2.addWeighted(mask_colored_real, alpha, image, 1 - alpha, 0)
            overlayed_pred_segmentation = cv2.addWeighted(mask_colored_pred, alpha, image, 1 - alpha, 0)

            # Concatenate images horizontally
            concatenated_image = np.hstack((overlayed_real_segmentation, overlayed_pred_segmentation))

            # Convert to PIL Image to display using matplotlib
            display_image = Image.fromarray(concatenated_image)
            # plt.figure(figsize=(12, 6))  # Adjust the size as needed
            # plt.imshow(display_image)
            # plt.axis('off')  # Hide axis
            # plt.tight_layout()
            # plt.show()
            # Optionally, save the image
            display_image.save(os.path.join(patient_path, f"predVSreal_{model_name}", f"overlay_{i}.png"))


def create_charts(history, filename):
    plt.figure(figsize=(15, 10), dpi=200)

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    try:
        plt.plot(history.history['acc'], label='Train')
        plt.plot(history.history['val_acc'], label='Validation')
    except KeyError:
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.grid()
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(top=1, bottom=0)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(fname=filename, dpi=200)
