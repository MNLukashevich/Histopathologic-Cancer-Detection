import pandas as pd
import shutil
import os

# Load the full training dataset (adjust the file path and format accordingly)
full_train_data = pd.read_csv('../../data-histopathologic-cancer-detection/train_labels.csv')

# Separate the dataset into two DataFrames based on the labels
label_0_data = full_train_data[full_train_data['label'] == 0]
label_1_data = full_train_data[full_train_data['label'] == 1]

# Define the number of samples you want for each label in the reduced dataset
samples_per_label = 5000  # Adjust this number as needed

# Sample a smaller subset of each label's data
reduced_label_0_data = label_0_data.sample(n=samples_per_label, random_state=42)
reduced_label_1_data = label_1_data.sample(n=samples_per_label, random_state=42)

# Concatenate the reduced datasets for both labels
reduced_train_data = pd.concat([reduced_label_0_data, reduced_label_1_data])

# Shuffle the combined dataset
reduced_train_data = reduced_train_data.sample(frac=1, random_state=42)

# Save the reduced dataset to a new CSV file
reduced_train_data.to_csv('../data/reduced_labels_10000.csv', index=False)


# Load the reduced training dataset (with IDs and labels)
reduced_train_data = pd.read_csv('../data/reduced_labels_10000.csv')

# Set the source folder containing all images
source_folder = '../../data-histopathologic-cancer-detection/train/'

# Set the destination folder where you want to save selected images
destination_folder = '../data/reduced_images_10000/'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Iterate through each row in the reduced dataset
for index, row in reduced_train_data.iterrows():
    image_id = row['id']
    label = row['label']
    
    # Define the source and destination paths for the image
    source_path = os.path.join(source_folder, f'{image_id}.tif')
    destination_path = os.path.join(destination_folder, f'{image_id}.tif')  # Keep the original filename
    
    # Copy the image from the source folder to the destination folder
    shutil.copyfile(source_path, destination_path)

print(f'Saved {len(reduced_train_data)} images to {destination_folder}')