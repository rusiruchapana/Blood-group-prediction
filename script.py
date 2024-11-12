import os
import csv

# Path to the root folder containing the blood group folders (e.g., 'fingerprints/')
root_folder = 'dataset_blood_group/'

# Define the CSV file name
csv_file = 'dataset.csv'

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'blood_group'])  # CSV header

    # Iterate over each blood group folder
    for blood_group_folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, blood_group_folder)

        # Check if the path is a directory (to avoid any other files in root_folder)
        if os.path.isdir(folder_path):
            # Iterate over each image file in the blood group folder
            for image_file in os.listdir(folder_path):
                # Get the relative path to the image file
                image_path = os.path.join(blood_group_folder, image_file)
                # Write the image path and blood group label to the CSV
                writer.writerow([image_path, blood_group_folder])

print(f"CSV file '{csv_file}' created successfully!")
