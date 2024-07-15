"""
Simple script to test the API
"""
import requests
import os
import base64

def test_prediction_api():
    # Directory containing test images
    test_images_dir = "test_images"
    
    # Get the first image from the test_images folder
    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(test_images_dir, filename)
            with open(file_path, "rb") as image_file:
                # Convert image to base64
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Make the API request with a single image
                response = requests.post("http://localhost:8055/predict",
                                         json={"images": base64_image})
                
                print(response.json())
                break  # Only process the first image

if __name__ == "__main__":
    test_prediction_api()