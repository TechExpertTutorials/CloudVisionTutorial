# Authentication to Google API
import os
import io
import math
from numpy import random
from collections import Counter
from google.cloud import vision
from Pillow_Library import draw_borders
from PIL import Image
import re
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='vision_key.json'
from my_timer import my_timer
import time

def main(path):
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    print("Landmarks:")

    for landmark in landmarks:
        print(landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print(f"Latitude {lat_lng.latitude}")
            print(f"Longitude {lat_lng.longitude}")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
 

if __name__ == "__main__":
     main(r'.\\content\eiffel.jpg')
     main(r'.\\content\moscow_small.jpeg')