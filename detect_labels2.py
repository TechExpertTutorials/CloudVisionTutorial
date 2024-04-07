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
    """Detects labels in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print("Labels:")

    for label in labels:
        print(label.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    


if __name__ == "__main__":
     main(r'.\\content\label1.jpeg')