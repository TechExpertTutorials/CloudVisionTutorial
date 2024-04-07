"""
need to install google-cloud-vision (GCP SDK) from conda -c conda-forge  
conda create -n gcp-cloud  
conda install -c conda-forge pillow=10.1.0 pandas=2.1.2 google-cloud-vision=3.4.5 scikit-learn=1.3.2 ipykernel jupyterlab notebook python=3.12.0  
to set up env as new kernel in jupyterlabs:  
python -m ipykernel install --user --name=gcp-cloud

repo: https://github.com/donaldsrepo/gcp-solution
"""

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

@my_timer
def main():
    vision_client = vision.ImageAnnotatorClient()

    file = 'woman_bicycle.png'
    image_path = os.path.join('.\\content', file)

    with io.open(image_path, 'rb') as image_file:
        image_content = image_file.read()

    image = vision.Image(content=image_content)
    response = vision_client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations

    pillow_image = Image.open(image_path)
    df = pd.DataFrame(columns=['object', 'score'])
    df_list = []
    for obj in localized_object_annotations:
        df_list.append({'object': obj.name, 'score': obj.score}) 
        
        r, g, b = random.randint(150, 255), random.randint(
            150, 255), random.randint(150, 255)

        draw_borders(pillow_image, obj.bounding_poly, (r, g, b),
                    pillow_image.size, obj.name, obj.score)

    df = pd.DataFrame.from_records(df_list)
    print(df)
    pillow_image.show()


if __name__ == "__main__":
     main()
