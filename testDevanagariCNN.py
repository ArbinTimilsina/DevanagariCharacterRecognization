from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import scipy.misc
from glob import glob
import cv2
from PIL import Image, ImageDraw, ImageFont

characterNames = [item.split("_")[-1] for item in sorted(glob("devanagari-character-set/Images/*"))]

def path_to_tensor(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def PredictCharacter(img_path, model):
    # obtain predicted vector
    predictedVector = model.predict(path_to_tensor(img_path))

    # return the character that is predicted by the model
    return characterNames[np.argmax(predictedVector)]

def DevanagariCharacterRecognizer(img_path, model):
    # Display the image
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("ArialBold.ttf", 50)
    draw.text((10, 40),"This is most likely a {}!".format(PredictCharacter(img_path, model)),(255,255,255),font=font)
    img.show()
 
myModel = load_model('DevanagariModel.hdf5')
myModel.load_weights('DevanagariModelBestWeights.hdf5')
DevanagariCharacterRecognizer("test.jpg", myModel)

