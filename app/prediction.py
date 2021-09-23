from tensorflow.keras.models import load_model
from flask.globals import request 
import cv2
from PIL import Image


class Prediction:
    
    def get_prediction():
        image = request.files.get("imagefile")
        image_np = cv2.imread(image,-1)
        image_np = image_np.reshape((1,512,512,3))
        model = load_model('satellitesegment.h5')
        result_img = model.predict(image_np)
        result_img = result_img[:,:,:,:]>0.5
        result_img = result_img[0,:,:,0]
        result_img = Image.fromarray(result_img)
        return result_img
