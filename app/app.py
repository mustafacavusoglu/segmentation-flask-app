from flask import Flask,render_template,request
from prediction import Prediction

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")

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
    


@app.route("/predict",methods=["GET","POST"])
def predict():
    prediction = get_prediction()
    return render_template("predict.html",prediction=prediction)


if __name__ =="__main__":
    app.run(debug=True,port=1247)
