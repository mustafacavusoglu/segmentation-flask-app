from flask import Flask,render_template,request
from prediction import Prediction

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    prediction = Prediction.get_prediction()
    return render_template("predict.html",prediction=prediction)


if __name__ =="__main__":
    app.run(debug=True,port=1247)