from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

def ValuePredictor(to_predict_list, size):
    result = 0  # default value
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 13:
        loaded_model = joblib.load("Health_App/Heart_API/heart_disease.pickle")
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predict", methods=["POST"])
def predict():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    # diabetes
    if len(to_predict_list) == 13:
        result = ValuePredictor(to_predict_list, 13)
        if int(result) == 1:
            prediction = "Sorry your chances of getting the disease are high. Please consult the doctor immediately"
        else:
            prediction = "No need to fear. You have no dangerous symptoms of the disease"
        return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
