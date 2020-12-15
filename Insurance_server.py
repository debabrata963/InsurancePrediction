import pandas as pd
from tensorflow.keras import models
from flask import Flask,request
import joblib

model = models.load_model(r"C:\Users\dell\Documents\Digital AI\Algorithms\Insurance\insurance_model.h5")
pre = joblib.load(r"C:\Users\dell\Documents\Digital AI\Algorithms\Insurance\Insurance_model_transform.h5")

app=Flask(__name__)

@app.route('/',methods=["POST"])
def predict():
    data=request.get_json(force=True)
    data1 = pd.DataFrame(data, index = [0])
    data2 = pre.transform(data1)
    print(data1)
    predi=model.predict(data2)
    output=predi
    print(output[0])
    
    return str(output)

if __name__ == '__main__':
    app.run(port=8000,debug=True)
    