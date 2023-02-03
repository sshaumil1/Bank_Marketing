from flask import Flask, render_template,jsonify,request
import numpy as np
from Project.utils import TermDeposit
import config

app = Flask(__name__)
@app.route("/")
def get_checked():
    return render_template

@app.route("/TermDeposit",methods =["GET"])
def get_predicted_class():
    data = request.form
    age       = eval(data["age"])
    education = data["education"]
    default   = data["default"]
    balance   = eval(data["balance"])
    housing   = data["housing"]
    loan      = data["loan"]
    duration  = eval(data["duration"])
    campaign  = eval(data["campaign"])
    pdays     = eval(data["pdays"])
    previous  = eval(data["previous"])
    job       = data["job"]

    output_dict = {0: "NO", 1: "Yes"}
    
    obj = TermDeposit(age,education,default,balance,housing,loan,duration,campaign,pdays,previous,job)
    prediction = obj.get_prediction()
    return jsonify({"Result": f"If the customer will subscribe bank term deposit:: {output_dict[prediction[0]]}"})

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = config.PORT_NUMBER)