# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback

import boto3
import flask
import pandas as pd

from sklearn.impute import SimpleImputer

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

s3_bucket_name = "demo-predict-diabetes"
model_file = "model/diabetes-model.pkl"

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            s3client = boto3.client('s3')
            response = s3client.get_object(Bucket=s3_bucket_name, Key=model_file)
            cls.model = pickle.loads(response['Body'].read())
        return cls.model

    @classmethod
    def predict(cls, input):
        model = cls.get_model()
        return model.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        input = flask.request.data.decode("utf-8")
        s = io.StringIO(input)

        print("HTTP Request: ", s)
        data = pd.read_csv(s)

        print('Input Data: ', data)
        print("Invoked with {} records".format(data.shape[0]))

        # Some transformation, result is available in current data set
        X_predict = data
        
        # Avoiding zeros
        fill_0 = SimpleImputer(missing_values=0, strategy="mean") #, axis=0)
        X_predict = fill_0.fit_transform(X_predict)
  
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Predict with {} records".format(X_predict.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(X_predict)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")
