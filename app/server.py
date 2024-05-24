#server.py contents

from fastapi import FastAPI
import joblib
import numpy as np

# Load the trained model and define names of output classes
model = joblib.load('app/model.joblib')
class_names = np.array(['setosa', 'versicolor', 'virginica'])

# Initialize the app as a FastAPI class instance
app = FastAPI()

# The @app.get("/") decorator tells FastAPI that the function
# right below is in charge of handling requests that go to this path (’/’)
# which is the root page of our web-service.
@app.get('/')
# And the only thing this function is going to do is just print
# a welcome message for the user.
def read_root():
    return {'message': 'Iris model API'}

# And in order to interact with our model, we need a post method, which will
# receive the data, predict with the model and return the prediction result.
# So just like above `@app.predict("/predict/")` tells FastAPI that the
# function right below is in charge of handling requests that go to this
# path (’/predict/’).
# This means that we are creating another API path called /predict/ which
# should be used to pass the data to the model and this incoming data will be
# handled by the predict function
@app.post('/predict')
def predict(data: dict):
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {"features": [1, 2, 3, 4]}

    Returns:
        dict: A dictionary containing the predicted class.
    """
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}
