from fastapi import FastAPI
import joblib
import numpy as np

# This is an example on how to serve a model by API
model = joblib.load('app/model.joblib')
class_names = np.array(['setosa', 'versicolor', 'virginica'])

# Initialize the app as a FastAPI class instance
app = FastAPI()

# The @app.get("/") decorator tells FastAPI that the function
@app.get('/')

# a welcome message for the user.
def read_root():
    return {'message': 'Iris model API'}

# And in order to interact with our model, we need a post method, which will
# receive the data, predict with the model and return the prediction result.
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
