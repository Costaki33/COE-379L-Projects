from flask import Flask, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('./models/altlenet5.keras')

@app.route('/model/info', methods=['GET'])
def model_info():
   return {
      "version": "v1",
      "name": "altlenet5",
      "description": "Classifies images containing damaged and undamaged buildings from Hurricane Harvey"
   }

def preprocess_input(im):
   """
   Converts user-provided input into an array that can be used with the model.
   This function could raise an exception.
   """
   # convert to a numpy array
   d = np.array(im)
   # then add an extra dimension
   return d.reshape(1, 128, 128, 3)

@app.route('/model/predict', methods=['POST'])
def classify_image():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   return { "result": model.predict(data).tolist()}


# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')
