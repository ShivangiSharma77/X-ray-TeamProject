# ###  Import Essentials
import base64
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from keras.preprocessing import image
from flask_cors import CORS, cross_origin
import numpy as np
from keras.models import load_model

app= Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
### Load Model 
def get_model():
    global model
    model = load_model('covid.h5')
    print(" * model loaded!")
# ### Image preprocessing
def  process(test_image):
    if test_image.mode != "RGB":
        test_image = test_image.convert("RGB")
    test_image= test_image.resize((64,64))
    test_image=image.img_to_array(test_image)
    test_image = test_image[:, :, :3]
    test_image=np.expand_dims(test_image,axis=0)
    return test_image

print(" * loading keras model ...")
get_model()

# ###  Classifier
def classifier(result):
    if result[0][0] == 1.0:
        return("Non-covid")
    else:
        return("Covid")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    test_image = Image.open(io.BytesIO(decoded))
    processed_image = process(test_image)
    prediction = classifier(model.predict(processed_image))
    response = {
        'prediction': {
            'Covid_prediction': prediction
            
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
