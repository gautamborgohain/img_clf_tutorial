"""
Main `app`, containing routes and API specs
"""
import os
from flasgger import Swagger
from flask import Flask, jsonify, request

from core.model import CatsDogsModel
from core.util import logging

model_path = os.environ.get('MODEL_PATH', "core/cats_dogs_model_0257_v2.model")
app = Flask(__name__)
model = CatsDogsModel(model_path)

@app.route("/", methods=['GET'])
def check():
    response =  "Welcome to cats and dogs classifier go to \
    /apidocs for the documentation for this app"

    return jsonify(response), 200

@app.route("/score", methods=['GET'])
def score():
    """
    App to classify images to Cats and Dogs
    ---
    parameters:
    - name: url
      in: query
      type: string
      required: true
      description: URL to the image to be scored
      default: https://d2ph5fj80uercy.cloudfront.net/06/cat129.jpg
    responses:
      '200':
        description: successful operation
        schema:
          "$ref": "#/definitions/result"
        examples:
          cat_or_dog: cat
          confidence: 0.99
      '400':
        description: Service not found
    definitions:
      result:
        type: object
        properties:
          cat_or_dog:
            description: Is it a cat or a dog?
            type: string
          confidence:
            description: confidence level of the prediction
            type: number

    """
    url = request.args.get('url')
    result = model.score_image_from_url(url=url)
    logging.debug(f"url: {url}")
    logging.debug(f"result: {result}")
    return jsonify(result), 200


swagger = Swagger(app)

if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
