from flask import Flask, request, jsonify
import json
import os
from flask_cors import CORS
from Five_Personality_Clusters import Predictor
import logging

predictor = Predictor()

class MyFlaskApp(Flask):

    def run(self, host='0.0.0.0', port=None, debug=True, load_dotenv=True, **options):
        if not self.debug or os.getenv("WERKZEUG_RUN_MAIN") == "true":
            with self.app_context():
                predictor.prepPredictor()
        super(MyFlaskApp, self).run(
            host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options
        )


app = MyFlaskApp(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        value = json.loads(request.data.decode("UTF-8"))
        input = [value['text']]
        logging.info(input)
        result = predictor.predict(input)
        response = jsonify({'output': str(result)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        #return jsonify(prediction=result)
        return response
    elif request.method == "GET":
        logging.info('get req received')
        return "testing :)"
    else:
        logging.info('unknown req')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run()

