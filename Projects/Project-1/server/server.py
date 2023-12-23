from flask import Flask, request, jsonify
import utils

app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'Hello World!'


@app.route('/get-location-names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': utils.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/esitmate-price', methods=['POST'])
def get_esitmate_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': utils.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Allow-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    print('Starting Python Flask Server')
    utils.load_saved_artifacts()
    app.run()