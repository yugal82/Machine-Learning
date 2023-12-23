import json
import pickle
import numpy as np

__locations = None
__data_colums = None
__model = None


def load_saved_artifacts():
    print('Loading saved artifacts')
    global __locations
    global __data_colums
    global __model

    with open('Projects\Project-1\server\columns.json', 'r') as f:
        __data_colums = json.load(f)['data_columns']
        __locations = __data_colums[3:]

    with open('Projects\Project-1\server\Bengaluru_house_price_predict.pickle', 'rb') as f:
        __model = pickle.load(f)

def get_location_names():
    return __locations

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_colums.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_colums))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    # print(get_estimated_price('Kalhalli', 1000, 2, 2))
    # print(get_estimated_price('Ejipura', 1000, 2, 2))