from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import json
import pickle
from sklearn.ensemble import RandomForestClassifier

configuration_filename = 'config.txt'

application = Flask(__name__)
application_api = Api(application)

class CarClassifier(Resource):
        def __init__(self, model, description):
                self.model = model
                self.description = description

        def check_data_correctness(self, data):
                #checking if the request data contains all categories required for prediction
                for catg in self.description['model']['category_order']:
                        if not catg in data:
                                return False
                #and number of data on each category is of same length
                data_values = list(data.values())
                values_len = len(data_values[0])
                for val in data_values:
                        if len(val) != values_len:
                                return False
                self.values_len = values_len
                return True

        def get_data_for_prediction(self, data):
                #the data is provided seperately for each category (seperated as key value) but the model requires it to be on same array
                #get the proper formatted array for ffeding to model
                #category_order defines the order as required by the model
                input_X = []
                catg_order = self.description['model']['category_order']
                for i in range(0, self.values_len):
                        row = []
                        for catg in catg_order:
                                row.append(data[catg][i])
                        input_X.append(row)
                return input_X
                
        def get(self):
                description_to_user = dict(self.description)
                #category_order may be unnecessary/unintended information to the user
                description_to_user['model'].pop('category_order')
                return jsonify({'description': description_to_user})
                
        def post(self):
                ##parse arguments of the request
                parser = reqparse.RequestParser()
                parser.add_argument('data')
                args = parser.parse_args()
                data = json.loads(args['data'])

                ##predict the result
                #try:
                if (not self.check_data_correctness(data)):
                        return {'error':'Wrong data provided'}
                input_X = self.get_data_for_prediction(data)
                output_Y = self.model.predict(input_X)
                return {'data': {'prediction': {'labels': self.description['model']['prediction'], 'values': output_Y.tolist()}}}
                #except:
                        #return {'error':'Error in server'}


if __name__=="__main__":
        with open(configuration_filename, 'r') as config_file:
                #config.txt file contains file name of where description and model are saved
                config = json.load(config_file)
        with open(config['description'], 'r') as description_file:
                description = json.load(description_file)
        with open(config['model'], 'rb') as model_file:
                model = pickle.load(model_file)
        
        application_api.add_resource(CarClassifier, config['resource_name'], resource_class_kwargs = {'model':model, 'description': description})
        application.run(debug = True)
