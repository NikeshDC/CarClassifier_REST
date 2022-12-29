import requests
import json

server_uri = 'http://127.0.0.1:5000/cars'


##        low	low	5more	4	big	low	unacc
##        low	low	5more	4	big	med	good
##        low	low	5more	4	big	high	vgood
##        low	low	5more	more	small	low	unacc
##"categories": {
##			"buying-price": ["high", "low", "med", "vhigh"], 
##			"maintainace-price": ["high", "low", "med", "vhigh"], 
##			"no-of-doors": ["2", "3", "4", "5more"], 
##			"person-capacity": ["2", "4", "more"], 
##			"size-of-luggage-boot": ["big", "med", "small"], 
##			"safety": ["high", "low", "med"], 
##			"evaluation": ["acc", "good", "unacc", "vgood"]
##			}

X_test = {
        "buying-price": [1,1,1,1],
        "maintainace-price":[1,1,1,1],
        "no-of-doors":[3,3,3,3],
        "person-capacity":[1,1,1,2],
        "size-of-luggage-boot":[0,0,0,1],
        "safety":[1,2,0,1],
}

data = {'data': json.dumps(X_test)}
print("Sending data")
y_predict = requests.post(server_uri, json=data, timeout = 15).json()

print(y_predict)
