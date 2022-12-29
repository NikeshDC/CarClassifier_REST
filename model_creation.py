from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas
import pickle
import json

##file path for loading datset and saving the model afterwards
dataset_filepath = 'car.csv'
model_save_filepath = 'carClassifier.rf'

##prepare dataset
car_dataFrame_raw = pandas.read_csv(dataset_filepath)
X_labels = ['buying-price','maintainace-price','no-of-doors','person-capacity','size-of-luggage-boot','safety']
Y_label = 'evaluation'
##converting the string values of car dataset to categorical int values
car_dataFrame = pandas.DataFrame()
categories = {}
for label in X_labels:
    car_dataFrame[label], categories[label] = pandas.factorize(car_dataFrame_raw[label].values, sort = True)
    categories[label] = categories[label].tolist()
car_dataFrame[Y_label], categories[Y_label] = pandas.factorize(car_dataFrame_raw[Y_label], sort = True)
categories[Y_label] = categories[Y_label].tolist()
#saving the categorized file 
car_dataFrame.to_csv('car_categorical.csv')
with open('car_categories.txt', 'w') as categories_file:
    json.dump(categories, categories_file)


X = car_dataFrame[X_labels].values
Y = car_dataFrame[Y_label].values

##split the datset to evaluate performance on unknown dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

##create the model
randomForest = RandomForestClassifier(n_estimators = 25, max_features = 5)
randomForest.fit(X_train, Y_train)
##save the model
with open(model_save_filepath, 'wb') as model_file:
    pickle.dump(randomForest, model_file)

##evaluate accuracy
Y_predicted = randomForest.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, Y_predicted)
print("Accuracy:- ",accuracy)

