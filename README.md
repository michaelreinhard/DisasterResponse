# DisasterResponse

This project seeks to classify tweets surrounding natural disasters into one of 36 categories. It also includes a web app that allows the user to classify new tweets into the existing categories. 

## Directions 

Though there three files needed to run the app, the data preparation file, the model building file and the app running file itself. 

First, open a command terminal from inside the 'DisasterResponse' folder and run the data pipeline, `process_data.py`, with the code:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
Second, run the model, 'train_classifier.py', which will create the model and pickle it in a file called 'classifier.pkl'. In the command line from inside the 'DisasterResponse' folder run this code: 
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
Third, and finally, run the app from your command line. From inside the 'DisasterResponse' folder run the code:
```
python app/run.py
```
The directions will be returned to the command line but basically all you have to do is open a browser and point it to http://0.0.0.0:3001/ .

