from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
with open("rating_prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Load individual label encoders
with open("type_encoder.pkl", "rb") as file:
    type_encoder = pickle.load(file)

with open("cuisine_encoder.pkl", "rb") as file:
    cuisine_encoder = pickle.load(file)

with open("location_encoder.pkl", "rb") as file:
    location_encoder = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def welcome():
    locations = [
        'Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
        'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
        'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
        'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
        'Koramangala 4th Block', 'Koramangala 5th Block',
        'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
        'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
        'Old Airport Road', 'Rajajinagar', 'Residency Road',
        'Sarjapur Road', 'Whitefield'
    ]
    restaurant_types = ['Casual Dining', 'Cafe', 'Quick Bites', 'Delivery', 'Mess', 'Dessert Parlor',
                        'Bakery', 'Pub', 'Takeaway', 'Fine Dining', 'Beverage Shop', 'Sweet Shop', 'Bar',
                        'Confectionery', 'Kiosk', 'Food Truck', 'Microbrewery', 'Lounge', 'Dhaba',
                        'Club', 'Food Court', 'Irani Cafe', 'Pop Up', 'Meat Shop']
    
    cuisines = [
        'North Indian', 'South Indian', 'Chinese', 'Italian', 'American',
        'Mexican', 'Japanese', 'Thai', 'Mediterranean', 'Lebanese'
    ]
    
    return render_template("index.html", locations=locations, rest_type=restaurant_types, cuisines=cuisines)

@app.route("/", methods=["POST"])
def predictor():
    if request.method == "POST":
        data = {
            'online_order': 1 if request.form.get("online_order") == "Yes" else 0,
            'book_table': 1 if request.form.get("book_table") == "Yes" else 0,
            'votes': int(request.form.get("votes", 0)),
            'avg_cost': float(request.form.get("avg_cost", 0)),
            'location': request.form.getlist("location"),
            'rest_type': request.form.getlist("rest_type"),
            'cuisines': request.form.getlist("cuisines")
        }

        # Encode location
        if data['location'] in location_encoder.classes_:
            data['location'] = location_encoder.transform([data['location']])[0]
        else:
            data['location'] = -1

        # Encode rest_type
        if data['rest_type'] in type_encoder.classes_:
            data['rest_type'] = type_encoder.transform([data['rest_type']])[0]
        else:
            data['rest_type'] = -1



        # encoded_rest_types = []
        # for rt in data['rest_type']:
        #     if rt in type_encoder.classes_:
        #         encoded_rest_types.append(type_encoder.transform([rt])[0])
        #     else:
        #         encoded_rest_types.append(-1)

        # Encode cuisines
        # encoded_cuisines = []
        # for cuisine in data['cuisines']:
        #     if cuisine in cuisine_encoder.classes_:
        #         encoded_cuisines.append(cuisine_encoder.transform([cuisine])[0])
        #     else:
        #         encoded_cuisines.append(-1)

        if data['cuisines'] in cuisine_encoder.classes_:
            data['cuisines'] = cuisine_encoder.transform([data['cuisines']])[0]
        else:
            data['cuisines'] = -1

     

        

        # Constructing the input features for the model
        input_features = [data['online_order'], data['book_table'], data['votes'], data['avg_cost'], data['location'],data['rest_type'],data['cuisines']] 

        # Convert to DataFrame
        df = pd.DataFrame([input_features], columns=['online_order', 'book_table', 'votes', 'avg_cost', 'location','rest_type','cuisines'])

        # Scaling numeric features
        numeric_features = ['online_order', 'book_table', 'votes', 'avg_cost', 'location','rest_type','cuisines']
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        print(input_features)
        print(df)
        # Making Predictions
        x_input = df.values
        print(x_input)
        rate = model.predict(x_input)[0]
        output = f"{round(rate, 1)} / 5"
        return render_template("index.html", prediction_text=output, locations=[
            'Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
            'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
            'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
            'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
            'Koramangala 4th Block', 'Koramangala 5th Block',
            'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
            'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
            'Old Airport Road', 'Rajajinagar', 'Residency Road',
            'Sarjapur Road', 'Whitefield'
        ], rest_type=[
            'Casual Dining', 'Cafe', 'Quick Bites', 'Delivery', 'Mess', 'Dessert Parlor',
            'Bakery', 'Pub', 'Takeaway', 'Fine Dining', 'Beverage Shop', 'Sweet Shop', 'Bar',
            'Confectionery', 'Kiosk', 'Food Truck', 'Microbrewery', 'Lounge', 'Dhaba',
            'Club', 'Food Court', 'Irani Cafe', 'Pop Up', 'Meat Shop'
        ], cuisines=[
            'North Indian', 'South Indian', 'Chinese', 'Italian', 'American',
            'Mexican', 'Japanese', 'Thai', 'Mediterranean', 'Lebanese'
        ])
     
    return render_template("index.html", prediction_text="output")

if __name__ == "__main__":
    app.run(debug=True)
