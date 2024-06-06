from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

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

scaler = load('scaler.joblib') 
app = Flask(__name__)

@app.route("/")
def welcome():
    city = [
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
    
    cuisines = ['African, Burger', 'African, Burger, Desserts, Beverages, Fast Food', 'American', 'American, Asian, Continental, North Indian, South Indian, Chinese', 'American, Asian, European, North Indian', 'American, BBQ', 'American, Bakery, Beverages, Cafe, Healthy Food, Juices, North Indian, Sandwich', 'American, Burger, Fast Food', 'American, Burger, Italian, Steak', 'American, Burger, Momos, Bengali', 'American, Cafe, Chinese, Italian, Desserts', 'American, Cafe, Continental', 'American, Cafe, Continental, French, Burger, Mexican, Desserts, Pizza', 'American, Chinese', 'American, Chinese, Continental, North Indian', 'American, Continental', 'American, Continental, BBQ, Steak', 'American, Continental, Chinese', 'American, Continental, Fast Food, Steak', 'American, Continental, Finger Food', 'American, Continental, North Indian, Mediterranean', 'American, Continental, North Indian, Salad', 'American, Continental, North Indian, South Indian', 'American, Continental, Pizza', 'American, Continental, Salad', 'American, Continental, Salad, Italian', 'American, Continental, Salad, Italian, Asian', 'American, Continental, Steak', 'American, Continental, Steak, Salad', 'American, European', 'American, European, Healthy Food', 'American, Fast Food', 'American, Finger Food', 'American, Finger Food, BBQ, Steak', 'American, Finger Food, Italian, Mexican', 'American, Goan', 'American, Italian', 'American, Italian, Bakery, Desserts, Sandwich, Beverages, Salad, Rolls', 'American, Mexican', 'American, Mexican, BBQ', 'American, Mexican, Italian, Steak', 'American, Modern Indian, Italian, South Indian', 'American, North Indian, Chinese', 'American, North Indian, Chinese, Finger Food', 'American, North Indian, Chinese, Finger Food, Momos', 'American, North Indian, European, Tex-Mex', 'American, North Indian, Pizza, Finger Food, Continental, Italian', 'American, North Indian, Salad', 'American, Pizza', 'American, Pizza, Burger', 'American, Sandwich', 'American, South Indian, Thai, Pizza, Italian', 'American, Tex-Mex, Burger, BBQ', 'American, Tex-Mex, Burger, BBQ, Mexican', 'American, Thai, Healthy Food', 'Andhra', 'Andhra, Beverages, Biryani, Chinese, Fast Food, Hyderabadi, North Indian, South Indian', 'Andhra, Biryani', 'Andhra, Biryani, Beverages, Kebab', 'Andhra, Biryani, Chinese', 'Andhra, Biryani, Chinese, Hyderabadi, North Indian, South Indian', 'Andhra, Biryani, Chinese, North Indian', 'Andhra, Biryani, Chinese, North Indian, Seafood, South Indian', 'Andhra, Biryani, Chinese, South Indian', 'Andhra, Biryani, Mughlai', 'Andhra, Biryani, North Indian', 'Andhra, Biryani, North Indian, Beverages', 'Andhra, Biryani, North Indian, Chinese', 'Andhra, Biryani, North Indian, South Indian', 'Andhra, Biryani, Seafood', 'Andhra, Biryani, Seafood, North Indian', 'Andhra, Biryani, South Indian', 'Andhra, Biryani, South Indian, Chinese', 'Andhra, Chettinad', 'Andhra, Chinese', 'Andhra, Chinese, Biryani', 'Andhra, Chinese, Continental', 'Andhra, Chinese, North Indian', 'Andhra, Chinese, North Indian, Biryani', 'Andhra, Chinese, North Indian, Biryani, Beverages', 'Andhra, Chinese, North Indian, Hyderabadi', 'Andhra, Chinese, North Indian, Seafood', 'Andhra, Chinese, North Indian, South Indian', 'Andhra, Chinese, South Indian', 'Andhra, Chinese, South Indian, North Indian', 'Andhra, Continental, Chinese', 'Andhra, Fast Food', 'Andhra, Hyderabadi', 'Andhra, Hyderabadi, Biryani', 'Andhra, Hyderabadi, Biryani, Chinese, North Indian', 'Andhra, Hyderabadi, Chinese', 'Andhra, Hyderabadi, North Indian', 'Andhra, Kerala, South Indian', 'Andhra, North Indian', 'Andhra, North Indian, Biryani', 'Andhra, North Indian, Biryani, Beverages', 'Andhra, North Indian, Biryani, Seafood', 'Andhra, North Indian, Chinese', 'Andhra, North Indian, Chinese, Biryani', 'Andhra, North Indian, Chinese, Biryani, Seafood', 'Andhra, North Indian, Chinese, Mangalorean', 'Andhra, North Indian, Chinese, Mangalorean, Seafood, Biryani', 'Andhra, North Indian, Chinese, Seafood', 'Andhra, North Indian, Chinese, Seafood, Biryani', 'Andhra, North Indian, Chinese, South Indian', 'Andhra, North Indian, South Indian', 'Andhra, North Indian, South Indian, Chinese', 'Andhra, Seafood, Biryani', 'Andhra, Seafood, North Indian, Chinese', 'Andhra, South Indian', 'Andhra, South Indian, Biryani', 'Andhra, South Indian, Biryani, North Indian', 'Andhra, South Indian, Biryani, North Indian, Mangalorean', 'Andhra, South Indian, Chettinad', 'Andhra, South Indian, Chinese', 'Andhra, South Indian, Chinese, North Indian', 'Andhra, South Indian, Fast Food', 'Andhra, South Indian, Hyderabadi', 'Andhra, South Indian, North Indian', 'Andhra, South Indian, North Indian, Biryani', 'Andhra, South Indian, North Indian, Chinese', 'Arabian', 'Arabian, Afghan']
    
    return render_template("index.html", city=city, rest_type=restaurant_types, cuisines=cuisines)

@app.route("/", methods=["POST"])
def predictor():
    if request.method == "POST":
        data = {
            'online_order': 1 if request.form.get("online_order") == "Yes" else 0,
            'book_table': 1 if request.form.get("book_table") == "Yes" else 0,
            'votes': int(request.form.get("votes", 0)),
            'avg_cost': float(request.form.get("avg_cost", 0)),
            'city': request.form.getlist("city"),
            'rest_type': request.form.getlist("rest_type"),
            'cuisines': request.form.getlist("cuisines")
        }

        # Encode location
        if data['city'] in location_encoder.classes_:
            data['city'] = location_encoder.transform([data['city']])[0]
        else:
            data['city'] = -1

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
        input_features = [data['online_order'], data['book_table'], data['votes'], data['avg_cost'], data['city'],data['rest_type'],data['cuisines']] 

        # Convert to DataFrame
        df = pd.DataFrame([input_features], columns=['online_order', 'book_table', 'votes','rest_type', 'cuisines','avg_cost', 'city'])



        # Scaling numeric features
        numeric_features = ['online_order', 'book_table', 'votes', 'rest_type','cuisines','avg_cost', 'city']

        # Use the loaded scaler to transform the features
        scaled_features = scaler.transform(df[numeric_features])

        # scaler = StandardScaler()
        # df[numeric_features] = scaler.fit_transform(df[numeric_features])
# Convert the numpy array back to a DataFrame
        df_scaled = pd.DataFrame(scaled_features, columns=numeric_features)

        print(input_features)
        print(df_scaled)
        # Making Predictions
        x_input = df_scaled.values
        print(x_input)
        rate = model.predict(x_input)[0]
        output = f"{round(rate, 1)} / 5"
        return render_template("index.html", prediction_text=output, city=[
            'Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
            'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
            'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
            'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
            'Koramangala 4th Block', 'Koramangala 5th Block',
            'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
            'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
            'Old Airport Road', 'Rajajinagar', 'Residency Road',
            'Sarjapur Road', 'Whitefield'
        ], rest_type = [
    'Bakery', 'Bakery, Beverage Shop', 'Bakery, Cafe', 'Bakery, Dessert Parlor',
    'Bakery, Food Court', 'Bakery, Kiosk', 'Bakery, Quick Bites', 'Bakery, Sweet Shop',
    'Bar', 'Bar, Cafe', 'Bar, Casual Dining', 'Bar, Lounge', 'Bar, Pub', 'Bar, Quick Bites',
    'Beverage Shop', 'Beverage Shop, Cafe', 'Beverage Shop, Dessert Parlor', 'Beverage Shop, Quick Bites',
    'Bhojanalya', 'Cafe', 'Cafe, Bakery', 'Cafe, Bar', 'Cafe, Casual Dining', 'Cafe, Dessert Parlor',
    'Cafe, Food Court', 'Cafe, Lounge', 'Cafe, Quick Bites', 'Casual Dining', 'Casual Dining, Bar',
    'Casual Dining, Cafe', 'Casual Dining, Irani Cafee', 'Casual Dining, Lounge', 'Casual Dining, Microbrewery',
    'Casual Dining, Pub', 'Casual Dining, Quick Bites', 'Casual Dining, Sweet Shop', 'Club', 'Club, Casual Dining',
    'Confectionery', 'Delivery', 'Dessert Parlor', 'Dessert Parlor, Bakery', 'Dessert Parlor, Beverage Shop',
    'Dessert Parlor, Cafe', 'Dessert Parlor, Food Court', 'Dessert Parlor, Kiosk', 'Dessert Parlor, Quick Bites',
    'Dessert Parlor, Sweet Shop', 'Dhaba', 'Fine Dining', 'Fine Dining, Bar', 'Fine Dining, Lounge',
    'Fine Dining, Microbrewery', 'Food Court', 'Food Court, Beverage Shop', 'Food Court, Casual Dining',
    'Food Court, Dessert Parlor', 'Food Court, Quick Bites', 'Food Truck', 'Kiosk', 'Lounge', 'Lounge, Bar',
    'Lounge, Cafe', 'Lounge, Casual Dining', 'Lounge, Microbrewery', 'Mess', 'Mess, Quick Bites', 'Microbrewery',
    'Microbrewery, Bar', 'Microbrewery, Casual Dining', 'Microbrewery, Lounge', 'Microbrewery, Pub', 'Pop Up',
    'Pub', 'Pub, Bar', 'Pub, Cafe', 'Pub, Casual Dining', 'Pub, Microbrewery', 'Quick Bites', 'Quick Bites, Bakery',
    'Quick Bites, Beverage Shop', 'Quick Bites, Cafe', 'Quick Bites, Dessert Parlor', 'Quick Bites, Food Court',
    'Quick Bites, Kiosk', 'Quick Bites, Meat Shop', 'Quick Bites, Mess', 'Quick Bites, Sweet Shop', 'Sweet Shop',
    'Sweet Shop, Dessert Parlor', 'Sweet Shop, Quick Bites', 'Takeaway', 'Takeaway, Delivery'
],
       cuisines = ['African, Burger', 'African, Burger, Desserts, Beverages, Fast Food', 'American', 'American, Asian, Continental, North Indian, South Indian, Chinese', 'American, Asian, European, North Indian', 'American, BBQ', 'American, Bakery, Beverages, Cafe, Healthy Food, Juices, North Indian, Sandwich', 'American, Burger, Fast Food', 'American, Burger, Italian, Steak', 'American, Burger, Momos, Bengali', 'American, Cafe, Chinese, Italian, Desserts', 'American, Cafe, Continental', 'American, Cafe, Continental, French, Burger, Mexican, Desserts, Pizza', 'American, Chinese', 'American, Chinese, Continental, North Indian', 'American, Continental', 'American, Continental, BBQ, Steak', 'American, Continental, Chinese', 'American, Continental, Fast Food, Steak', 'American, Continental, Finger Food', 'American, Continental, North Indian, Mediterranean', 'American, Continental, North Indian, Salad', 'American, Continental, North Indian, South Indian', 'American, Continental, Pizza', 'American, Continental, Salad', 'American, Continental, Salad, Italian', 'American, Continental, Salad, Italian, Asian', 'American, Continental, Steak', 'American, Continental, Steak, Salad', 'American, European', 'American, European, Healthy Food', 'American, Fast Food', 'American, Finger Food', 'American, Finger Food, BBQ, Steak', 'American, Finger Food, Italian, Mexican', 'American, Goan', 'American, Italian', 'American, Italian, Bakery, Desserts, Sandwich, Beverages, Salad, Rolls', 'American, Mexican', 'American, Mexican, BBQ', 'American, Mexican, Italian, Steak', 'American, Modern Indian, Italian, South Indian', 'American, North Indian, Chinese', 'American, North Indian, Chinese, Finger Food', 'American, North Indian, Chinese, Finger Food, Momos', 'American, North Indian, European, Tex-Mex', 'American, North Indian, Pizza, Finger Food, Continental, Italian', 'American, North Indian, Salad', 'American, Pizza', 'American, Pizza, Burger', 'American, Sandwich', 'American, South Indian, Thai, Pizza, Italian', 'American, Tex-Mex, Burger, BBQ', 'American, Tex-Mex, Burger, BBQ, Mexican', 'American, Thai, Healthy Food', 'Andhra', 'Andhra, Beverages, Biryani, Chinese, Fast Food, Hyderabadi, North Indian, South Indian', 'Andhra, Biryani', 'Andhra, Biryani, Beverages, Kebab', 'Andhra, Biryani, Chinese', 'Andhra, Biryani, Chinese, Hyderabadi, North Indian, South Indian', 'Andhra, Biryani, Chinese, North Indian', 'Andhra, Biryani, Chinese, North Indian, Seafood, South Indian', 'Andhra, Biryani, Chinese, South Indian', 'Andhra, Biryani, Mughlai', 'Andhra, Biryani, North Indian', 'Andhra, Biryani, North Indian, Beverages', 'Andhra, Biryani, North Indian, Chinese', 'Andhra, Biryani, North Indian, South Indian', 'Andhra, Biryani, Seafood', 'Andhra, Biryani, Seafood, North Indian', 'Andhra, Biryani, South Indian', 'Andhra, Biryani, South Indian, Chinese', 'Andhra, Chettinad', 'Andhra, Chinese', 'Andhra, Chinese, Biryani', 'Andhra, Chinese, Continental', 'Andhra, Chinese, North Indian', 'Andhra, Chinese, North Indian, Biryani', 'Andhra, Chinese, North Indian, Biryani, Beverages', 'Andhra, Chinese, North Indian, Hyderabadi', 'Andhra, Chinese, North Indian, Seafood', 'Andhra, Chinese, North Indian, South Indian', 'Andhra, Chinese, South Indian', 'Andhra, Chinese, South Indian, North Indian', 'Andhra, Continental, Chinese', 'Andhra, Fast Food', 'Andhra, Hyderabadi', 'Andhra, Hyderabadi, Biryani', 'Andhra, Hyderabadi, Biryani, Chinese, North Indian', 'Andhra, Hyderabadi, Chinese', 'Andhra, Hyderabadi, North Indian', 'Andhra, Kerala, South Indian', 'Andhra, North Indian', 'Andhra, North Indian, Biryani', 'Andhra, North Indian, Biryani, Beverages', 'Andhra, North Indian, Biryani, Seafood', 'Andhra, North Indian, Chinese', 'Andhra, North Indian, Chinese, Biryani', 'Andhra, North Indian, Chinese, Biryani, Seafood', 'Andhra, North Indian, Chinese, Mangalorean', 'Andhra, North Indian, Chinese, Mangalorean, Seafood, Biryani', 'Andhra, North Indian, Chinese, Seafood', 'Andhra, North Indian, Chinese, Seafood, Biryani', 'Andhra, North Indian, Chinese, South Indian', 'Andhra, North Indian, South Indian', 'Andhra, North Indian, South Indian, Chinese', 'Andhra, Seafood, Biryani', 'Andhra, Seafood, North Indian, Chinese', 'Andhra, South Indian', 'Andhra, South Indian, Biryani', 'Andhra, South Indian, Biryani, North Indian', 'Andhra, South Indian, Biryani, North Indian, Mangalorean', 'Andhra, South Indian, Chettinad', 'Andhra, South Indian, Chinese', 'Andhra, South Indian, Chinese, North Indian', 'Andhra, South Indian, Fast Food', 'Andhra, South Indian, Hyderabadi', 'Andhra, South Indian, North Indian', 'Andhra, South Indian, North Indian, Biryani', 'Andhra, South Indian, North Indian, Chinese', 'Arabian', 'Arabian, Afghan']     
    
        
        )   
     
    return render_template("index.html", prediction_text="output")

if __name__ == "__main__":
    app.run(debug=True)
