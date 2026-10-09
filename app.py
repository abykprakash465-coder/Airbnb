import streamlit as st
import pandas as pd
import joblib

# Load your newly saved Airbnb model
MODEL_PATH = "airbnb.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🛏️")

st.title("🛏️ Airbnb Nightly Price Predictor")
st.caption("Educational demonstration using Airbnb listing data.")

# Set up the UI inputs to match your exact X_train features
name = st.text_input("Listing Title (Name)", "Luxury apartment with a beautiful view")

col1, col2 = st.columns(2)

with col1:
    neighbourhood_group = st.selectbox(
        "Neighbourhood Group", 
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )
    room_type = st.selectbox(
        "Room Type", 
        ["Entire home/apt", "Private room", "Shared room"]
    )
    latitude = st.number_input("Latitude", value=40.7831, format="%.4f")
    longitude = st.number_input("Longitude", value=-73.9712, format="%.4f")
    
with col2:
    minimum_nights = st.number_input("Minimum Nights", 1, 365, 3)
    number_of_reviews = st.number_input("Number of Reviews", 0, 5000, 25)
    reviews_per_month = st.number_input("Reviews Per Month", 0.0, 50.0, 1.5, 0.1)
    calculated_host_listings_count = st.number_input("Host Total Listings", 1, 500, 1)
    availability_365 = st.number_input("Availability (Days per year)", 0, 365, 120)


if st.button("Predict Nightly Price"):
    # Construct the dataframe exactly as your ColumnTransformer expects it
    row = pd.DataFrame([{
        "name": name,
        "neighbourhood_group": neighbourhood_group,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365
    }])

    # Generate the regression prediction
    prediction = model.predict(row)[0]

    # Display the final dollar amount
    st.success(f"Predicted Nightly Price: ${prediction:.2f}")

st.info("Educational demonstration only. Predictions should not be treated as guaranteed market outcomes.")
