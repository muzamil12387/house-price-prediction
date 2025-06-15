import streamlit as st
import pandas as pd
import pickle

class HousePricePredictorApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        try:
            with open("svm_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model or scaler: {e}")

    def user_input_features(self):
        st.sidebar.header("Input House Features")
        area = st.sidebar.number_input("Area (sq ft)", 500, 10000, step=50)
        bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
        stories = st.sidebar.slider("Stories", 1, 4, 1)
        parking = st.sidebar.slider("Parking spaces", 0, 5, 1)

        mainroad = st.sidebar.selectbox("On Main Road", ["Yes", "No"])
        guestroom = st.sidebar.selectbox("Has Guest Room", ["Yes", "No"])
        basement = st.sidebar.selectbox("Has Basement", ["Yes", "No"])
        hotwaterheating = st.sidebar.selectbox("Has Hot Water Heating", ["Yes", "No"])
        airconditioning = st.sidebar.selectbox("Has Air Conditioning", ["Yes", "No"])
        furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])

        data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': 1 if mainroad == "Yes" else 0,
            'guestroom': 1 if guestroom == "Yes" else 0,
            'basement': 1 if basement == "Yes" else 0,
            'hotwaterheating': 1 if hotwaterheating == "Yes" else 0,
            'airconditioning': 1 if airconditioning == "Yes" else 0,
            'furnishingstatus_unfurnished': 1 if furnishingstatus == "Unfurnished" else 0,
            'furnishingstatus_semi-furnished': 1 if furnishingstatus == "Semi-Furnished" else 0,
            'furnishingstatus_furnished': 1 if furnishingstatus == "Furnished" else 0,
        }
        return pd.DataFrame([data])

    def predict_price(self, input_df):
        try:
            scaled_input = self.scaler.transform(input_df)
            prediction = self.model.predict(scaled_input)
            return prediction[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

    def run(self):
        st.title("House Price Prediction App")
        st.write("Enter the details in the sidebar to predict house price.")

        input_df = self.user_input_features()
        st.subheader("Input Summary")
        st.write(input_df)

        if st.button("Predict Price"):
            price = self.predict_price(input_df)
            if price is not None:
                st.success(f"Predicted House Price: ₹ {price:,.2f}")

# Run the app
if __name__ == "__main__":
    app = HousePricePredictorApp()
    app.run()

