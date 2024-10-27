import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def galaxy_cluster_analysis():
    # Load the data
    sky = pd.read_csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv")

    # Drop non-informative columns and select specific features
    features_to_use = ["ra", "dec", "u", "g", "r", "i", "z", "redshift"]
    sky = sky[features_to_use + ["class"]]

    # Encode the target variable
    le = preprocessing.LabelEncoder()
    sky["class"] = le.fit_transform(sky["class"])

    # Prepare the features and target
    X = sky.drop("class", axis=1)
    y = sky["class"]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Train a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Streamlit app
    st.markdown(
        """
        <h1 style="text-align: center;">Galaxy Cluster Analysis</h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center; font-weight: bold; font-size: 18px;">
            Utilize machine learning algorithms to classify galaxies into types such as spiral, elliptical, and irregular based on key features like morphology, brightness, and color. <br>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create individual input fields for each feature

    # Create four columns for the description
    col1, col2, col3, col4 = st.columns(4)

    with col2:
        ra = st.text_input("Right Ascension (RA)")
        dec = st.text_input("Declination (DEC)")
        u_mag = st.text_input("u band magnitude")
        g_mag = st.text_input("g band magnitude")

    with col3:
        r_mag = st.text_input("r band magnitude")
        i_mag = st.text_input("i band magnitude")
        z_mag = st.text_input("z band magnitude")
        redshift = st.text_input("Redshift")

    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)

    with c5:
        # Predict button
        if st.button("Predict"):
            try:
                # Convert inputs to float and create feature list
                input_features = [
                    float(ra),
                    float(dec),
                    float(u_mag),
                    float(g_mag),
                    float(r_mag),
                    float(i_mag),
                    float(z_mag),
                    float(redshift),
                ]

                # Reshape and scale the input data
                input_data_scaled = scaler.transform(
                    np.array(input_features).reshape(1, -1)
                )

                # Make prediction
                prediction = rf.predict(input_data_scaled)
                class_name = le.inverse_transform(prediction)[0]

                # Display result
                st.success(f"The predicted galaxy class is: {class_name}")

            except ValueError as e:
                st.error("Please ensure all inputs are valid numbers.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "_main_":
    galaxy_cluster_analysis()
