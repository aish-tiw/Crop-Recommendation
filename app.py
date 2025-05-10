import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Load model components
model = joblib.load("voting_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
X_train = joblib.load("X_train.pkl")

# Feature settings
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
valid_ranges = {
    'N': (0, 140),
    'P': (5, 145),
    'K': (5, 205),
    'temperature': (8, 45),
    'humidity': (10, 100),
    'ph': (3.5, 9.5),
    'rainfall': (10, 300)
}

# Streamlit UI
st.title("Crop Recommendation System")
st.write("Select top 3 crops you can grow based on your soil and weather data.")


st.subheader("Enter Detail")

# Input sliders
user_input = {}
for feature in feature_names:
    min_val, max_val = valid_ranges[feature]
    default = (min_val + max_val) / 2
    user_input[feature] = st.slider(f"{feature.capitalize()} ({min_val}-{max_val})", float(min_val), float(max_val), float(default))

if st.button("Predict Best Crops"):
    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df)

    # Top 3 predictions
    probabilities = model.predict_proba(scaled_input)[0]
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_labels = encoder.inverse_transform(top3_indices)
    top3_probs = probabilities[top3_indices]

    st.subheader("🌱 Top 3 Recommended Crops:")
    for i in range(3):
        st.markdown(f"**{i+1}. {top3_labels[i]}** – Confidence: {top3_probs[i]*100:.2f}%")

    st.markdown("---")
    st.subheader("Why these Crops?")
    st.markdown("""
    - **Green bars** show features that helped.
    - **Red bars** show features that worked against it.
    The longer the bar, the more it affected the decision.
    """)

    # LIME explanation
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=encoder.classes_,
        mode='classification'
    )

    for i, label_index in enumerate(top3_indices):
        st.markdown(f"#### Recommendation: **{encoder.inverse_transform([label_index])[0]}**")

        explanation = explainer.explain_instance(
            scaled_input[0],
            model.predict_proba,
            labels=[label_index],
            num_features=5
        )

        fig = explanation.as_pyplot_figure(label=label_index)
        st.pyplot(fig)

        # Short explanation
        st.markdown("**Summary:**")
        brief_text = " | ".join([
            f"{feat.split(' ')[0]} {('↑' if weight > 0 else '↓')} impact"
            for feat, weight in explanation.as_list(label=label_index)
        ])
        st.info(brief_text)

