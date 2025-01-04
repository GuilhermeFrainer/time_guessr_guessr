import streamlit as st
import pathlib
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from scaler import Scaler
import plotly.graph_objects as go
import polars as pl
import plotly.express as px


MODEL_DIR = pathlib.Path("models")
MODEL_PATH = MODEL_DIR / "dummy_model.keras"


def main():
    model = load_model(MODEL_PATH)

    image_file = st.file_uploader("Upload your image file", type=["jpg", "png"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Image")

        image = image.resize((224, 224))
        arr = np.array(image)
        arr = np.expand_dims(arr, axis=0)

        scaled_pred = model.predict(arr)
        pred = Scaler.descale_preds(scaled_pred)
        pred

        latitude = -31
        longitude = -50

        fig = go.Figure(go.Scattermapbox(
            lat=[latitude],
            lon=[longitude],
            mode='markers',
            marker=dict(size=12, color='blue'),
            text='Predicted Location'
        ))

        # Set the layout of the map
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",  # Open Street Map (free to use)
                center=dict(lat=latitude, lon=longitude),
                zoom=2
            ),
            title="Predicted Location"
        )

        # Display the map
        st.plotly_chart(fig)



if __name__ == "__main__":
    main()