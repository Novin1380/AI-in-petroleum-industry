import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
import pickle
import time
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="Reservoir Production Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Caching and Loading Models & Data -----------------
# (Loading functions remain the same as before)
@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

@st.cache_data
def load_data(path):
    try:
        try:
            return pd.read_csv(path)
        except:
            return pd.read_excel(path)
    except Exception as e:
        st.error(f"Error loading data from {path}: {e}")
        return None

@st.cache_resource
def load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading joblib file from {path}: {e}")
        return None

@st.cache_data
def load_history(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.warning(f"History file not found at {path}. Displaying empty plot.")
        return {'loss': [], 'val_loss': []}

# --- Load all necessary files ---
model_main = load_model('model_main.keras')
model_outlier = load_model('model_outlier.keras')
y_scaler = load_joblib('y_scaler.gz')
num_scaler = load_joblib('num_scaler.gz')
image_params = load_joblib('image_params.gz')
y_scaler_out = load_joblib('y_scaler_out.gz')
num_scaler_out = load_joblib('num_scaler_out.gz')
image_params_out = load_joblib('image_params_out.gz')
history_main = load_history('training_history_main.pkl')
history_outlier = load_history('training_history_outlier.pkl')
processed_data = load_data('processed_tabular_data.csv')
prediction_main = load_data('full_dataset_predictions.xlsx')
prediction_outlier = load_data('out_dataset_predictions.xlsx')

if processed_data is not None:
    target_cols = [col for col in processed_data.columns if col not in ['sample number', 'Initial Sw']]
else:
    target_cols = [f"Target_{i}" for i in range(1, 13)]

# ----------------- Page Definitions -----------------

def page_home():
    st.title("Deep Learning for Reservoir Production Forecasting")
    st.markdown("---")
    st.header("Project Overview")
    st.markdown("""
    This project focuses on developing a deep learning-based proxy model to forecast oil production in hydrocarbon reservoirs, replacing computationally intensive traditional reservoir simulations. The model uses deep neural networks and image processing to predict oil production rates and cumulative production based on reservoir property maps.

    **Dataset**:
    
    1,050 reservoir model samples, each including:
    - Permeability and porosity maps (64×64 TIFF grids).
    - Initial water saturation (Sw) values and production data (oil rates and cumulative production) for months 1, 3, 5, 7, 9, and 11 from an Excel file.

    **Inputs**:
    - Permeability map (64×64).
    - Porosity map (64×64).
    - Initial water saturation (Sw).

    **Target Outputs**:
    - Oil production rates for months 1, 3, 5, 7, 9, and 11.
    - Cumulative oil production for the same months.

    **Objective**:
    Create a fast, accurate proxy model to predict production metrics, enabling rapid scenario evaluation and optimization in reservoir engineering with reduced computational cost.
    
    Created by: $$Novin Nekuee (403134029)$$ & $$Soroush Danesh (810403045)$$""")



def page_loss_curves():
    st.header("Training and Validation Loss Curves")
    model_choice = st.selectbox("Select Model:", ("Trained on All Data (Main Model)", "Trained After Outlier Removal"))
    history = history_main if model_choice == "Trained on All Data (Main Model)" else history_outlier

    if not history or not history.get('loss'):
        st.warning("No history data available to plot.")
        return

    loss = history['loss']
    val_loss = history['val_loss']
    df = pd.DataFrame({
        "Train Loss": loss,
        "Validation Loss": val_loss
    })
    info_placeholder = st.empty()
    with info_placeholder.container():
        st.info("Animating the training history...")
    
    chart = st.line_chart(df.iloc[:1])

    for i in range(2, len(df)+1):
        chart.add_rows(df.iloc[i-1:i])
        time.sleep(0.05)
    info_placeholder.empty()
    with info_placeholder.container():
        st.success("Animation complete!")
    time.sleep(3)
    info_placeholder.empty()

def page_dataset():
    st.header("Processed Tabular Dataset Explorer")

    # Dropdown for dataset selection
    dataset_option = st.selectbox(
        "Select Dataset to View:",
        [
            "Clean dataset",
            "Prediction (Main Model)",
            "Prediction (Outlier Model)"
        ]
    )

    # Load and display the selected dataset
    if dataset_option == "Clean dataset":
        st.write("This table contains the final, cleaned, and pivoted data used for training the models.")
        if processed_data is not None:
            st.dataframe(processed_data)
        else:
            st.warning("Could not load processed_tabular_data.csv")
    elif dataset_option == "Prediction (Main Model)":
        try:
            st.write("Predictions from the main model:")
            st.dataframe(prediction_main)
        except Exception as e:
            st.warning(f"Could not load full_dataset_predictions.xlsx: {e}")
    elif dataset_option == "Prediction (Outlier Model)":
        try:
            st.write("Predictions from the outlier-removed model:")
            st.dataframe(prediction_outlier)
        except Exception as e:
            st.warning(f"Could not load out_dataset_predictions.xlsx: {e}")

def page_feature_analysis():
    st.header("Actual vs. Predicted Values Analysis")

    feature_choice = st.selectbox("Select a Target Feature to Plot:", target_cols)
    model_choice = st.selectbox(
        "Select Prediction Model:",
        ("Trained on All Data (Main Model)", "Trained After Outlier Removal"),
        key="analysis_model"
    )

    pred_df = prediction_main if model_choice == "Trained on All Data (Main Model)" else prediction_outlier

    if processed_data is None or pred_df is None:
        st.warning("Required data not loaded.")
        return

    merged = pd.merge(
        processed_data[['sample number', feature_choice]],
        pred_df[['sample number', feature_choice]],
        on='sample number',
        suffixes=('_actual', '_predicted')
    )

    if merged.empty:
        st.warning("No matching data found for the selected feature.")
        return

    plot_df = pd.DataFrame({
        "Actual": merged[f"{feature_choice}_actual"],
        "Predicted": merged[f"{feature_choice}_predicted"],
        "Type": "Data"
    })

    min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
    line_df = pd.DataFrame({
        "Actual": [min_val, max_val],
        "Predicted": [min_val, max_val],
        "Type": "Ideal"
    })

    full_df = pd.concat([plot_df, line_df], ignore_index=True)

    st.scatter_chart(
        full_df,
        x="Actual",
        y="Predicted",
        color="Type"
    )


def page_live_prediction():
    st.header("Live Production Forecast")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input Parameters")
        sw_input = st.number_input("Initial Water Saturation (Sw)", 0.0, 1.0, 0.25, 0.01)
        month_input = st.selectbox("Select Target Month:", [1, 3, 5, 7, 9, 11])
        # Model selection dropdown
        model_choice = st.selectbox(
            "Select Prediction Model:",
            ("Trained on All Data (Main Model)", "Trained After Outlier Removal"),
            key="live_pred_model"
        )
    with c2:
        st.subheader("Input Maps (.tiff)")
        perm_file = st.file_uploader("Upload Permeability Map", type=['tiff', 'tif'])
        poro_file = st.file_uploader("Upload Porosity Map", type=['tiff', 'tif'])

    if st.button("Run Forecast", type="primary"):
        # Select model based on dropdown
        if model_choice == "Trained on All Data (Main Model)":
            selected_model = model_main
            selected_num_scaler = num_scaler
            selected_y_scaler = y_scaler
            selected_image_params = image_params
            # Normalize with saved parameters (per-channel)
            perm_min, perm_max = selected_image_params['perm_min'], selected_image_params['perm_max']
            poro_min, poro_max = selected_image_params['poro_min'], selected_image_params['poro_max']
        else:
            selected_model = model_outlier
            selected_num_scaler = num_scaler_out
            selected_y_scaler = y_scaler_out
            selected_image_params = image_params_out
            # Normalize with saved parameters (per-channel)
            perm_min, perm_max = selected_image_params['perm_min_out'], selected_image_params['perm_max_out']
            poro_min, poro_max = selected_image_params['poro_min_out'], selected_image_params['poro_max_out']
        if perm_file and poro_file and all([selected_model, y_scaler, num_scaler, image_params]):
            with st.spinner("Processing inputs and running model..."):
                try:
                    # 1. Process Images
                    perm_img = Image.open(perm_file)
                    poro_img = Image.open(poro_file)

                    # Ensure both images are the same size and single-channel
                    perm_array = np.array(perm_img, dtype=np.float32)
                    poro_array = np.array(poro_img, dtype=np.float32)

                    if perm_array.shape != poro_array.shape:
                        st.error("Permeability and Porosity maps must have the same shape!")
                        return

                    # If images are 2D, add channel axis
                    if perm_array.ndim == 2:
                        perm_array = perm_array[..., np.newaxis]
                    if poro_array.ndim == 2:
                        poro_array = poro_array[..., np.newaxis]

                    # Stack to shape (H, W, 2)
                    combined_image = np.concatenate([perm_array, poro_array], axis=-1)

                    

                    combined_image_scaled = combined_image.copy()
                    if (perm_max - perm_min) != 0:
                        combined_image_scaled[..., 0] = (combined_image[..., 0] - perm_min) / (perm_max - perm_min)
                    if (poro_max - poro_min) != 0:
                        combined_image_scaled[..., 1] = (combined_image[..., 1] - poro_min) / (poro_max - poro_min)

                    combined_image_scaled = np.nan_to_num(combined_image_scaled)

                    # Add batch dimension: (1, H, W, 2)
                    final_image_input = np.expand_dims(combined_image_scaled, axis=0)

                    # 3. Process Numerical Input
                    final_num_input = selected_num_scaler.transform(np.array([[sw_input]], dtype=np.float32))

                    # 4. Make Prediction (check input names)
                    model_inputs = selected_model.input_names if hasattr(selected_model, "input_names") else [i.name for i in selected_model.inputs]
                    input_dict = {}
                    for name in model_inputs:
                        if "image" in name:
                            input_dict[name] = final_image_input
                        elif "num" in name or "sw" in name:
                            input_dict[name] = final_num_input

                    prediction_scaled = selected_model.predict(input_dict)

                    # 5. Inverse Transform and Post-Process
                    prediction_original = selected_y_scaler.inverse_transform(prediction_scaled)
                    prediction_original[prediction_original < 0] = 0  # Clip negative values

                    # 6. Select Output for the Chosen Month
                    month_to_index = {1: [0, 6], 3: [1, 7], 5: [2, 8], 7: [3, 9], 9: [4, 10], 11: [5, 11]}
                    indices = month_to_index[month_input]
                    cum_oil_pred = prediction_original[0, indices[0]]
                    oil_rate_pred = prediction_original[0, indices[1]]

                    st.success("Forecast Complete!")
                    st.subheader(f"Predicted Results for Month {month_input}")
                    col1, col2 = st.columns(2)
                    col1.metric("Cumulative Oil (M m3)", f"{cum_oil_pred:.2f}")
                    col2.metric("Oil Rate (m3/day)", f"{oil_rate_pred:.2f}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.error("Please upload both permeability and porosity maps.")

# ----------------- Sidebar and Main Navigation -----------------
with st.sidebar:
    
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
        min-width: 290px;
        max-width: 400px;
        width: 350px;
        }
        .sidebar-logo {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 1rem;
        }
        .sidebar-logo img {
            width: 80px;
            border-radius: 20px;
        }
        .sidebar-name {
            font-size: 1.2rem;
            font-weight: bold;
            color: #444;
            margin-top: 0.6rem;
        }
        .sidebar-names {
            font-size: 1rem;
            color: #444;
            margin-top: 0.6rem;
        }
        </style>
    """, unsafe_allow_html=True)
    selected_page = option_menu(
        menu_title="Dashboard",
        options=["Home", "Loss Curves", "Dataset Explorer", "Feature Analysis", "Live Prediction"],
        icons=["house", "activity", "database", "bar-chart-line", "cpu"],
        menu_icon="cast",
        default_index=0,

    )
    st.markdown(
        '''
        <div class="sidebar-logo">
            <img src="data:image/png;base64,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" alt="Logo" />
            <div class="sidebar-name">Created by:<br></div>
            <div class="sidebar-names">Novin Nekuee & Soroush Danesh</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

# --- Check if all necessary files were loaded successfully ---
files_loaded = all(v is not None for v in [model_main, model_outlier, y_scaler, num_scaler, image_params, history_main, history_outlier, processed_data])

# --- Display the selected page ---
if files_loaded:
    if selected_page == "Home":
        page_home()
    elif selected_page == "Loss Curves":
        page_loss_curves()
    elif selected_page == "Dataset Explorer":
        page_dataset()
    elif selected_page == "Feature Analysis":
        page_feature_analysis()
    elif selected_page == "Live Prediction":
        page_live_prediction()
else:
    st.sidebar.error("One or more essential files could not be loaded. Please check the prerequisites.")
    st.header("Welcome!")
    st.warning("Please place all the required files in the same directory as the app and refresh the page.")