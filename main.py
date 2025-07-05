import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model('training_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Top Bar Simulation
st.markdown("""
    <style>
        .top-bar {
            display: flex;
            justify-content: space-around;
            background-color: #4CAF50;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .top-bar button {
            font-size: 18px;
            color: white;
            background-color: #4CAF50;
            border: none;
            cursor: pointer;
        }
        .top-bar button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Top navigation bar
app_mode = st.radio(
    "",
    ["Home", "About", "Plant Disease Prediction"],
    horizontal=True,
    label_visibility="collapsed"
)

# Home Page
if app_mode == "Home":
    st.markdown("<h2 style='text-align: center; color: green;'>🌿 Plant Disease Recognition System 🌿</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**, your go-to solution for identifying plant diseases quickly and accurately.  
    **Why it matters?** 🌾 Early detection of plant diseases is crucial for safeguarding crops and ensuring a bountiful harvest.  
    ### Key Features
    - **AI-Powered**: Using advanced deep learning models for precise detection.
    - **User-Friendly Interface**: Intuitive and easy to use.
    - **Real-Time Analysis**: Upload an image and get results instantly.
    
    ### Advantages
    1. **Accuracy:** AI-powered predictions ensure high accuracy.
    2. **User-Friendly:** Simple and intuitive interface for non-technical users.
    3. **Time-Saving:** Instant results compared to manual diagnosis.
    4. **Scalability:** Can handle a wide variety of plant diseases.

    💡 **Explore the About section** to learn more about the dataset and technology used.
    """)
    
# About Page
elif app_mode == "About":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌿 About This Project 🌿</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### About the Dataset
    - **Source**: Derived from an open-source repository with additional augmentation.  
    - **Classes**: 38 categories, including healthy and diseased crop images.
    - **Dataset Structure**:
      1. **Training Set**: 70,295 images  
      2. **Validation Set**: 17,572 images  
      3. **Test Set**: 33 images  
    
    ### Goal
    This model helps farmers by providing early disease detection, which protects crops and supports healthier yields.
    It also assists agriculture experts in monitoring and maintaining crop health.
    """)

# Disease Recognition Page
elif app_mode == "Plant Disease Prediction":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌿 Plant Disease Prediction 🌿</h1>", unsafe_allow_html=True)
    st.markdown("### Upload an image of the plant and let our AI model analyze it for potential diseases.")
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, caption="Uploaded Image")
        else:
            st.warning("Please upload an image first!")
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("Processing the image... ⏳"):  
                result_index = model_prediction(test_image)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                            'Tomato___healthy']
            st.success(f"🌟 Our model predicts the disease as: **{class_name[result_index]}**")
        else:
            st.warning("Please upload an image before predicting!")
