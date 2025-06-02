# Piston Anomaly Detection

This project provides a web application for detecting anomalies in piston images using a machine learning model trained with Teachable Machine. The app allows users to upload or capture images and predicts whether the piston is "Normal" or "Anomaly".

## Features

- Upload or capture piston images directly in the browser
- Automatic image preprocessing and resizing
- Real-time anomaly detection using a pre-trained TensorFlow model
- Displays prediction label and confidence score

## Requirements

- Python 3.7+
- See [requirements.txt](requirements.txt) for Python dependencies

## Getting Started

1. **Clone and open the repository:**
   ```sh
   git clone https://github.com/shaayar/piston-anomaly-detection.git
   cd piston-anomaly-detection
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Add your trained model**
   - Place your TensorFlow/Keras model in a folder named model/ in the project root.
   - Ensure the model is saved in the TensorFlow SavedModel format.
   - The folder structure should look like this:
    ```
    piston-anomaly-detection/
    ├── app.py
    ├── model/
    │   └── saved_model.h5
    │   └── labels.txt
    ├── requirements.txt
    └── README.md
    ``` 

4. **Run the application:**
   ```sh
    streamlit run app.py
    ```

5. Usage:

- Open the provided local URL in your browser.
- Choose to upload an image or capture one using your webcam.
- View the prediction and confidence score.

## Notes
- The model should be compatible with TensorFlow's load_model and accept images of size 224x224x3.
- Class names are currently set to ["Normal", "Anomaly"]. Adjust in app.py if your model uses different labels.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.