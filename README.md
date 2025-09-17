This project trains an Optical Character Recognition (OCR) model to accurately transcribe text from CAPTCHA images. The script is self-contained and handles the entire machine learning workflow:

* **Data Acquisition:** Automatically downloads and unzips the image dataset from below mentioned github repo.
* **Preprocessing:** Processes images and labels for use in a TensorFlow data pipeline.
* **Model Building:** Defines a custom Convolutional Recurrent Neural Network (CRNN) architecture.
* **Training & Evaluation:** Trains the model using a custom CTC loss function and evaluates its accuracy.
* **Model Saving:** Saves the final trained model for inference.

## Dataset Used

The script uses the **Captcha Images V2** dataset.

*  The dataset is automatically downloaded from a public [GitHub repository](https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip).
*  It consists of CAPTCHA images where the filename serves as the ground-truth label (e.g., `2b827.png`). The script identifies all unique characters to build its vocabulary.

---

## About The Code

The core of the project is the model architecture and the custom loss function, which work together to solve this sequence-to-sequence task.

### Model Architecture (`build_model` function)

The model is a **Convolutional Recurrent Neural Network (CRNN)**, a standard and powerful architecture for OCR tasks. It is composed of several stages:

1.  **Input Layer:** The model is defined with two inputs: the CAPTCHA image and the corresponding text labels. The labels are required during training for the loss calculation.

2.  **Convolutional Base (CNN):** Two blocks of `Conv2D` and `MaxPooling2D` layers serve as the visual feature extractor. These layers scan the image to learn spatial patterns like edges and shapes.

3.  **Map-to-Sequence Transformation:** A `Reshape` layer transforms the 2D feature map from the CNN into a sequence of feature vectors. This acts as the bridge between the model's visual understanding and its sequential processing.

4.  **Recurrent Layers (RNN):** Two stacked `Bidirectional LSTM` layers process the sequence of features. Bidirectional processing allows the model to learn context from characters to the left and right, which is crucial for accurately identifying ambiguous characters.

5.  **Output Layer:** A final `Dense` layer with a `softmax` activation outputs a probability distribution over all characters in the vocabulary, plus an additional "blank" character required by the CTC loss function.

### The CTC Loss Function (`CTCLayer` class)

In OCR, the alignment between the input (image slices) and the output (characters) is unknown. The **Connectionist Temporal Classification (CTC) Loss** is a special function designed for this exact problem. This project implements it cleanly inside a custom Keras layer.

* **working:**
    * The `CTCLayer` takes the true labels and the model's predictions as input.
    * It calculates the required sequence lengths and feeds them to Keras's backend CTC function, `ctc_batch_cost`.
    * The key feature is **`self.add_loss(loss)`**. This line adds the computed CTC loss directly to the model during training. This allows the model to be compiled with only an optimizer specified, simplifying the training setup.
    * During inference, the layer is transparent and simply passes the model's predictions through.

---

## Getting Started
This project takes a pre-trained Keras model (`prediction_model.h5`) for Optical Character Recognition (OCR) and makes it accessible through a web API.The application is built with **Flask** and served by **Gunicorn**.

The entire application is containerized using **Docker**, making it easy to build, run, and deploy in any environment. The `Dockerfile` is optimized to leverage Docker's layer caching for faster rebuilds by installing dependencies separately from the application code.

### Built With
* Flask 
* Gunicorn
* TensorFlow
* Pillow
* Docker

---


### Prerequisites
You must have **Docker** installed and running on your machine.

### Installation & Execution

1.  **Clone the repository:**
    
2.  **Navigate to the project directory:**
  
3.  **Build the Docker image:** This command reads the `Dockerfile` and installs all the packages from `requirements.txt`.
    ```sh
    docker build -t captcha-solver .
    ```
4.  **Run the Docker container:** This starts the application and exposes it on port 8080.
    ```sh
    docker run -p 8080:8080 captcha-solver
    ```
    The API will now be running and accessible at `http://localhost:8080`.

---

## API Usage

To use the API, send a `POST` request to the `/predict` endpoint with an image file.

### Example using `curl`:
```sh
# Replace 'path/to/your/captcha.png' with an actual image file
curl -X POST -F "image=@path/to/your/captcha.png" http://localhost:8080/predict
