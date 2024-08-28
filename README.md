

# College Menfess Tweet Emotion Project

This project is a Flask web application that predicts the emotions or sentiment of tweets sent to a college menfess account. The application uses machine learning models to analyze and classify the tweets.

## Features

- **Emotion Prediction**: Predicts the emotion of a tweet using a custom prediction function.
- **Support Vector Machine (SVM) Prediction**: Provides an additional prediction using a Support Vector Machine (SVM) model.

## Project Structure

```
.
├── main.py              # The main Flask application script
├── script/
│   └── func.py          # Contains the function `predictTweet` used for predicting the tweet emotion
├── kmeans.py            # Contains the functions `predict_label` and `predict_label_svm` for SVM prediction
├── templates/
│   └── ...              # Directory for HTML templates (if any)
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- NumPy
- Pickle (for model loading)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/akhmadramadani/collegemenfess-emotion.git
    cd collegemenfess-emotion
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the Flask application:

    ```bash
    python main.py
    ```

2. The application will start on `http://127.0.0.1:5000/` by default.

### API Endpoints

- **`GET /`**: Returns a simple greeting message.
- **`POST /predict`**: Predicts the emotion of the provided tweet. Requires the following form data:
  - `tweet`: The tweet text to be analyzed.
  
  Example request:

    ```bash
    curl -X POST -F 'tweet=Your tweet text here' http://127.0.0.1:5000/predict
    ```

  Example response:

    ```json
    {
      "prediction": "positive",
      "predict_using_svm": "neutral"
    }
    ```

- **`POST /predict_svm`**: Predicts the emotion using the SVM model only. Requires the following form data:
  - `tweet`: The tweet text to be analyzed.
  
  Example request:

    ```bash
    curl -X POST -F 'tweet=Your tweet text here' http://127.0.0.1:5000/predict_svm
    ```

  Example response:

    ```json
    {
      "prediction": "neutral"
    }
    ```

### Environment Variables

- **`PORT`**: Set the port for the Flask application. Default is `5000`.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contributing

Contributions are welcome! Please submit a pull request or open an issue for any feature requests or bug reports.

### Authors

- **Akhmad Ramadani** - [Github](https://github.com/akhmadramadani)

```

This `README.md` file provides a clear overview of the project, instructions on how to set it up, and details about the API endpoints. Be sure to replace placeholders like `Your Name` and `Your GitHub` with the appropriate details for your project.
```
