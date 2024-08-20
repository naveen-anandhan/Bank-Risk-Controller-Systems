# Bank Risk Controller Systems

Welcome to the Bank Risk Controller Systems project! This application provides insights and predictions related to loan defaulters and non-defaulters using advanced machine learning models and interactive visualizations. It also features a chatbot that can answer questions based on the content of uploaded PDFs.

## Project Overview

This project is designed to help users understand their risk profile by predicting whether an individual will default on a loan based on various features. It also includes exploratory data analysis (EDA) and a chatbot feature to interact with PDF documents.

## Features

1. **Model Predictions**:
   - **Single Prediction**: Input loan and personal details to predict if the user is a defaulter or non-defaulter.
   - **Multiple Predictions**: Upload a CSV file containing multiple user records to get predictions for all users.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize distributions of gender, target (defaulter/non-defaulter), contract types, and loan types.
   - Includes pie charts and bar charts to illustrate the data.

3. **Chatbot with PDF Interaction**:
   - Upload a PDF and interact with it through a chatbot interface.
   - The chatbot uses the content of the PDF to provide relevant responses to user queries.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `pickle`
  - `streamlit`
  - `matplotlib`
  - `seaborn`
  - `pdfplumber`
  - `scikit-learn`
  - `langchain`
  - `langchain_community`
  - `langchain_huggingface`
  - `python-dotenv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/naveen-anandhan/Bank-Risk-Controller-Systems.git

   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file and add your Hugging Face API token:
     ```env
     HF_TOKEN=your_huggingface_api_token
     ```

## Usage

1. **Run the Streamlit Application**:
   ```bash
   streamlit run web.py
   ```

2. **Navigate through the application**:
   - **Data**: View sample datasets and model performance metrics.
   - **EDA**: Explore various data visualizations.
   - **Model**: Input user details to get loan defaulter predictions.
   - **Multiple Prediction**: Upload a CSV to get predictions for multiple users.
   - **Chat with PDF**: Upload a PDF and interact with it through the chatbot.
  
## sample images
![data_page](https://github.com/user-attachments/assets/5d149092-1806-4f67-b3f5-070249871cf9)
![EDA_page](https://github.com/user-attachments/assets/a50f4363-0d22-4f75-a5cb-2d3930064268)
![model_page](https://github.com/user-attachments/assets/576a53c9-ff9b-4a94-935c-b483416a9f54)
![multiple_page](https://github.com/user-attachments/assets/c031fa8d-aadc-4e10-980f-ec2657a631db)
![bot3](https://github.com/user-attachments/assets/7e676471-9d49-4a0b-a079-b400f2f1dd37)

## Files

- `app.py`: Main Streamlit application script.
- `eda_data.csv`: Sample data used for exploratory data analysis.
- `model_data.csv`: Data used for displaying sample records.
- `ET_Classifier_model.pkl`: Pickle file for the Extra Trees Classifier model.
- `label_encoders.pkl`: Pickle file for label encoders used in the model.
- `.env`: File to store environment variables (e.g., Hugging Face API token).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Created by [Naveen Anandhan](https://www.linkedin.com/in/naveen-anandhan-8b03b62a5/?trk=public-profile-join-page) - feel free to contact me!
