# Traffic Analysis using Generative AI

## Description:
This project focuses on analyzing TLS 1.3 network traffic using Generative AI. The code is based on methods referenced from the paper *ET-BERT: A Contextualized Datagram Representation with
Pre-training Transformers for Encrypted Traffic Classification*. The project includes scripts for data preprocessing, model training, and evaluation on TLS 1.3 traffic data.

## Project structure:

- **data/**: Directory containing training, testing datasets, and vocabulary files.
- **models/**: Directory for storing trained models and model parameters.
- **runs/**: Directory containing logs of the training process, including loss and performance charts.
- **uer/**: Directory containing definition of ET-BERT network.

## System requirements:
```
pip install -r requirements.txt
```

## Dataset:

CSTNET-TLS-1.3 dataset, download from [here](https://drive.google.com/drive/folders/1KlZatGoNm-4qu04z0LfrTpZr2oDaHfzr).

## Contents of the scripts:

1. **process_data.py**:
   - Function: Preprocess raw data into a format suitable for model training and testing.

2. **model.py**:
   - Function: Classifier architecture to traffic analysis.

3. **train.py**:
   - Function: Train the classification model for TLS 1.3 traffic based on preprocessed data.
   - Command to run:
     ```bash
     python train.py
     ```

4. **test.py**:
   - Function: Test the trained model on the test dataset and compute performance metrics such as accuracy, precision, recall, and F1-score.
   - Command to run:
     ```bash
     python test.py
     ```
     
## Acknowledgements

This project was inspired by and referenced from the following repository:
- [ET_BERT](https://github.com/linwhitehat/ET-BERT.git)
