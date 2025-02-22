# Malware classification using Generative AI

## Description:
This project focuses on Deepfake Counter-measurements using Generative AI. The project includes scripts for data preprocessing, model training, and evaluation on TLS 1.3 traffic data.

## Project structure:

- **data/**: Directory containing training, testing datasets, and vocabulary files.
- **checkpoints/**: Directory for storing trained models.
- **runs/**: Directory containing logs of the training process, including loss and performance charts.

## System requirements:
```
pip install -r requirements.txt
```

## Dataset:

DFFD dataset, download from [here](https://www.cse.msu.edu/computervision/dffd_dataset/).

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
- [OC-FakeDect](https://github.com/jumpycat/OC-FakeDect.git) 
