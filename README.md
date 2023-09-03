# Exploring-AAD: Audio Anomaly Detection in Machine Sounds

## Introduction

This project focuses on exploring and comparing various methods for unsupervised audio anomaly detection in machine operation noise, with a specific emphasis on the MIMII and ToyADMOS2 datasets. 
Audio anomaly detection is useful in that it ensuring equipment safety and identifying potential faults. 
This README provides an overview of the project, including its objectives, feature space, applied machine learning models, and key findings.

## Previous (Thesis) Project
The methods in this project are based on my Master Thesis in Mathemtical Statitics. A Abstract of the Thesis can be read below. A link to the full project will be linked ones it is published.

### Abstract
Audio anomaly detection in the context of car driving is a crucial task for ensuring vehicle safety and identifying potential faults. This paper aims to investigate and compare different methods for unsupervised audio anomaly detection using a dataset consisting of recorded audio data from fault injections and normal "no fault" driving.

The feature space used in the final modeling consisted of CENS (Chroma energy normalized Statistic), LMFE (Log Mel Frequency Energy), and MFCC (Mel-frequency cepstral coefficients) features. These features exhibit promising capabilities in distinguishing between normal and abnormal classes. Notably, the CENS features, which revealed specific pitch classes, contribute to the distinguishing characteristics of abnormal sounds.

Four machine learning methods were tested to evaluate the performance of different models for audio anomaly detection: Isolation Forest, One-Class Support Vector Machines, Local Outlier Factor, and Long Short-Term Memory Autoencoder. These models are applied to the extracted feature space, and their respective performance was assessed using metrics such as ROC curves, AUC scores, PR curves, and AP scores.

The final results demonstrate that all four models perform well in detecting audio anomalies in cars, where LOF and LSTM-AE achieve the highest AUC scores of 0.98, while OCSVM and IF exhibit AUC scores of 0.97. However, LSTM-AE displays a lower average precision score due to a significant drop in precision beyond a certain reconstruction error threshold, particularly for the normal class. This study demonstrates the effectiveness of Mel frequency and chroma features in modeling for audio anomaly detection in cars and shows great potential for further research and development of effective anomaly detection systems in automotive applications.

***

## Package Requirements

To run the code and reproduce the results, ensure you have the following Python packages installed:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- torchaudio (for audio processing and audio feature extraction)
- librosa (for audio feature extraction)
- tqdm 

You can install these packages using `pip`:
```console
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torchaudio librosa tqdm
```
Using conda you can replicate the environment in environment.yml:
```console
conda env create -n ENVNAME --file environment.yml
```

## Usage

1. Clone this repository to your local machine:

git clone https://github.com/AHruler/Exploring-AAD.git
cd Exploring-AAD

2. Run the Jupyter notebooks provided in the `notebooks` directory to explore the methods and reproduce the results.

3. Experiment with different configurations, models, and datasets to further explore audio anomaly detection techniques.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The MIMII and ToyADMOS2 datasets for providing valuable audio data for experimentation.

