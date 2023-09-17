# Exploring-AAD: Audio Anomaly Detection in Machine Sounds using Chroma-based Feature: *CENS*

## Introduction

This project focuses on exploring and comparing various methods for unsupervised audio anomaly detection in machine operation noise, using the [MIMII Dataset](https://zenodo.org/record/3384388). 
Audio anomaly detection is useful in that it ensuring equipment safety and identifying potential faults. 
This README provides an overview of the project, including its objectives, feature space, machine learning models, and key findings.

## Chroma and CENS

Chroma features, as inspired by insights from musical analysis (for more info: see [intro](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_CENS.html) or [longer intro](https://www.researchgate.net/publication/330796993_Chroma_Feature_Extraction), have been employed in this project as a way of capturing distinct information within audio data. These features are based on the twelve pitch spelling attributes (C, C♯, D, ..., B) used in Western music notation. They measure the energy in an audio signal's frame is distributed across these twelve chroma bands.

To obtain chroma energy normalized statistics (CENS), a smoothing window of length ℓ is applied, similar to a Hann window, calculating local weighted averages for each of the twelve chroma components. This process results in sequences of 12-dimensional vectors with nonnegative entries. Subsequently, this sequence is downsampled by a factor of d, and the resulting vectors are normalized with respect to the Euclidean norm (ℓ2-norm).

For instance, consider a rate of 10Hz for a original chroma sequence. With ℓ=41, corresponding to a window size of 4100 milliseconds, and downsampling parameter d=10, the feature rate reduces to 1Hz. The resulting CENS sequences exhibit a higher degree of similarity between performances while still preserving valuable musical information. This property makes CENS features a valuable asset for content-based retrieval tasks, including audio matching and version identification.

To asses how will chroma and CENS features can detect anomalies in machine sounds, this project will compare CENS features proformance in a AE with the more commmanly used [Mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).


## Inspired by Previous (Thesis) Project
The methods employed in this project draw inspiration from my [Master's Thesis in Mathematical Statistics](https://hdl.handle.net/2077/78510), where I reserched audio anomaly detection in cars. The motivation for utilizing chromagram-based features stems from interactions with professionals in the Noise and Vibration Harshness (NVH) during my thesis research. I noticed that, when characterizing abnormal noises in vehicles, professionals often resorted to descriptors like "humming" or "clicking.".
These descriptions led me to explore musical analysis and classification as a viable avenue for audio anomaly detection. The chromagram feature was thus integrated into the project's methodology and gave promising results.

A Abstract of the thesis can be read at the [bottom](#Abstract) of this page. 

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
```console
git clone https://github.com/AHruler/Exploring-AAD.git
cd Exploring-AAD
```


2. Run the Jupyter notebooks provided in the `notebooks` directory to explore the methods and reproduce the results.

3. Experiment with different configurations, models, and datasets to further explore audio anomaly detection techniques.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The MIMII dataset for providing valuable audio data for experimentation.
  
***
## Abstract
Audio anomaly detection in the context of car driving is a crucial task for ensuring vehicle safety and identifying potential faults. This paper aims to investigate and compare different methods for unsupervised audio anomaly detection using a dataset consisting of recorded audio data from fault injections and normal "no fault" driving.

The feature space used in the final modeling consisted of CENS (Chroma energy normalized Statistic), LMFE (Log Mel Frequency Energy), and MFCC (Mel-frequency cepstral coefficients) features. These features exhibit promising capabilities in distinguishing between normal and abnormal classes. Notably, the CENS features, which revealed specific pitch classes, contribute to the distinguishing characteristics of abnormal sounds.

Four machine learning methods were tested to evaluate the performance of different models for audio anomaly detection: Isolation Forest, One-Class Support Vector Machines, Local Outlier Factor, and Long Short-Term Memory Autoencoder. These models are applied to the extracted feature space, and their respective performance was assessed using metrics such as ROC curves, AUC scores, PR curves, and AP scores.

The final results demonstrate that all four models perform well in detecting audio anomalies in cars, where LOF and LSTM-AE achieve the highest AUC scores of 0.98, while OCSVM and IF exhibit AUC scores of 0.97. However, LSTM-AE displays a lower average precision score due to a significant drop in precision beyond a certain reconstruction error threshold, particularly for the normal class. This study demonstrates the effectiveness of Mel frequency and chroma features in modeling for audio anomaly detection in cars and shows great potential for further research and development of effective anomaly detection systems in automotive applications.

***

