# Exploring-AAD: Audio Anomaly Detection in Machine Sounds using Chroma snf Mel Frequency Feature: *CENS*, *Log Mel Frequncy* and *MFCC*.

## Introduction

This project focuses on exploring and comparing various methods for unsupervised audio anomaly detection in machine operation noise, using the [MIMII Dataset](https://zenodo.org/record/3384388). 
Audio anomaly detection is useful in that it ensuring equipment safety and identifying potential faults. 
This README provides an overview of the project, including its objectives, feature space, machine learning models, and key findings.

## Chroma and CENS

Chroma features, as inspired by insights from musical analysis (for more info: see [intro](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_CENS.html) or [longer intro](https://www.researchgate.net/publication/330796993_Chroma_Feature_Extraction), have been employed in this project as a way of capturing distinct information within audio data. These features are based on the twelve pitch spelling attributes (C, C♯, D, ..., B) used in Western music notation. They measure the energy in an audio signal's frame is distributed across these twelve chroma bands.

To obtain chroma energy normalized statistics (CENS), a smoothing window of length ℓ is applied, similar to a Hann window, calculating local weighted averages for each of the twelve chroma components. This process results in sequences of 12-dimensional vectors with nonnegative entries. Subsequently, this sequence is downsampled by a factor of d, and the resulting vectors are normalized with respect to the Euclidean norm (ℓ2-norm). For instance, consider a rate of 10Hz for a original chroma sequence. With ℓ=41, corresponding to a window size of 4100 milliseconds, and downsampling parameter d=10, the feature rate reduces to 1Hz. The resulting CENS sequences will how lower dimiensions while still retaining important information

To asses how well chroma and CENS features can detect anomalies in machine sounds, this project will compare CENS features preformance with the more commmanly used [Mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). 

Using Unsupervised learning models: Autoencoders (AE), Isolation Forest (IF) and Local outlier detector (LOF). And the abnormal and normal machine sound clips of fans and valves from the [MIMII Dataset](https://zenodo.org/record/3384388).    


## Inspired by Previous (thesis) Project
The methods employed in this project draw inspiration from my [Master's Thesis in Mathematical Statistics](https://hdl.handle.net/2077/78510), where I reserched audio anomaly detection in cars. The motivation for utilizing chromagram-based features stems from interactions with professionals in the Noise and Vibration Harshness (NVH) during my thesis research. I noticed that, when characterizing abnormal noises in vehicles, professionals often resorted to descriptors like "humming" or "clicking.".
These descriptions led me to explore musical analysis and classification as a viable avenue for audio anomaly detection. The chromagram feature was thus integrated into the project's methodology and gave promising results.

***
## Results 
### Fan machine sound anomaly detection - All models and feature combos 
***Sorted by best F1 score of the abnormal class -1***

| Model            | Machine   | Features     |      AUC |   Precision |   Recall |   Abnormal (-1) F1 |
|:-----------------|:----------|:-------------|---------:|------------:|---------:|-------------------:|
| LOF              | fan       | All-means    | 0.99597  |    0.946133 | 0.933876 |           0.889246 |
| LOF              | fan       | mel-means    | 0.994428 |    0.94199  | 0.927887 |           0.880223 |
| LOF              | fan       | chroma-means | 0.982158 |    0.92743  | 0.909679 |           0.851983 |
| Isolation Forest | fan       | All-means    | 0.945388 |    0.891961 | 0.85769  |           0.779182 |
| LOF              | fan       | mel          | 0.947371 |    0.885866 | 0.83517  |           0.756201 |
| AE               | fan       | chroma       | 0.823611 |    0.916898 | 0.861765 |           0.725539 |
| AE               | fan       | mel          | 0.696196 |    0.831592 | 0.884314 |           0.556539 |

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
- tabulate (for printing tables)

You can install these packages using `pip`:
```console
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torchaudio librosa tabulate
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
2. download the Fan and Vlave machine sound datasets at [MIMII Dataset](https://zenodo.org/record/3384388), add them to a ./data map.

4. Run the Jupyter notebooks provided in the `notebooks` directory to explore the methods and reproduce the results.

5. Experiment with different configurations, models, and datasets to further explore audio anomaly detection techniques.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The MIMII dataset for providing valuable audio data for experimentation.
  


