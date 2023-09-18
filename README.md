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


## Inspired by Previous (thesis) Project
The methods employed in this project draw inspiration from my [Master's Thesis in Mathematical Statistics](https://hdl.handle.net/2077/78510), where I reserched audio anomaly detection in cars. The motivation for utilizing chromagram-based features stems from interactions with professionals in the Noise and Vibration Harshness (NVH) during my thesis research. I noticed that, when characterizing abnormal noises in vehicles, professionals often resorted to descriptors like "humming" or "clicking.".
These descriptions led me to explore musical analysis and classification as a viable avenue for audio anomaly detection. The chromagram feature was thus integrated into the project's methodology and gave promising results.

***
## Results 


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
```console
git clone https://github.com/AHruler/Exploring-AAD.git
cd Exploring-AAD
```
2. download the Fan and Vlave machine sound datasets at [MIMII Dataset](https://zenodo.org/record/3384388, add them to a ./data map.

4. Run the Jupyter notebooks provided in the `notebooks` directory to explore the methods and reproduce the results.

5. Experiment with different configurations, models, and datasets to further explore audio anomaly detection techniques.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The MIMII dataset for providing valuable audio data for experimentation.
  


