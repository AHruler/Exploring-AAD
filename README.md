# Exploring-AAD: Audio Anomaly Detection in Machine Sounds using Chroma snf Mel Frequency Feature: *CENS*, *Log Mel Frequncy* and *MFCC*.

This project focuses on exploring and comparing various methods for unsupervised audio anomaly detection in machine operation noise, using the [MIMII Dataset](https://zenodo.org/record/3384388). 
Audio anomaly detection is useful in ensuring equipment safety and identifying potential faults. 

This README provides an overview of the project, including its objectives, feature space, machine learning models, and key results.

# Contents
- [Previous Project](#Previous) 💡
- [Approach](#Approach) 🧐
    - [Chroma](#Chroma) 🎼
    - [ML Models](#ML) 🛠
- [Final Result](#Results) 😁
- [Usage](#Usage) 🔆
    - [Package Requirements](#req) ❕
- [License](#License) 📄
- [Acknowledgments](#Acknowledgments) 📣 
  
  
## Inspired by Previous (thesis) Project 
<a name="Previous"></a>
The methods employed in this project draw inspiration from my [Master's Thesis in Mathematical Statistics](https://hdl.handle.net/2077/78510), where I researched audio anomaly detection in cars. The motivation for utilizing chromagram-based features stems from interactions with professionals in the Noise and Vibration Harshness (NVH) during my thesis research. I noticed that, when characterizing abnormal noises in vehicles, professionals often resorted to descriptors like "humming" or "clicking.".
These descriptions led me to explore musical analysis and classification as a viable avenue for audio anomaly detection. The chromagram feature was thus integrated into the project's methodology and gave promising results.

# Approach

## Chroma and CENS
<a name="Chroma"></a>
Chroma features, as inspired by insights from musical analysis (for more info: see [intro](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_CENS.html) or [longer intro](https://www.researchgate.net/publication/330796993_Chroma_Feature_Extraction), have been employed in this project as a way of capturing distinct information within audio data. These features are based on the twelve pitch spelling attributes (C, C♯, D, ..., B) used in Western music notation. They measure the energy in an audio signal's frame is distributed across these twelve chroma bands.

To obtain chroma energy normalized statistics (CENS), a smoothing window of length ℓ is applied, similar to a Hann window, calculating local weighted averages for each of the twelve chroma components. This process results in sequences of 12-dimensional vectors with nonnegative entries. Subsequently, this sequence is downsampled by a factor of d, and the resulting vectors are normalized with respect to the Euclidean norm (ℓ2-norm). For instance, consider a rate of 10Hz for a original chroma sequence. With ℓ=41, corresponding to a window size of 4100 milliseconds, and downsampling parameter d=10, the feature rate reduces to 1Hz. The resulting CENS sequences will how lower dimiensions while still retaining important information

## Models
<a name="ML"></a>
To asses how well chroma and CENS features can detect anomalies in machine sounds, this project will compare CENS features preformance with the more commmanly used [Mel spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). Using Unsupervised learning models: Autoencoders (AE), Isolation Forest (IF) and Local outlier detector (LOF). And the abnormal and normal machine sound clips of fans and valves from the [MIMII Dataset](https://zenodo.org/record/3384388). Below are short intros to the models. 

- **Autoencoders**:
  - A type of neural network architecture.
  - Effective for handling high-dimensional audio features.
  - Works by reconstructing input data while capturing abnormal patterns.
  - Utilizes reconstruction error as a measure of deviation from the norm.

- **Isolation Forest (IF)**:
  - An unsupervised ensemble learning algorithm.
  - Assumes that anomalous data points are rare and different.
  - Divides data into subspaces to isolate anomalies.
  - Provides an anomaly score based on the number of splits needed to isolate a data point.

- **Local Outlier Factor (LoF)**:
  - Detects anomalies by measuring local density deviations.
  - Compares data points to their neighbors.
  - Not commonly used for audio data but has shown strong performance on high-dimensional spectral data, as demonstrated by Yu et al. in [this study](https://ieeexplore.ieee.org/document/9104925).

***
# Results 

Below are som key reults from using Mel frequeny and chroma features on the fan and valve datasets: which show interestig insights into the performance of various models and feature combinations for machine sound anomaly detection:

- **Mean Features**: Across all models, using mean features performed exceptionally well, with LOF utilizing all features means and just mel feature means achieving near-perfect classification. This demonstrates the effectiveness of mean-based statistics in capturing abnormal patterns.

- **Autoencoder (AE) Improvement with Chroma**: In the case of fan machine sound anomaly detection, it's interesting to observe a substantial improvement in the F1 score of the AE model when using chroma features compared to mel spectrogram features. This improvement may be due to the chromagrams of abnormal and normal sound clips looking more diffent then the mel spectrogram. This [bar plot](#means) of the means, show significant differences in mean pitch between the classes for G and G#. To see the encoding and decoding of spectrograms themselves: *Check out the notebooks!*

- **Valve Machine Sound Anomaly Detection**: The valve machine sound detection results display comparatively lower metrics, indicating the complexity of anomaly detection task, as the feature space + model that may work great in one context does not necessarily translate to all sound anomoly detection tasks. LOF with various feature combinations showcases better performance, but the overall metrics for valve machine sound detection remain modest.

- **LOF Dominance (!!)**: Across both machine types and feature combinations, the LOF model consistently outperforms other models in terms of F1 scores, with LOF using chroma-means for fan sound achieving an abnormal F1 score of 0.851983 aswell as a AUc score near very close to 1. This highlights the usefullness of LOF for audio anomaly detection.

***In summary***, these results show the usefullness of mean-based features, the advantage of chroma features for AE models in specific scenarios, and the dominance of LOF as a robust choice for audio anomaly detection, achieving near-perfect classification even with mel features alone. While the project aimed to explore the utility of chroma in machine anomaly detection, it suggests that chroma features hold substantial promise when coupled with certain models like AE, while LOF consistently delivers excellent results.

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

### Valve machine sound anomaly detection - All models and feature combos 
***Sorted by best F1 score of the abnormal class -1***

| Model     | Machine   | Features     |      AUC |   Precision |   Recall |   Abnormal (-1) F1 |
|:----------|:----------|:-------------|---------:|------------:|---------:|-------------------:|
| LOF       | valve     | All-means    | 0.665219 |    0.859019 | 0.414323 |           0.258385 |
| LOF       | valve     | mel-means    | 0.666815 |    0.859777 | 0.402551 |           0.25641  |
| LOF + PCA | valve     | All-means    | 0.641049 |    0.851031 | 0.362982 |           0.243201 |
| LOF + PCA | valve     | mel-means    | 0.618747 |    0.839748 | 0.339111 |           0.233017 |
| LOF       | valve     | chroma-means | 0.575024 |    0.83062  | 0.292675 |           0.223339 |
| AE        | valve     | chroma       | 0.497541 |    0.882027 | 0.883661 |           0.112202 |
| AE        | valve     | mel          | 0.494097 |    0.881232 | 0.8907   |           0.101597 |


<a name="means"></a> ***Mean CENS plot***
![(#fig-means) CENS Means by Machine type](figs/mean_cens.png?raw=true "Title")

***
# Usage

1. Clone this repository to your local machine:
```console
git clone https://github.com/AHruler/Exploring-AAD.git
cd Exploring-AAD
```
2. download the Fan and Vlave machine sound datasets at [MIMII Dataset](https://zenodo.org/record/3384388), add them to a ./data map.

4. Run the Jupyter notebooks provided in the `notebooks` directory to explore the methods and reproduce the results.

5. Experiment with different configurations, models, and datasets to further explore audio anomaly detection techniques.
   
## Package Requirements
<a name="req"></a>
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

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The MIMII dataset for providing valuable audio data for experimentation.
  


