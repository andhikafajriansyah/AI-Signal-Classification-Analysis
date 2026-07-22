# AI-Driven Wireless Signal Modulation Classification under Severe AWGN Channels

## 1. Background & Problem Statement
In next-generation wireless communications (6G), dynamic spectrum management is highly vulnerable to environmental noise and signal fading. Conventional digital signal processing architectures struggle to maintain real-time classification accuracy under low Signal-to-Noise Ratios (SNR) without substantial computational overhead, a challenge recently addressed by MIT's MAFT-ONN research. This project simulates the performance degradation of machine learning classifiers when exposed to simulated Additive White Gaussian Noise (AWGN).

## 2. Methodology & Experimental Setup
This research implements a comparative framework utilizing a Random Forest Classifier to identify signal modulation types based on synthetic frequency and amplitude feature sets. 

Two distinct environments were evaluated ($N = 1000$ samples):
* **Experiment 1 (Ideal Laboratory Condition):** Zero environmental interference ($\sigma = 0.0$).
* **Experiment 2 (Extreme Weather Condition):** Simulated high-amplitude AWGN interference ($\sigma = 1.2$) added directly to the feature matrix.

## 3. Results & Empirical Analysis
* **Experiment 1 Accuracy:** 100.00%
* **Experiment 2 Accuracy:** 74.50%

### Key Insight:
The addition of Gaussian noise resulted in a critical performance drop of **25.50%**. This degradation mathematically justifies the necessity for advanced hardware-software co-design solutions—such as optical neural networks—to compute deep learning inferences within nanosecond windows before signal digitization occurs.

## 4. How to Run
```bash
pip install numpy scikit-learn
python simulasi_sinyal.py
```

## References
[1] R. Davis III, Z. Chen, R. Hamerly, and D. Englund, "Photonic processor could streamline 6G wireless signal processing," *Science Advances*, 2025.
