import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def simulate_modulation_classification(noise_level):
    """
    Simulate BPSK and QPSK modulation signals under AWGN channel
    conditions and evaluate classification accuracy using a
    Random Forest classifier.

    Parameters
    ----------
    noise_level : float
        Standard deviation of Additive White Gaussian Noise (AWGN).

    Returns
    -------
    float
        Classification accuracy.
    """

    # Reproducibility
    np.random.seed(42)

    n_samples = 1000

    # ------------------------------------------------------------------
    # Generate synthetic BPSK signal samples
    # ------------------------------------------------------------------
    bpsk_feature1 = np.random.normal(
        loc=1.0,
        scale=0.2,
        size=(n_samples // 2, 1)
    )

    bpsk_feature2 = np.random.normal(
        loc=2.0,
        scale=0.3,
        size=(n_samples // 2, 1)
    )

    # ------------------------------------------------------------------
    # Generate synthetic QPSK signal samples
    # ------------------------------------------------------------------
    qpsk_feature1 = np.random.normal(
        loc=2.5,
        scale=0.3,
        size=(n_samples // 2, 1)
    )

    qpsk_feature2 = np.random.normal(
        loc=3.5,
        scale=0.4,
        size=(n_samples // 2, 1)
    )

    # Combine all samples
    X_clean = np.vstack((
        np.hstack((bpsk_feature1, bpsk_feature2)),
        np.hstack((qpsk_feature1, qpsk_feature2))
    ))

    # Labels
    y = np.concatenate((
        np.zeros(n_samples // 2),
        np.ones(n_samples // 2)
    ))

    # ------------------------------------------------------------------
    # Simulate Additive White Gaussian Noise (AWGN)
    # ------------------------------------------------------------------
    noise = np.random.normal(
        loc=0.0,
        scale=noise_level,
        size=X_clean.shape
    )

    X_noisy = X_clean + noise

    # ------------------------------------------------------------------
    # Split dataset
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy,
        y,
        test_size=0.20,
        random_state=42
    )

    # ------------------------------------------------------------------
    # Train Random Forest classifier
    # ------------------------------------------------------------------
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate model
    # ------------------------------------------------------------------
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# ==============================================================
# Experiment
# ==============================================================

noise_levels = [0.0, 0.3, 0.6, 0.9, 1.2]

print("=" * 60)
print("AI-Based Wireless Signal Modulation Classification")
print("=" * 60)

for noise in noise_levels:

    accuracy = simulate_modulation_classification(noise)

    print(
        f"Noise Level: {noise:.1f} | "
        f"Classification Accuracy: {accuracy * 100:.2f}%"
    )
