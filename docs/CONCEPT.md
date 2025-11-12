# Kernel Density Estimate-Augmented Conformal Anomaly Detection

## Experimental Setup

#### Considerations

Especially for Exp.3, there might be the necessity to run with different (artificially controlled) anomaly ratios, in order to emphasise the differences between the KDE-augmented and empirical approach.

### Exp.1: Small Datasets – Marginal FDR – Static Batch

#### Rationale
The purpose of this experiment is to show that modeling a reasonably small calibration set via Kernel Density Estimation can increase the performance of a trained anomaly detector.

#### Data

| Data          | # Samples | # Features | # Anomaly | % Anomaly | Category    | Model   |
|---------------|-----------|------------|-----------|-----------|-------------|---------|
| Lymphography  | 148       | 18         | 6         | 4.05      | Healthcare  | COPOD   |
| Ionosphere    | 351       | 33         | 126       | 35.9      | Oryctognosy | IForest |
| Breast Cancer | 683       | 9          | 239       | 35.0      | Chemistry   | GMM     |
| Cardio        | 1831      | 21         | 176       | 9.61      | Healthcare  | ECOD    |
| Musk          | 3062      | 166        | 97        | 3.17      | Chemistry   | GMM     |
| Thyroid       | 3772      | 6          | 507       | 2.47      | Healthcare  | HBOS    |

### Exp.2: Large Datasets – Marginal FDR – Static Batch

#### Rationale
The purpose of this experiment is to show that modeling a sufficiently large calibration set via Kernel Density Estimation converges when compared to the classical (empirical approach) regarding its performance of a trained anomaly detector.

#### Data

| Data        | # Samples | # Features | # Anomaly | % Anomaly | Category     | Model |
|-------------|-----------|------------|-----------|-----------|--------------|-------|
| Fraud       | 284807    | 29         | 492       | 0.17      | Finance      | INNE  |
| Shuttle     | 49097     | 9          | 3511      | 7.15      | Astrophysics | HBOS  |
| Mammography | 11183     | 6          | 260       | 2.32      | Healthcare   | ECOD  |

### Exp.3: Large Datasets – Marginal FDR – Batch Streaming

#### Rationale
Extension of Exp.1. The purpose of this experiment is to show that modeling a reasonably small calibration set via Kernel Density Estimation in context of a streaming-batch setup with respective (online? and) batch FDR control procedures can reliably control the FDR over all batches without loosing statistical power.

#### Data

| Data        | # Samples | # Features | # Anomaly | % Anomaly | Category     | Model |
|-------------|-----------|------------|-----------|-----------|--------------|-------|
| Fraud       | 284807    | 29         | 492       | 0.17      | Finance      | INNE  |
| Shuttle     | 49097     | 9          | 3511      | 7.15      | Astrophysics | HBOS  |
| Mammography | 11183     | 6          | 260       | 2.32      | Healthcare   | ECOD  |

#### Setup

Batch Number: 100
Batch Size: 100
Anomaly Proportion: Empirical (Probabilistic)
Batch FDR Procedure: Batch-BH