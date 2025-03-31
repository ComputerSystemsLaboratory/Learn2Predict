
# Learn2Predict

## Overview

Welcome to the *Learn2Predict (L2P)* repository! This project introduces a transfer learning-based performance prediction model specifically designed for the UPMEM compute-near-memory (CNM) system. The primary goal of L2P is to enhance the accuracy of execution time predictions for various applications, facilitating optimization of CNM systems without requiring extensive training data.

CNM system tackles the challenges posed by the von Neumann bottleneck in conventional CPU/GPU systems, which arise from excessive data movement between memory and processing units. By positioning computation units closer to memory, CNM architectures like UPMEM unlock significant computational efficiency, particularly for memory-bound workloads. The L2P model leverages transfer learning techniques to predict execution performance across diverse application domains, making it an essential tool for those working with CNM systems.

---

## Key Features

- **Transfer Learning for Performance Prediction:** L2P leverages knowledge from previously analyzed workloads to predict performance for new applications, significantly reducing the need for large training datasets (less than 1% required).
- **Application-Agnostic Design:** Applicable across a wide range of benchmarks and application domains, showcasing robust generalization performance.
- **High Prediction Accuracy:** Achieves a mean absolute percentage error (MAPE) within ±17%, validated through comprehensive experimental evaluations.
- **Feature Extraction Framework:** Extracts computation and memory access patterns tailored to CNM systems, ensuring effective transfer learning and accurate performance prediction.
- **Open Source:** The model is open-sourced for accessibility, enabling other researchers and practitioners to test and apply it to their use cases.

---

## Motivation

Traditional CPU/GPU systems struggle with data-intensive tasks due to the von Neumann bottleneck, where data movement between the processing unit and memory severely limits performance. CNM architectures like UPMEM offer a promising solution by embedding compute units directly within memory. However, realizing the full potential of CNM systems requires robust models to predict application performance accurately.

L2P addresses these challenges by:

- Minimizing training effort while retaining accuracy.
- Adapting to different application domains with ease.
- Taking advantage of structural changes in applications for improved prediction performance.

---

## Results

The L2P model has been tested thoroughly on diverse benchmarks, including Prim and PIM-ML benchmarks suites. Key results include:

- Consistent predictive performance across structurally varied applications.
- Ability to leverage structural diversity (e.g., different programming languages, compilers, or developer styles) to improve prediction accuracy.
- Validation with a MAPE of ±17% across benchmarks.

---

## Repository Contents

This repository includes:

1. **Source Code:** Implementation of the L2P model and its related modules.
2. **Example Benchmarks:** A suite of benchmarks used to evaluate the model, such as GEMV, HST, and RED.
4. **Scripts:** Utilities for feature extraction, and data generation.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python libraries (see `requirements.txt`)
- UPMEM simulator or access to UPMEM hardware

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourOrganization/Learn2Predict.git
   cd Learn2Predict
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1) Generate data
```bash
cd Learn2Predict/bash
./upmem_data_generator.sh <options>
```

#### 2) Generate embeddings
```bash
cd Learn2Predict/bash
./x86_embeddings_generator.sh <options>
```

#### 3) Update the embeddings
```bash
cd Learn2Predict/py
python update_x86_embeddings_with_upmem_data.py <options>
```

#### 4) Generate datasets
```bash
cd Learn2Predict/py
python dataset_generator.py <options>
```

#### 5) Train the model
```bash
cd Learn2Predict/py
python training.py <options>
```

#### 6) Predict performance
```bash
cd Learn2Predict/py
python inference.py <options>
```

---

## Contributing

Contributions are welcome! If you have ideas for features, enhancements, or bug fixes, feel free to open an issue or submit a pull request.

---

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for details.


