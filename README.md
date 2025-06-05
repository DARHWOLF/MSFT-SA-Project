# MSFT-SA-Project

**Project**: Reducing the Carbon Footprint of AI Models on Azure  
**Student Ambassadors**: Raunak Dev, Daksh Gupta  
**Repository**: [DARHWOLF/MSFT-SA-Project](https://github.com/DARHWOLF/MSFT-SA-Project)

---

## 📖 Project Overview

As AI models grow larger and more complex, their training and inference consume vast amounts of energy—contributing to rising carbon emissions. This project demonstrates how to use Azure Machine Learning tools to:

1. Train a **baseline CNN** on a small MNIST subset.
2. Apply **model optimization** (pruning + mixed precision) to create a “sustainable” model that retains accuracy but uses significantly less compute.
3. Track and compare **estimated carbon emissions** (kg CO₂e) for both models using a custom **Azure CarbonFootprintTracker**.
4. Register both models in Azure ML for versioning and side-by-side comparison.

---

## 📂 Repository Structure


- **`cnn.py`**  
  - Builds and trains a simple CNN on a tiny MNIST subset (500 train / 100 val / 100 test).  
  - Demonstrates baseline accuracy, training time, and parameter count.  

- **`Azure_CarbonFootprintTracker.py`**  
  - Defines the `CarbonFootprintTracker` class for Azure ML.  
  - Wraps any training loop and logs:
    - Elapsed training hours
    - Estimated kWh
    - Estimated kg CO₂e (using region-specific emission factors or override)
    - Compute target (CPU/GPU) and Azure region

- **`environment.yml`**  
  - Conda environment specification with required packages:
    ```yaml
    name: sustainable-ai
    channels:
      - defaults
    dependencies:
      - python=3.8
      - pip
      - pip:
          - tensorflow==2.12.0
          - tensorflow-model-optimization
          - matplotlib
          - azureml-core
          - azureml-sdk
    ```

---

## ⚙️ Prerequisites

1. An **Azure Subscription** with permission to create:
   - Azure ML Workspace
   - Compute Instance or Compute Cluster (CPU/GPU)
   - Storage account (default managed by Azure ML)
2. Local machine (Windows/Linux/macOS) with:
   - Python 3.8 (or 3.9) installed
   - Git CLI (to clone the repo)
   - Conda (recommended) or a virtualenv tool
3. Basic familiarity with:
   - Azure Machine Learning Python SDK
   - TensorFlow/Keras
   - ML best practices (training, logging, model versioning)

---

## 🚀 Setup & Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/DARHWOLF/MSFT-SA-Project.git
   cd MSFT-SA-Project
