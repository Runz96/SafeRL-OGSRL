# INSTRUCTION

---

## 🚀 Project Workflow

### 1. Access the MIMIC-III Dataset
Follow the official instructions to obtain access to the MIMIC-III v1.4 clinical dataset from PhysioNet:

🔗 [MIMIC-III Access Instructions](https://physionet.org/content/mimiciii/1.4/)

---

### 2. Extract Sepsis Treatment Dataset

Clone and follow the instructions from the official GitHub repository:

🔗 [Microsoft/mimic_sepsis Repository](https://github.com/microsoft/mimic_sepsis)

Use the provided SQL scripts and Python code to extract intermediate tables for the sepsis cohort.

---

### 3. Generate Continuous Treatment Variables

Run the following script to derive continuous treatment actions (e.g., IV fluid and vasopressor dosage):

```bash
python sepsis_cohort_continous.py
```

This forms the action space for RL.

---

### 4. Feature Selection and Preprocessing

Open and execute:

```bash
feat_selection.ipynb
```

This notebook selects relevant features and produces the final dataset for model input.

---

### 5. Split Dataset

To divide the data into training, validation, and test sets, run:

```bash
data_split.ipynb
```

---

### 6. Learn State Transition Model

To build the state transition model using k-Nearest Neighbors (kNN):

```bash
transition_model.ipynb
```

This is used for model-based RL algorithms.

---

### 7. Build OOD Guardian

Run the following script to compute the Out-of-Distribution (OOD) guardian based on a Gaussian Kernel method:

```bash
python guardian.py
```

This protects the RL policy from unsafe decisions on out-of-distribution states.

---

## 🧠 Train Reinforcement Learning Models

To train different RL models, run the following Python files:

| Model         | Script Path                   |
|---------------|-------------------------------|
| **CQL**       | `models/ddpg_cql.py`          |
| **CCQL**      | `models/ddpg_cql_ts.py`       |
| **GCQL**      | `models/ddpg_cql_guard.py`    |
| **MB-TRPO**   | `models/trpo.py`              |
| **GMB-TRPO**  | `models/trpo_guard.py`        |
| **MB-CPO**    | `models/cpo.py`               |
| **GMB-CPO**   | `models/cpo_guard.py`         |

---

### 🧪 Evaluate the Trained Policies

Use the following script to perform offline rollout evaluation on the test dataset:

```bash
python eval.py
```

---

## 📁 Project Structure Overview

```text
├── data/                   # Intermediate and final datasets
├── models/                 # Offline RL implementations
├── notebooks/
│   ├── feat_selection.ipynb
│   ├── data_split.ipynb
│   └── transition_model.ipynb
├── sepsis_cohort_continous.py
├── guardian.py
├── eval.py
└── README.md               # (this file)
```

---

## ⚠️ Data Availability Notice

Due to the usage restrictions of the MIMIC-III dataset, we are unable to provide any demonstration data or preprocessed files in this repository. Access to the MIMIC-III dataset must be obtained individually through PhysioNet after completing the required credentialing process.

For more information and to request access, please visit: https://physionet.org/content/mimiciii/1.4/