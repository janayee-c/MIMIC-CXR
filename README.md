# 🫁 MIMIC-CXR Multimodal Pneumonia Severity Prediction Pipeline

*A clinical ML pipeline combining radiomics, image embeddings, and natural-language representations for ICU pneumonia severity classification.*
*This project, its design and implementation was spearheaded and produced in collaboration with Data Scientist and Head and Neck Surgeon, Dr. Nicholas Shannon; original and developing project research accredited tohttps://github.com/nbshannon. *

---

## **Problem Statement**

Pneumonia is one of the most common ICU admissions, and severity assessment typically depends on radiologist interpretation of chest-x-rays (CXR) or CT scans. In real clinical workflows, this process may be:

* **Delayed** due to radiologist availability
* **Subjective** with inter-reader variability
* **Inconsistent** across hospitals and severity guidelines

**Goal:**
Build a **multimodal ML pipeline** that predicts *pneumonia severity* using three complementary data sources from the MIMIC-CXR and MIMIC-IV datasets:

* **Medical Images (CXR/CT)** → deep visual embeddings
* **Radiomics Features** → quantitative regional texture descriptors
* **Radiology Reports** → structured RadGraph NER entities + semantic topic embeddings

This system aims to provide **early, automated**, and **interpretable** predictions to support clinicians at the point of admission.

---

## **2. Domain Space Overview**

This project integrates multiple technical and clinical domains:

###**Clinical Imaging**

* CXR & CT interpretation
* Lesion location & density
* Radiomics quantification (GLCM, GLRLM, GLSZM)
* Mask-based ROI extraction

### **Medical NLP**

* Extraction of radiology “Findings” & “Impressions”
* RadGraph-XL entity + relation mapping
* Transformers for clinical semantic embeddings
* BERTopic clustering for global anatomical patterns

### **Predictive Modeling**

* Classical models (XGBoost, SVM) using multimodal features
* Imbalanced ICU labels (severity, ventilation, mortality)
* Feature-level fusion and model-level fusion

### **Critical Care**

* MIMIC-CXR & MIMIC-IV cohort structure
* ICU metadata: LOS, mortality, ventilation duration, clinical severity
* Need for interpretable, auditable predictions

---

## **3.Concepts Utilized **

### **Machine Learning & Computer Vision**

* Radiomics extraction (Pyradiomics)
* CNN visual encoders (ResNet — external code)
* Embedding extraction + latent fusion
* Dimensionality reduction (PCA, UMAP)

### **Medical NLP**

* RadGraph-XL NER + relationship mapping
* Clinical transformers (Bio/ClinicalBERT)
* BERTopic (UMAP + HDBSCAN + c-TF-IDF)
* KeyBERT-Inspired representation modeling
* Topic-probability extraction

### **Multimodal Modeling**

* Fusion of:

  * Radiomics
  * Image embeddings
  * NER entities
  * Topic probabilities
* XGBoost multimodal classifier
* SVM + embedding baselines
* Interpretability via XGBoost feature importance

### **Validation & Clinical Rigor**

* ICU cohort definition
* Class-imbalance handling (weights, SMOTE)
* Statistical cluster validation (Silhouette, DB Index)
* K-Fold CV

### **Data Engineering & Pipeline Design**

* ETL pipelines for radiology text → embeddings
* ETL for radiomics → harmonized tensors
* Repository structure split into:

  * NLP_processing pipeline
  * Prediction Model modules
  * Standalone scripts for training

---

## **4. User Stories**

### **Clinical Users**

* *As an ICU physician*, I want early severity predictions to guide escalation of care.
* *As a radiologist*, I want interpretable outputs that highlight anatomical & textual drivers of classification.

### **Data Science Users**

* *As an ML researcher*, I want a modular multimodal pipeline for experimentation.
* *As a data engineer*, I want clean processing scripts for images, radiomics, and text.

---

## **5. Tech Stack**

### **Languages**

* Python
* SQL

### **Core Libraries**

| Modality          | Libraries                                                             |
| ----------------- | --------------------------------------------------------------------- |
| **Imaging**       | PyTorch, torchvision, OpenCV, pydicom                                 |
| **Radiomics**     | Pyradiomics                                                           |
| **NLP**           | spaCy, RadGraph-XL, HuggingFace Transformers, BERTopic, UMAP, HDBSCAN |
| **ML Models**     | XGBoost, sklearn, PyTorch                                             |
| **Visualization** | Matplotlib, Seaborn                                                   |
| **Data**          | Pandas, NumPy                                                         |

### **Data Sources**

* **MIMIC-CXR**: CXR images + radiology reports
* **MIMIC-IV**: ICU labels (severity, ventilation, LOS, mortality)

---

## **6. Pipeline Architecture**

### **Step 1 — Radiology Report Processing**

* Extract Findings + Impressions
* Clean text (lowercase, punctuation removal, clinical stopwords)
* RadGraph-XL extraction:

  * Anatomy
  * Observations
  * Uncertainty modifiers
  * Relations
* Convert structured entities → embedding vectors
* Optional: concatenate with ClinicalBERT embeddings

---

### **Step 2 — Topic Modeling (BERTopic)**

* Generate 768-dim sentence embeddings
* UMAP (n_neighbors=30) for reduction
* HDBSCAN for density clustering
* c-TF-IDF for cluster representations
* KeyBERT-Inspired → candidate keywords
* MMR (0.7) → diversity-optimized topic terms
* Output: cluster IDs + topic probability vector per report

---

### **Step 3 — Radiomics Extraction**

* Apply bounding box / lung mask
* Extract 50+ radiomics features
* Normalize across patients (z-score)
* Output: radiomics feature vector

---

### **Step 4 — Image Embedding (External Code Not Included)**

* Proprietary NUS research team ResNet encoder
* Extract latent 2048-dim vector
* Stored & loaded as pretrained image features

**Note:** The actual image model code is not included in the repo due to licensing.

---

### **Step 5 — Fusion Model**

Combine:

* Radiomics vector
* RadGraph entity vector
* BERTopic topic-probability vector
* Precomputed image embeddings

Fusion methods tested:

* **Concatenation → XGBoost**
* **Concatenation → SVM baseline**
* (Planned) **Dual-head CNN + Transformer fusion network**

---

### **Step 6 — Prediction**

Outputs:

* Binary or 3-class severity prediction
* Feature importances (XGBoost)
* Topic & entity-level interpretability

---

## **7. Key Performance Results (KPRs)**



### **Technical Milestones**

* Full RadGraph entity extraction across ICU cohort
* BERTopic clustering generated: **23 clinical topics**
* Radiomics pipeline (mask → features) fully integrated
* XGBoost multimodal baseline completed
* Interpretable cluster-level and feature-level insights generated

---

## **8. Future Extensions**

* Add structured EHR variables (SpO₂, labs, vitals)
* Develop unified ResNet + Transformer fusion model
* Evaluate against clinical benchmarks:

  * BioViL
  * CheXzero
  * Vision-Language clinical models
* Deploy clinician-facing web dashboard
* Reliability analysis across:

  * disease subtypes
  * age groups
  * ventilation status

---

## **9. Repository Structure (Actual Repo-Aligned)**

```
MIMIC-CXR/
│
├── NLP_processing/                    # RadGraph, embeddings, topic modeling
│   ├── radgraph_extraction.py
│   ├── bert_embedding_generation.py
│   ├── topic_modeling_bertopic.py
│   └── ...
│
├── Prediction Model/                  # Classical models, evaluation, fusion
│   ├── model_xgboost.ipynb
│   ├── SVM_Y.ipynb
│   └── multimodal_processing.py
│
├── scripts/                           # ETL helpers and utilities
│   ├── clean_text.py
│   ├── extract_features.py
│   └── ...
│
├── Embedding_feature_train_xgb.py     # XGBoost multimodal training script
├── python310_requirements.txt         # Environment for RadGraph pipeline
├── requirements.txt                   # General environment
└── README.md
```

**Important:**
*The full ResNet image-encoding pipeline used in the datathon is proprietary to the NUS research team and is not included.*

---
