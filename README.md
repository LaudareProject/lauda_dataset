
# 🎼 Lauda Manuscript Dataset & OMR Learning Pipeline

This repository accompanies the **Lauda Medieval Music Dataset** and the related learning experiments described in our IEEE MLSP 2025 paper:

> 📝 **Experimenting Active and Sequential Learning in a Medieval Music Manuscript**  
> *Sachin Sharma, Federico Simonetta, Michele Flammini*  
> GSSI – Gran Sasso Science Institute, L’Aquila, Italy  
> 📅 Accepted at **IEEE MLSP 2025**  
> 🎨 *Special Session: Applications of AI in Cultural and Artistic Heritage*

---

## 📦 Dataset Access (340 pages)

👉 The full dataset is hosted on **Zenodo**, not GitHub:  
🔗 **[https://zenodo.org/records/15835507](https://zenodo.org/records/15835507)**

### 📁 Archive contents:
- 340 high-resolution digitized manuscript pages (`images/`)
- `annotations/coco_annotations.json` (bounding boxes + class labels)
- Scripts for dataset splits and learning experiments
- All code used in the MLSP 2025 paper

⚠️ **Note:** GitHub does **not** include image data. Please download from Zenodo.

---

## 🧠 Project Overview

This project explores **Optical Music Recognition (OMR)** and active learning on medieval chant manuscripts using:

- ✅ YOLOv8-based object detection
- ✅ Active Learning (uncertainty sampling and sequential training)
- ✅ COCO-format annotations


---

## 📁 Repository Structure

```
              
│──data_split.py            # Splits dataset into train/val
│──Sequential_learning.py   # Sequential baseline training
│──Uncertainty_AL.py        # Active learning with uncertainty sampling
│──coco_annotations.json
├──LICENSE                     # CC BY 4.0 license
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Download the Dataset  
Download and unzip from: [https://zenodo.org/records/15835507](https://zenodo.org/records/15835507)

---

### 2. Prepare Dataset Splits
```bash
python code/data_split.py
```

---

### 3. Train Sequential Baseline
```bash
python code/Sequential_learning.py
```

---

### 4. Run Active Learning Loop
```bash
python code/Uncertainty_AL.py
```

---

## 🔧 Requirements

- Python 3.9+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- NumPy, OpenCV, Pandas, Matplotlib, etc.

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This dataset and code are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
Feel free to use, share, and adapt with proper attribution.

---

## 🙏 Acknowledgment

This work is part of the **LAUDARE ERC Advanced Grant** (Project No. 101054750), funded by the **European Union Horizon Europe Programme (2021–2027)**.

> The views and opinions are those of the authors and do not necessarily reflect those of the European Union or the European Research Council.

---

## 📚 Citation

### MLSP 2025 Paper
```bibtex
@inproceedings{sharma2025lauda,
  title     = {Experimenting Active and Sequential Learning in a Medieval Music Manuscript},
  author    = {Sharma, Sachin and Simonetta, Federico and Flammini, Michele},
  booktitle = {IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year      = {2025},
  institution = {Gran Sasso Science Institute (GSSI), L’Aquila, Italy},
  note      = {Special Session: Applications of AI in Cultural and Artistic Heritage}
}
```

### Zenodo Dataset
```bibtex
@dataset{sharma_2025_15835507,
  author       = {Sharma, Sachin Umesh},
  title        = {Laudare Medieval OMR Dataset and Code},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15835507},
  url          = {https://doi.org/10.5281/zenodo.15835507}
}
```

---

## 📌 Links

- 📦 **Zenodo Archive**: https://zenodo.org/records/15835507  
- 🧠 **LAUDARE Project Info**: [https://laudare.eu/]  
- 🏛️ **GSSI – Gran Sasso Science Institute**: https://www.gssi.it/
