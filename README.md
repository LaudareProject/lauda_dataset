
Lauda Manuscript Codebase & OMR Learning Pipeline

This repository accompanies the learning experiments described in our IEEE MLSP 2025 paper:

📝 Experimenting Active and Sequential Learning in a Medieval Music Manuscript  
Sachin Sharma, Federico Simonetta, Michele Flammini  
GSSI – Gran Sasso Science Institute, L’Aquila, Italy  
📅 Accepted at IEEE MLSP 2025  
🎨 Special Session: Applications of AI in Cultural and Artistic Heritage

📁 What This Repository Contains  
This repository provides Python code used in the MLSP 2025 paper, including:

✅ YOLOv8-based object detection  
✅ Active Learning (uncertainty sampling and sequential training)  
✅ COCO-format preprocessing and splitting tools  

⚠️ **Note:** The dataset used in these experiments is **not yet publicly released**. It will be made available after the conclusion of the LAUDARE ERC Project.

📁 Repository Structure

│──data_split.py            # Splits dataset into train/val  
│──Sequential_learning.py   # Sequential baseline training  
│──Uncertainty_AL.py        # Active learning with uncertainty sampling  
│──coco_annotations.json    # Sample structure only (no image data)  
├──LICENSE                  # CC BY 4.0 license  
└──README.md                # This file

🚀 Quick Start

1. Clone this repository  
2. Install dependencies  
```bash
pip install -r requirements.txt
3. Run the scripts as needed for training or evaluation.

🔧 Requirements

Python 3.9+

Ultralytics YOLOv8

NumPy, OpenCV, Pandas, Matplotlib, etc.

📜 License
This code is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
Feel free to use, share, and adapt it with proper attribution.

🙏 Acknowledgment
This work is part of the LAUDARE ERC Advanced Grant (Project No. 101054750), funded by the European Union Horizon Europe Programme (2021–2027).

The views and opinions expressed are those of the authors and do not necessarily reflect those of the European Union or the European Research Council.

📚 Citation

MLSP 2025 Paper
@inproceedings{sharma2025lauda,
  title     = {Experimenting Active and Sequential Learning in a Medieval Music Manuscript},
  author    = {Sharma, Sachin and Simonetta, Federico and Flammini, Michele},
  booktitle = {IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year      = {2025},
  institution = {Gran Sasso Science Institute (GSSI), L’Aquila, Italy},
  note      = {Special Session: Applications of AI in Cultural and Artistic Heritage}
}

📦 Dataset Access  
The full dataset (340 high-resolution manuscript pages with COCO-format annotations) is **not yet released**.  
We will upload it to Zenodo after the LAUDARE project ends.

## 📌 Links

- 🧠 **LAUDARE Project Info**: [https://laudare.eu/]  
- 🏛️ **GSSI – Gran Sasso Science Institute**: https://www.gssi.it/
