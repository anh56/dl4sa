# Deep Learning for Code-centric Software Vulnerability Assessment

This repository contains the data and scripts used in the paper "Automated Code-centric Software Vulnerability Assessment: How Far Are We? An Empirical Study in C/C++".

## Structure
The repository is organized as follows:
- `data`: Contains the data used in the paper.
- `ml`: Contains the code for the machine models.
- `dl`: Contains the code for the deep learning models, with two versions for each model type (graph and non-graph):
  - `mutliclass`: the original models.
  - `multitask`: modified models that support multitask.


## Requirements
The dependencies required to run the code can be found in the `requirements.txt` file for each model type.


## Data
Due to large size, the data is not included in the repository.
The data can be downloaded from the following link: [data](https://drive.google.com/drive/folders/17zrM4V9b8eOuc9-2SC90hF8I72siTpyc?usp=sharing)
and put into the `data` folder.


## Models
Each model is provided in a separate folder, and contains a `README.md` file that explains how to run the model.


## Citation
```
@inproceedings{nguyen2024automated,
  title={Automated Code-centric Software Vulnerability Assessment: How Far Are We? An Empirical Study in C/C++},
  author={Nguyen, Anh The and Le, Triet Huynh Minh and Babar, M Ali},
  booktitle={Proceedings of the 18th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement},
  pages={72--83},
  year={2024}
}
```
