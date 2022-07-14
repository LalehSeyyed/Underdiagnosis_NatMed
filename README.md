# Underdiagnosis_NatMed

## Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations

Laleh Seyyed-Kalantari, Haoran Zhang, Matthew B. A. McDermott, Irene Y. Chen, Marzyeh Ghassemi 

This is the code for the paper **'Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations' ** accepted in 'Nature Medicine', Dec. 2021.

In this paper, we demonstrate underdiagnosis bias in AI-based chest X-ray diagnostic tool, whereby the AI algorithm would inaccurately label an individual with a disease as healthy, potentially delaying access to care. Here we examine algorithmic underdiagnosis in chest X-ray pathology classification across three large and one multi-source chest X-ray datasets. We find that classifiers produced using state-of-the-art computer vision techniques consistently and selectively underdiagnosed under-served patient populations and that the underdiagnosis rate was higher for intersectional under-served subpopulations, e.g., Hispanic females. 

This repository provide the code for the paper. Here for each of the three prominent public chest X-ray datasets MIMIC-CXR (MIMIC), Chest-Xray14 (NIH), CheXpert (CXP), as well as a multi-site aggregation of all those datasets (ALLData) we train convolution neural networks (CNN) on 14, 15, 14 and 8 diagnostic labels, respectively. Among the labels of each dataset there is a 'No Finding' label which indicates the absence of the disease labels. The focus of this study is on the 'No Finding' label. Because false positive outcome on 'No Finding' (underdiagnosis) means falsely claiming the patient is healthy while they are not, which may leads to no clinical treatment when a patient needs it most. 


This code is also a good learning resource for researcher/students interested in training multi-label medical image pathology classifiers.

Citation in Bibtex format:


@article{Underdiagnosis_2021,
  title={Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations},
  author={Seyyed-Kalantari, Laleh and Zhang, Haoran and McDermott, Matthew and Chen, Irene and Marzyeh, Ghassemi},
  journal={Nature Medicine},
  volume={27},
  pages={2176–2182},
  year={2021}
}

----------------------------------------------------------------------------------------------------------------------------
## Dataset access:
All three MIMIC-CXR, CheXpert, and ChestX-ray14 datasets used for this work are public under data use agreements. 

MIMIC-CXR dataset is available at: https://physionet.org/content/mimic-cxr/2.0.0/

CheXpert dataset is available at: https://stanfordmlgroup.github.io/competitions/chexpert/

ChestX-ray14 dataset is available at: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

Access to all three datasets requires user registration and the signing of a data use agreement. Only the MIMIC-CXR dataset requires the completion of an additional credentialing process. After following these procedures, the MIMIC-CXR data is available through PhysioNet (https://physionet.org/). The race/ethnicities and insurance type of the patients are not provided directly with the download of the MIMIC-CXR dataset. However, this data is available through merging the patient IDs in MIMIC-CXR with subject IDs in MIMIC-IV (https://physionet.org/content/mimiciv/0.4/) datasets, using the patient and admissions tables. Access to MIMIC-IV requires a similar procedure as MIMIC-CXR and the same credentialing process is applicable for both datasets. 

----------------------------------------------------------------------------------------------------------------------------
## Reproducing the results:
We have provided the Conda environment (f1.yml) for reproducibility purposes. We are not able to share the trained model and the true label and predicted label CSV files of the test set due to the data-sharing agreement. However, we have provided the patient ID per test splits, random seed, and the code. Then, the true label and predicted label CSV files and trained models can be generated by users who have downloaded the data from the original source following the procedure that is described in the “Data access” session.

You can find 4 folders, 'MIMIC', 'CXP', 'NIH', 'ALLData', where each folder contains the code and the results of the paper on 4 datasets MIMIC-CXR(CXR), CheXpert(CXP), Chest-Xray14(NIH), and ALL dataset. 
There is a 'ReadMe - Steps of runing code.txt' file in MIMIC folder that contain the instruction of runing the code, for training the classifiers and reproducing the results. Similar steps are valid for CXP and ALL datasets. NIH folder has its own 'ReadMe - Steps of runing code.txt' file. Training the classifier will generate the model per dataset per random seed,X, store the trained model in 'resultsX' folder and generate the binay prediction files per model. Then the results of the paper are reproducable by following the rest of steps in 'ReadMe - Steps of runing code.txt' file. 



