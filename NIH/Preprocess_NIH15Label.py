import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

diseases_abbr = {'Atelectasis': 'At',
                'Cardiomegaly': 'Cd',
                'Effusion': 'Ef',
                'Infiltration': 'In',
                'Mass': 'M',
                'Nodule': 'N',
                'Pneumonia': 'Pa',
                'Pneumothorax': 'Px',
                'Consolidation': 'Co',
                'Edema': 'Ed',
                'Emphysema': 'Em',
                'Fibrosis': 'Fb',
                'Pleural_Thickening': 'PT',
                'Hernia': 'H',
                'No Finding':'NF'
                }

ylabel = {'Age': 'AGE',
        'Sex': 'SEX',
        'M': 'MALE',
        'F': 'FEMALE'
        }


def preprocess_NIH(split):
    split['Patient Age'] = np.where(split['Patient Age'].between(0,19), 19, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(20,39), 39, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(40,59), 59, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(60,79), 79, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age']>=80, 81, split['Patient Age'])
    
    copy_sunbjectid = split['Patient ID'] 
    split.drop(columns = ['Patient ID'])
    
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
    
    split['subject_id'] = copy_sunbjectid
    split['Sex'] = split['Patient Gender'] 
    split['Age'] = split['Patient Age']
    split['path'] = split['Image Index']
    split = split.drop(columns=["Patient Gender", 'Patient Age', 'Image Index'])
    
    return split

