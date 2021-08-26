import pandas as pd
import numpy as np
from pathlib import Path
import os

def preprocess_mimic():
    img_dir = Path('/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR-JPG/')
    out_folder = img_dir/'laleh_split'
    out_folder.mkdir(parents = True, exist_ok = True)  

    patients = pd.read_csv(img_dir/'patients.csv.gz')
    admissions = pd.read_csv(img_dir/'admissions.csv.gz').drop_duplicates(subset = ['subject_id']).set_index('subject_id')
    ethnicities = admissions['ethnicity'].to_dict()
    insurance = admissions['insurance'].to_dict()
    patients['ethnicity'] = patients['subject_id'].map(ethnicities)
    patients['insurance'] = patients['subject_id'].map(insurance)
    labels = pd.read_csv(img_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta = pd.read_csv(img_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df = meta.merge(patients, on = 'subject_id').merge(labels, on = ['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins = list(range(0, 100, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['path'] = df.apply(lambda x: os.path.join('files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'), axis = 1)
    
    df.to_csv(out_folder/"preprocessed.csv", index=False)
    
if __name__ == '__main__':
    preprocess_mimic()