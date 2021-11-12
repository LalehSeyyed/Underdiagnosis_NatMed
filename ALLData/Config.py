import pandas as pd
import os
import numpy as np


#----------------------------- q
path_image_MIMIC = "/PATH TO MIMIC DATASET IMAGES IN YOUR SERVER/split/MIMIC-CXR/"


train_df_path_MIMIC ="/PATH TO DATASET CSV FILES IN YOUR SERVER/split/new_train.csv"
test_df_path_MIMIC ="/PATH TO DATASET CSV FILES IN YOUR SERVER/split/new_test.csv"
val_df_path_MIMIC = "/PATH TO DATASET CSV FILES IN YOUR SERVER/split/new_valid.csv"

path_image_NIH = "/PATH TO NIH DATASET IMAGES IN YOUR SERVER/images/"

train_df_path_NIH ="/PATH TO DATASET CSV FILES IN YOUR SERVER/split/train.csv"
test_df_path_NIH = "/PATH TO DATASET CSV FILES IN YOUR SERVER/split/test.csv"
val_df_path_NIH = "/PATH TO DATASET CSV FILES IN YOUR SERVER/split/valid.csv"

path_image_CXP = "/PATH TO CXP DATASET IMAGES IN YOUR SERVER/CheXpert/"


train_df_path_CXP ="/PATH TO CXP DATASET CSV FILES IN YOUR SERVER/split/new_train.csv"
test_df_path_CXP ="/PATH TO CXP DATASET CSV FILES IN YOUR SERVER/split/new_test.csv"
val_df_path_CXP = "/PATH TO CXP DATASET CSV FILES IN YOUR SERVER/split/new_valid.csv"

val_df_NIH = pd.read_csv(val_df_path_NIH)
val_df_CXP = pd.read_csv(val_df_path_CXP)
val_df_MIMIC = pd.read_csv(val_df_path_MIMIC)

test_df_NIH = pd.read_csv(test_df_path_NIH)
test_df_CXP = pd.read_csv(test_df_path_CXP)
test_df_MIMIC = pd.read_csv(test_df_path_MIMIC)

train_df_NIH = pd.read_csv(train_df_path_NIH)
train_df_CXP = pd.read_csv(train_df_path_CXP)
train_df_MIMIC = pd.read_csv(train_df_path_MIMIC)

def preprocess_MIMIC(split):
    
    details = pd.read_csv("/PATH TO MIMIC METADATA/mimic-cxr-metadata-detail.csv")
    details = details.drop(columns=['dicom_id', 'study_id', 'religion', 'race', 'insurance', 'marital_status', 'gender'])
    details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
    df = pd.merge(split, details)
    
    copy_sunbjectid = df['subject_id'] 
    df.drop(columns = ['subject_id'])
    
    df = df.replace(
            [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
             'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
             '>=90'],
            [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
             'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    
    df['subject_id'] = copy_sunbjectid
    df['Age'] = df["age_decile"]
    df['Sex'] = df["gender"]
    df = df.drop(columns=["age_decile", 'gender'])

    
    return df

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
    split = split.drop(columns=["Patient Gender", 'Patient Age'])

    return split


def preprocess_CXP(split):
    split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
    split['Age'] = np.where(split['Age']>=80, 81, split['Age'])
    
    copy_sunbjectid = split['subject_id'] 
    split.drop(columns = ['subject_id'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
    
    split['subject_id'] = copy_sunbjectid
    split['Sex'] = np.where(split['Sex']=='Female', 'F', split['Sex'])
    split['Sex'] = np.where(split['Sex']=='Male', 'M', split['Sex'])
    
    return split



test_df_MIMIC = preprocess_MIMIC(test_df_MIMIC)
test_df_CXP = preprocess_CXP(test_df_CXP)
test_df_NIH = preprocess_NIH(test_df_NIH)

train_df_MIMIC = preprocess_MIMIC(train_df_MIMIC)
train_df_CXP = preprocess_CXP(train_df_CXP)
train_df_NIH = preprocess_NIH(train_df_NIH)

val_df_MIMIC = preprocess_MIMIC(val_df_MIMIC)
val_df_CXP = preprocess_CXP(val_df_CXP)
val_df_NIH = preprocess_NIH(val_df_NIH)


test_df_MIMIC['Jointpath'] = path_image_MIMIC + test_df_MIMIC['path'].astype(str)
test_df_CXP['Jointpath']   = path_image_CXP   + test_df_CXP['Path'].astype(str)
test_df_NIH['Jointpath']   = path_image_NIH   + test_df_NIH['Image Index'].astype(str)

train_df_MIMIC['Jointpath'] = path_image_MIMIC + train_df_MIMIC['path'].astype(str)
train_df_CXP['Jointpath']   = path_image_CXP   + train_df_CXP['Path'].astype(str)
train_df_NIH['Jointpath']   = path_image_NIH   + train_df_NIH['Image Index'].astype(str)

val_df_MIMIC['Jointpath'] = path_image_MIMIC + val_df_MIMIC['path'].astype(str)
val_df_CXP['Jointpath']   = path_image_CXP   + val_df_CXP['Path'].astype(str)
val_df_NIH['Jointpath']   = path_image_NIH   + val_df_NIH['Image Index'].astype(str)

val_df_NIH["subject_id"] =  val_df_NIH['subject_id'].astype(int)
val_df_CXP["subject_id"] =  val_df_CXP['subject_id'].astype(int)
val_df_MIMIC["subject_id"] = val_df_MIMIC['subject_id'].astype(int)

test_df_NIH["subject_id"] =  test_df_NIH['subject_id'].astype(int)
test_df_CXP["subject_id"] = test_df_CXP['subject_id'].astype(int)
test_df_MIMIC["subject_id"] =  test_df_MIMIC['subject_id'].astype(int)

train_df_NIH["subject_id"] = train_df_NIH['subject_id'].astype(int)
train_df_CXP["subject_id"] =  train_df_CXP['subject_id'].astype(int)
train_df_MIMIC["subject_id"] =  train_df_MIMIC['subject_id'].astype(int)

val_df_CXP["Effusion"] = val_df_CXP["Pleural Effusion"]
val_df_MIMIC["Effusion"] = val_df_MIMIC["Pleural Effusion"]
val_df_CXP['Airspace Opacity'] = val_df_CXP['Lung Opacity']

train_df_CXP["Effusion"] = train_df_CXP["Pleural Effusion"]
train_df_MIMIC["Effusion"] = train_df_MIMIC["Pleural Effusion"]
train_df_CXP['Airspace Opacity'] = train_df_CXP['Lung Opacity']

test_df_CXP["Effusion"] = test_df_CXP["Pleural Effusion"]
test_df_MIMIC["Effusion"] = test_df_MIMIC["Pleural Effusion"]
test_df_CXP['Airspace Opacity'] = test_df_CXP['Lung Opacity']

val_df_CXP_new = val_df_CXP[['subject_id','Jointpath','Sex',"Age",'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]
train_df_CXP_new = train_df_CXP[['subject_id','Jointpath','Sex',"Age",'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]

val_df_NIH_new = val_df_NIH[['subject_id','Jointpath','Sex',"Age",'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]
train_df_NIH_new = train_df_NIH[['subject_id','Jointpath','Sex',"Age",'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]

val_df_MIMIC_new = val_df_MIMIC[['subject_id','Jointpath','Sex',"Age",'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]
train_df_MIMIC_new = train_df_MIMIC[['subject_id','Jointpath','Sex',"Age",'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]

test_df_CXP_new = test_df_CXP[['subject_id','Jointpath','Sex','Age','No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]
test_df_NIH_new = test_df_NIH[['subject_id','Jointpath','Sex','Age','No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]
test_df_MIMIC_new = test_df_MIMIC[['subject_id','Jointpath','Sex','Age','No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]]


test_df = test_df_CXP_new.append([test_df_NIH_new, test_df_MIMIC_new])
train_df = train_df_CXP_new.append([train_df_NIH_new, train_df_MIMIC_new])
val_df = val_df_CXP_new.append([val_df_NIH_new, val_df_MIMIC_new])




