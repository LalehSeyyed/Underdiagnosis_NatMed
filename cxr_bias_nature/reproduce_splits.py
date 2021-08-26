from pathlib import Path

def reproduce_split(preprocessed_path: str, dataset: str, output_dir: str, fold_mapping_path: str = './fold_mapping_laleh.csv'):
    '''
    preprocessed_path (str): path to the preprocessed csv file (map.csv for CXP, preprocessed.csv for MIMIC and NIH)
    dataset (str): one of ['MIMIC', 'CXP', 'NIH']
    output_dir (str): folder to output split
    fold_mapping_path (str): path to the fold_mapping csv file
    '''
    fold_mapping = pd.read_csv(fold_mapping_path).query(f"dataset == '{dataset}'")
    subject_id_col = 'subject_id' if dataset in ['MIMIC', 'CXP'] else 'Patient ID'
    df = pd.read_csv(preprocessed_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok = True, parents = True)
    
    for fold in ['train', 'valid', 'test']:
        sub_df = df[df[subject_id_col].isin(fold_mapping.loc[fold_mapping['fold'] == fold, 'subject_id'])]
        print(len(sub_df))
        sub_df.to_csv(output_dir/f"new_{fold}.csv", index=False)

# sample usage: reproduce_split('/scratch/hdd001/projects/ml4h/projects/CheXpert/map.csv', 'CXP', './test')