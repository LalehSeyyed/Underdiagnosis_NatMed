 # This code provide the results of subgroup-specific underdiagnosis and Intersectional specific chronic underdiagnosis by studing the NoFinding label
# since

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fpr(df, d, c, category_name):
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 0) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] == c), :]
    if len(gt) != 0:
        FPR = len(pred) / len(gt)
        return FPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return -1

def preprocess_MIMIC(split):
    # total_subject_id = pd.read_csv("total_subject_id_with_gender.csv")
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/mimic-cxr-metadata-detail.csv")
    details = details.drop(columns=['dicom_id', 'study_id'])
    details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
    if "subject_id" not in split.columns:
        subject_id = []
        for idx, row in split.iterrows():
            subject_id.append(row['path'].split('/')[1][1:])
        split['subject_id'] = subject_id
        split = split.sort_values("subject_id")
    if "gender" not in split.columns:
        split["subject_id"] = pd.to_numeric(split["subject_id"])
        split = split.merge(details, left_on="subject_id", right_on="subject_id")
    split = split.replace(
        [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
         'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>=90'],
        [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
         'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    return split



def FP_NF_MIMIC(TrueWithMeta_df, df, diseases, category, category_name):
   # return FPR and FNR per subgroup and the unber of patients with 0 No-finding in test set.
   
    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")
   
    
    FP_total = []
    percentage_total = []
    FN_total = []
    

    if category_name == 'insurance':
        FPR_Ins = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        FPR_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'gender':
        FPR_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        FPR_age = pd.DataFrame(diseases, columns=["diseases"])

    print("FP in MIMIC====================================")
    
    for c in category:
        FP_y = []
        FN_y = []
        percentage_y = []
        
        for d in diseases:
            pred_disease = "bi_" + d
            
            gt_fp = df.loc[(df[d] == 0) & (df[category_name] == c), :]
            gt_fn = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            
            pred_fp = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] == c), :]
            
            pred_fn = df.loc[(df[pred_disease] == 0) & (df[d] == 1) & (df[category_name] == c), :]
            
            # number of patient in subgroup with actual NF=0
            pi_gy = df.loc[(df[d] == 0) & (df[category_name] == c), :]
         #   pi_y = df.loc[(df[d] == 0) & (df[category_name] != 0), :]
                
            if len(gt_fp) != 0 :
                FPR = len(pred_fp) / len(gt_fp)
                Percentage = len(pi_gy) # we remove# to number 
                
                FP_y.append(round(FPR,3))
                percentage_y.append(round(Percentage,3))
            else:
                FP_y.append(np.NaN)
                percentage_y.append(0)
                

            if len(gt_fn) != 0 :
                FNR = len(pred_fn) / len(gt_fn)
                FN_y.append(round(FNR,3))
                
            else:
                FN_y.append(np.NaN)
                           
        FP_total.append(FP_y)
        percentage_total.append(percentage_y)
        FN_total.append(FN_y)
                
#                 print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: " + str(FPR))

    for  i in range(len(FN_total)):
        
        if category_name == 'gender':
            if i == 0:
                Perc_S = pd.DataFrame(percentage_total[i], columns=["#M"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)
                
                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_M"])
                FPR_sex = pd.concat([FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)
                
                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_M"])
                FPR_sex = pd.concat([FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)
                
                

            if i == 1:
                Perc_S = pd.DataFrame(percentage_total[i], columns=["#F"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)
                
                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_F"])
                FPR_sex = pd.concat([FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)
                
                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_F"])
                FPR_sex = pd.concat([FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)

            FPR_sex.to_csv("./results/FPR_FNR_NF_sex.csv")    

        if category_name == 'age_decile':
            if i == 0:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#60-80"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_60-80"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_60-80"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)                
                
            if i == 1:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#40-60"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_40-60"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_40-60"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1) 

            if i == 2:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#20-40"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_20-40"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_20-40"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1) 

            if i == 3:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#80-"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_80-"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_80-"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1) 

            if i == 4:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#0-20"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_0-20"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_0-20"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)
            FPR_age.to_csv("./results/FPR_FNR_NF_age.csv")
            
        if category_name == 'insurance':
            if i == 0:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Medicare"])
                FPR_Ins = pd.concat([FPR_Ins, Perc_A.reindex(FPR_Ins.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Medicare"])
                FPR_Ins = pd.concat([FPR_Ins, FPR_A.reindex(FPR_Ins.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Medicare"])
                FPR_Ins = pd.concat([FPR_Ins, FNR_A.reindex(FPR_Ins.index)], axis=1)            

            if i == 1:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Other"])
                FPR_Ins = pd.concat([FPR_Ins, Perc_A.reindex(FPR_Ins.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Other"])
                FPR_Ins = pd.concat([FPR_Ins, FPR_A.reindex(FPR_Ins.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Other"])
                FPR_Ins = pd.concat([FPR_Ins, FNR_A.reindex(FPR_Ins.index)], axis=1)    

            if i == 2:
                
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Medicaid"])
                FPR_Ins = pd.concat([FPR_Ins, Perc_A.reindex(FPR_Ins.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Medicaid"])
                FPR_Ins = pd.concat([FPR_Ins, FPR_A.reindex(FPR_Ins.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Medicaid"])
                FPR_Ins = pd.concat([FPR_Ins, FNR_A.reindex(FPR_Ins.index)], axis=1)                   
            
            FPR_Ins.to_csv("./results/FPR_FNR_NF_insurance.csv")    

        if category_name == 'race':
            if i == 0:
                
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#White"])
                FPR_race = pd.concat([FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_White"])
                FPR_race = pd.concat([FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_White"])
                FPR_race = pd.concat([FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)                  
                

            if i == 1:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Black"])
                FPR_race = pd.concat([FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Black"])
                FPR_race = pd.concat([FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Black"])
                FPR_race = pd.concat([FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)                  
                

            if i == 2:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Hisp"])
                FPR_race = pd.concat([FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Hisp"])
                FPR_race = pd.concat([FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Hisp"])
                FPR_race = pd.concat([FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)   
                

            if i == 3:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Other"])
                FPR_race = pd.concat([FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Other"])
                FPR_race = pd.concat([FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Other"])
                FPR_race = pd.concat([FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)                  


            if i == 4:
                
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#Asian"])
                FPR_race = pd.concat([FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_Asian"])
                FPR_race = pd.concat([FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_Asian"])
                FPR_race = pd.concat([FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)                 
              


            if i == 5:
                
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#American"])
                FPR_race = pd.concat([FPR_race, Perc_A.reindex(FPR_race.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_American"])
                FPR_race = pd.concat([FPR_race, FPR_A.reindex(FPR_race.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_American"])
                FPR_race = pd.concat([FPR_race, FNR_A.reindex(FPR_race.index)], axis=1)    

            FPR_race.to_csv("./results/FPR_FNR_NF_race.csv")  

#     if category_name == 'insurance':
        

#     if category_name == 'race':
        

#     if category_name == 'gender':
        

#     if category_name == 'age_decile':
        

    
    #return FPR


def FP_FN_NF_MIMIC_Inter(TrueWithMeta_df, df, diseases, category1, category_name1,category2, category_name2 ):

    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")


    if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
        FP_InsSex = pd.DataFrame(category2, columns=["Insurance"])

    if (category_name1 == 'gender')  &  (category_name2 == 'race'):
        FP_RaceSex = pd.DataFrame(category2, columns=["race"])
    
    if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])
    
    if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
        FP_InsRace = pd.DataFrame(category2, columns=["race"])

    if (category_name1 == 'insurance')  &  (category_name2 == 'age_decile'):
        FP_InsAge = pd.DataFrame(category2, columns=["age"])        

    if (category_name1 == 'race')  &  (category_name2 == 'age_decile'):
        FP_RaceAge = pd.DataFrame(category2, columns=["age"])         


    print("FP in MIMIC====================================")
    i = 0
    for c1 in range(len(category1)):
        FPR_list = []
        FNR_list = []


        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                pred_disease = "bi_" + diseases[d]
                gt_fp =   df.loc[((df[diseases[d]] == 0)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                gt_fn =   df.loc[((df[diseases[d]] == 1)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                pred_fp = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                pred_fn = df.loc[((df[pred_disease] == 0) & (df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                


                if len(gt_fp) != 0:
                    FPR = len(pred_fp) / len(gt_fp)
                    print(len(pred_fp),'--' ,len(pred_fp))
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: " + str(FPR))

                else:
                    FPR = np.NaN
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: N\A")

                if len(gt_fn) != 0:
                    FNR = len(pred_fn) / len(gt_fn)
                    print(len(pred_fn),'--' ,len(pred_fn))
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: " + str(FPR))

                else:
                    FNR = np.NaN
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: N\A")
                    
                    
                    
            FPR_list.append(round(FPR,3))
            FNR_list.append(round(FNR,3))
            
        if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)
                
                FNR_SA = pd.DataFrame(FNR_list, columns=["FNR_M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FNR_SA.reindex(FP_AgeSex.index)], axis=1)

                

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)            

                FNR_SA = pd.DataFrame(FNR_list, columns=["FNR_F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FNR_SA.reindex(FP_AgeSex.index)], axis=1)                
                
        if (category_name1 == 'gender')  &  (category_name2 == 'race'):
            if i == 0:
                FPR_SR = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)
                
                FNR_SR = pd.DataFrame(FNR_list, columns=["FNR_M"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

            if i == 1:
                FPR_SR = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)           
            
                FNR_SR = pd.DataFrame(FNR_list, columns=["FNR_F"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

        if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
            if i == 0:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_InsSex = pd.concat([FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)
                
                FNR_SIn = pd.DataFrame(FNR_list, columns=["FNR_M"])
                FP_InsSex = pd.concat([FP_InsSex, FNR_SIn.reindex(FP_InsSex.index)], axis=1)

            if i == 1:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_InsSex = pd.concat([FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)

                FNR_SIn = pd.DataFrame(FNR_list, columns=["FNR_F"])
                FP_InsSex = pd.concat([FP_InsSex, FNR_SIn.reindex(FP_InsSex.index)], axis=1)                
                
        if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Medicare"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)
                
                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Medicare"])
                FP_InsRace = pd.concat([FP_InsRace, FNR_RIn.reindex(FP_InsRace.index)], axis=1)
                

            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Other"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)
                
                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Other"])
                FP_InsRace = pd.concat([FP_InsRace, FNR_RIn.reindex(FP_InsRace.index)], axis=1)

            if i == 2:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Medicaid"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)
                
                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Medicaid"])
                FP_InsRace = pd.concat([FP_InsRace, FNR_RIn.reindex(FP_InsRace.index)], axis=1)

        if (category_name1 == 'insurance')  &  (category_name2 == 'age_decile'):
            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Medicare"])
                FP_InsAge = pd.concat([FP_InsAge, FPR_RIn.reindex(FP_InsAge.index)], axis=1)

                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Medicare"])
                FP_InsAge = pd.concat([FP_InsAge, FNR_RIn.reindex(FP_InsAge.index)], axis=1)                
                
            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Other"])
                FP_InsAge = pd.concat([FP_InsAge, FPR_RIn.reindex(FP_InsAge.index)], axis=1)

                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Other"])
                FP_InsAge = pd.concat([FP_InsAge, FNR_RIn.reindex(FP_InsAge.index)], axis=1)                
                
            if i == 2:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Medicaid"])
                FP_InsAge = pd.concat([FP_InsAge, FPR_RIn.reindex(FP_InsAge.index)], axis=1)

                FPR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Medicaid"])
                FP_InsAge = pd.concat([FP_InsAge, FNR_RIn.reindex(FP_InsAge.index)], axis=1)                

        if (category_name1 == 'race')  &  (category_name2 == 'age_decile'):
            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_White"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_White"])
                FP_RaceAge = pd.concat([FP_RaceAge, FNR_RIn.reindex(FP_RaceAge.index)], axis=1)                
                
            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Black"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Black"])
                FP_RaceAge = pd.concat([FP_RaceAge, FNR_RIn.reindex(FP_RaceAge.index)], axis=1)                
                
            if i == 2:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Hisp"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)                

                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Hisp"])
                FP_RaceAge = pd.concat([FP_RaceAge, FNR_RIn.reindex(FP_RaceAge.index)], axis=1)                 
                
            if i == 3:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Other"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Other"])
                FP_RaceAge = pd.concat([FP_RaceAge, FNR_RIn.reindex(FP_RaceAge.index)], axis=1)                   
                
            if i == 4:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_Asian"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)
                
                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_Asian"])
                FP_RaceAge = pd.concat([FP_RaceAge, FNR_RIn.reindex(FP_RaceAge.index)], axis=1)                 
                

            if i == 5:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["FPR_American"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)   

                FNR_RIn = pd.DataFrame(FNR_list, columns=["FNR_American"])
                FP_RaceAge = pd.concat([FP_RaceAge, FNR_RIn.reindex(FP_RaceAge.index)], axis=1)                
                

        i = i + 1

    if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
        FP_InsSex.to_csv("./results/FP_FN_InsSex.csv")

    if (category_name1 == 'gender')  &  (category_name2 == 'race'):
        FP_RaceSex.to_csv("./results/FP_FN_RaceSex.csv")

    if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
        FP_InsRace.to_csv("./results/FP_FN_InsRace.csv")
    
    if (category_name1 == 'insurance')  &  (category_name2 == 'age_decile'):
        FP_InsAge.to_csv("./results/FP_FN_InsAge.csv")
        
    if (category_name1 == 'race')  &  (category_name2 == 'age_decile'):
        FP_RaceAge.to_csv("./results/FP_FN_RaceAge.csv")    
    
    if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
        FP_AgeSex.to_csv("./results/FP_FN_AgeSex.csv")

   #return FPR




   #return FPR
# Return membership number
def FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta_df, df, diseases, category1, category_name1,category2, category_name2 ):

    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")


    if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
        FP_InsSex = pd.DataFrame(category2, columns=["Insurance"])

    if (category_name1 == 'gender')  &  (category_name2 == 'race'):
        FP_RaceSex = pd.DataFrame(category2, columns=["race"])
    
    if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])
    
    if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
        FP_InsRace = pd.DataFrame(category2, columns=["race"])

    if (category_name1 == 'insurance')  &  (category_name2 == 'age_decile'):
        FP_InsAge = pd.DataFrame(category2, columns=["age"])        

    if (category_name1 == 'race')  &  (category_name2 == 'age_decile'):
        FP_RaceAge = pd.DataFrame(category2, columns=["age"])         


    print("FP in MIMIC====================================")
    i = 0
    for c1 in range(len(category1)):
        FPR_list = []

        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                pred_disease = "bi_" + diseases[d]
                gt =   df.loc[((df[diseases[d]] == 0)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                pred = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                
                # Number of patient with actual NF=0
                SubDivision= df.loc[((df[diseases[d]] == 0)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                TotalInterSec = df.loc[((df[diseases[d]] == 0)  & (df[category_name1] != 0) & (df[category_name2] != 0)), : ]

                if len(TotalInterSec) != 0:
                    Percent = len(SubDivision) / len(TotalInterSec)
                    print(len(SubDivision),'--' ,len(TotalInterSec))
                    print("Membership Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: " + str(Percent))

                else:
                    Percent = np.NaN
                    print("Membership Number in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: N\A")


            FPR_list.append(round(len(SubDivision),4))
            
        if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)            
            
        if (category_name1 == 'gender')  &  (category_name2 == 'race'):
            if i == 0:
                FPR_SR = pd.DataFrame(FPR_list, columns=["M"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

            if i == 1:
                FPR_SR = pd.DataFrame(FPR_list, columns=["F"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)           
            
            

        if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
            if i == 0:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["M"])
                FP_InsSex = pd.concat([FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)

            if i == 1:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["F"])
                FP_InsSex = pd.concat([FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)

        if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Medicare"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Other"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

            if i == 2:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Medicaid"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

        if (category_name1 == 'insurance')  &  (category_name2 == 'age_decile'):
            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Medicare"])
                FP_InsAge = pd.concat([FP_InsAge, FPR_RIn.reindex(FP_InsAge.index)], axis=1)

            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Other"])
                FP_InsAge = pd.concat([FP_InsAge, FPR_RIn.reindex(FP_InsAge.index)], axis=1)

            if i == 2:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Medicaid"])
                FP_InsAge = pd.concat([FP_InsAge, FPR_RIn.reindex(FP_InsAge.index)], axis=1)


        if (category_name1 == 'race')  &  (category_name2 == 'age_decile'):
            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["White"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Black"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 2:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Hisp"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)                

            if i == 3:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Other"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 4:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Asian"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)

            if i == 5:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["American"])
                FP_RaceAge = pd.concat([FP_RaceAge, FPR_RIn.reindex(FP_RaceAge.index)], axis=1)   
                

        i = i + 1

    if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
        FP_InsSex.to_csv("./results/Num_InsSex.csv")

    if (category_name1 == 'gender')  &  (category_name2 == 'race'):
        FP_RaceSex.to_csv("./results/Num_RaceSex.csv")

    if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
        FP_InsRace.to_csv("./results/Num_InsRace.csv")
    
    if (category_name1 == 'insurance')  &  (category_name2 == 'age_decile'):
        FP_InsAge.to_csv("./results/Num_InsAge.csv")
        
    if (category_name1 == 'race')  &  (category_name2 == 'age_decile'):
        FP_RaceAge.to_csv("./results/Num_RaceAge.csv")    
    
    if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
        FP_AgeSex.to_csv("./results/Num_AgeSex.csv")

   #return FPR



def FPR_Underdiagnosis():
    

    #MIMIC data
    diseases_MIMIC = ['No Finding']
    age_decile_MIMIC = ['60-80', '40-60', '20-40', '80-', '0-20']

    gender_MIMIC = ['M', 'F']

    race_MIMIC = ['WHITE', 'BLACK/AFRICAN AMERICAN','HISPANIC/LATINO',
            'OTHER', 'ASIAN', 'AMERICAN INDIAN/ALASKA NATIVE']
    insurance_MIMIC = ['Medicare', 'Other', 'Medicaid']

    pred_MIMIC = pd.read_csv("./results/bipred.csv")
    TrueWithMeta = pd.read_csv("./True_withMeta.csv")  
    # This TrueWithMeta came from adding meta-data to the test data. It is equivalent to : 
    #TrueTest = pd.read_csv("./results/True.csv")
    #True_withMeta = preprocess_MIMIC(TrueTest)
    # We have done this in TRUEDatawithMeta.iybn, store it as True_withMeta.csv and use it whenever it is needed.
    
    factor_MIMIC = [gender_MIMIC, age_decile_MIMIC, race_MIMIC, insurance_MIMIC]
    factor_str_MIMIC = ['gender', 'age_decile', 'race', 'insurance']


#     FP_NF_MIMIC(TrueWithMeta, pred_MIMIC, diseases_MIMIC, insurance_MIMIC, 'insurance')
    
#     FP_NF_MIMIC(TrueWithMeta, pred_MIMIC, diseases_MIMIC, age_decile_MIMIC, 'age_decile')
    
#     FP_NF_MIMIC(TrueWithMeta, pred_MIMIC, diseases_MIMIC, race_MIMIC, 'race')
    
#     FP_NF_MIMIC(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender')

    
#--------------------------------------------------------------    
    #Intersectional-specific Chronic Underdiagnosis
#--------------------------------------------------------------     
#     FP_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender',race_MIMIC,'race')

#     FP_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', insurance_MIMIC, 'insurance')

#     FP_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', age_decile_MIMIC, 'age_decile')
    
#     FP_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, race_MIMIC, 'race', age_decile_MIMIC, 'age_decile')
    
#     FP_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, insurance_MIMIC, 'insurance', age_decile_MIMIC, 'age_decile')
    
#     FP_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, insurance_MIMIC,'insurance', race_MIMIC, 'race')


    
      

    FP_FN_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender',race_MIMIC,'race')

    FP_FN_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', insurance_MIMIC, 'insurance')

    FP_FN_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', age_decile_MIMIC, 'age_decile')
    
    FP_FN_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, race_MIMIC, 'race', age_decile_MIMIC, 'age_decile')
    
    FP_FN_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, insurance_MIMIC, 'insurance', age_decile_MIMIC, 'age_decile')
    
    FP_FN_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, insurance_MIMIC,'insurance', race_MIMIC, 'race')

    
#--------------------------------------------
    # Intersection Membership number
#----------------------------------------------    

#     FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender',race_MIMIC,'race')

#     FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', insurance_MIMIC, 'insurance')

#     FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', age_decile_MIMIC, 'age_decile')
    
#     FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, race_MIMIC, 'race', age_decile_MIMIC, 'age_decile')
    
#     FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, insurance_MIMIC, 'insurance', age_decile_MIMIC, 'age_decile')
    
#     FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, insurance_MIMIC,'insurance', race_MIMIC, 'race')

    
  

    
    
    
    
if __name__ == '__main__':
    FPR_Underdiagnosis()
    
 


#  # This code provide the results of subgroup-specific underdiagnosis and Intersectional specific chronic underdiagnosis by studing the NoFinding label
# # since

# import pandas as pd
# import numpy as np
# from numpy import linalg as LA
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# def fnr(df, d, c, category_name):
#     pred_disease = "bi_" + d
#     gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
#     pred = df.loc[(df[pred_disease] == 0) & (df[d] == 1) & (df[category_name] == c), :]
#     if len(gt) != 0:
#         FNR = len(pred) / len(gt)
#         return FNR
#     else:
#         # print("Disease", d, "in category", c, "has zero division error")
#         return -1

# def preprocess_MIMIC(split):
#     # total_subject_id = pd.read_csv("total_subject_id_with_gender.csv")
#     details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/mimic-cxr-metadata-detail.csv")
#     details = details.drop(columns=['dicom_id', 'study_id'])
#     details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
#     if "subject_id" not in split.columns:
#         subject_id = []
#         for idx, row in split.iterrows():
#             subject_id.append(row['path'].split('/')[1][1:])
#         split['subject_id'] = subject_id
#         split = split.sort_values("subject_id")
#     if "gender" not in split.columns:
#         split["subject_id"] = pd.to_numeric(split["subject_id"])
#         split = split.merge(details, left_on="subject_id", right_on="subject_id")
#     split = split.replace(
#         [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
#          'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>=90'],
#         [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
#          'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
#     return split

# def preprocess_CXP(split):
#     details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/CheXpert/map.csv")
#     if 'Atelectasis' in split.columns:
#         details = details.drop(columns=['No Finding',
#        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
#        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
#        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
#        'Support Devices'])
#     split = split.merge(details, left_on="Path", right_on="Path")
#     split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
#     split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
#     split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
#     split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
#     split['Age'] = np.where(split['Age']>=80, 81, split['Age'])
#     split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81, 'Male', 'Female'], 
#                             [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-", 'M', 'F'])
#     return split


# def FN_NF_MIMIC(df, diseases, category, category_name):
#     df = preprocess_MIMIC(df)
#     map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
#     df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
#     GAP_total = np.zeros((len(category), len(diseases)))
#     percentage_total = np.zeros((len(category), len(diseases)))
#     cate = []



#     if category_name == 'insurance':
#         FN_insurance = pd.DataFrame(diseases, columns=["diseases"])

#     if category_name == 'race':
#         FN_race = pd.DataFrame(diseases, columns=["diseases"])

#     if category_name == 'gender':
#         FN_sex = pd.DataFrame(diseases, columns=["diseases"])

#     if category_name == 'age_decile':
#         FN_age = pd.DataFrame(diseases, columns=["diseases"])

#     print("FN in MIMIC====================================")
#     i=0
#     for c in range(len(category)):
#         for d in range(len(diseases)):
#             pred_disease = "bi_" + diseases[d]
#             gt = df.loc[(df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
#             pred = df.loc[(df[pred_disease] == 0) & (df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
#             # n_gt = df.loc[(df[diseases[d]] == 0) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
#             # n_pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
#             # pi_gy = df.loc[(df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
#             # pi_y = df.loc[(df[diseases[d]] == 0) & (df[category_name] != 0), :]


#             if len(gt) != 0:
#                 FNR = len(pred) / len(gt)
#                 print("False Negative Rate in " + category[c] + " for " + diseases[d] + " is: " + str(FNR))

#                 if category_name == 'gender':
#                     if i == 0:
#                         FNR_S = pd.DataFrame([FNR], columns=["M"])
#                         FN_sex = pd.concat([FN_sex, FNR_S.reindex(FN_sex.index)], axis=1)

#                     if i == 1:
#                         FNR_S = pd.DataFrame([FNR], columns=["F"])
#                         FN_sex = pd.concat([FN_sex, FNR_S.reindex(FN_sex.index)], axis=1)

#                 # make sure orders are right

#                 if category_name == 'age_decile':
#                     if i == 0:
#                         FNR_A = pd.DataFrame([FNR], columns=["60-80"])
#                         FN_age = pd.concat([FN_age, FNR_A.reindex(FN_age.index)], axis=1)

#                     if i == 1:
#                         FNR_A = pd.DataFrame([FNR], columns=["40-60"])
#                         FN_age = pd.concat([FN_age, FNR_A.reindex(FN_age.index)], axis=1)

#                     if i == 2:
#                         FNR_A = pd.DataFrame([FNR], columns=["20-40"])
#                         FN_age = pd.concat([FN_age, FNR_A.reindex(FN_age.index)], axis=1)

#                     if i == 3:
#                         FNR_A = pd.DataFrame([FNR], columns=["80-"])
#                         FN_age = pd.concat([FN_age, FNR_A.reindex(FN_age.index)], axis=1)

#                     if i == 4:
#                         FNR_A = pd.DataFrame([FNR], columns=["0-20"])
#                         FN_age = pd.concat([FN_age, FNR_A.reindex(FN_age.index)], axis=1)

#                 if category_name == 'insurance':
#                     if i == 0:
#                         FNR_Ins = pd.DataFrame([FNR], columns=["Medicare"])
#                         FN_insurance = pd.concat([FN_insurance, FNR_Ins.reindex(FN_insurance.index)], axis=1)

#                     if i == 1:
#                         FNR_Ins = pd.DataFrame([FNR], columns=["Other"])
#                         FN_insurance = pd.concat([FN_insurance, FNR_Ins.reindex(FN_insurance.index)], axis=1)


#                     if i == 2:
#                         FNR_Ins = pd.DataFrame([FNR], columns=["Medicaid"])
#                         FN_insurance = pd.concat([FN_insurance, FNR_Ins.reindex(FN_insurance.index)], axis=1)

#                 if category_name == 'race':
#                     if i == 0:
#                         FNR_Rac = pd.DataFrame([FNR], columns=["White"])
#                         FN_race = pd.concat([FN_race, FNR_Rac.reindex(FN_race.index)], axis=1)

#                     if i == 1:
#                         FNR_Rac = pd.DataFrame([FNR], columns=["Black"])
#                         FN_race = pd.concat([FN_race, FNR_Rac.reindex(FN_race.index)], axis=1)


#                     if i == 2:
#                         FNR_Rac = pd.DataFrame([FNR], columns=["Hisp"])
#                         FN_race = pd.concat([FN_race, FNR_Rac.reindex(FN_race.index)], axis=1)

#                     if i == 3:
#                         FNR_Rac = pd.DataFrame([FNR], columns=["Other"])
#                         FN_race = pd.concat([FN_race, FNR_Rac.reindex(FN_race.index)], axis=1)

#                     if i == 4:
#                         FNR_Rac = pd.DataFrame([FNR], columns=["Asian"])
#                         FN_race = pd.concat([FN_race, FNR_Rac.reindex(FN_race.index)], axis=1)

#                     if i == 5:
#                         FNR_Rac = pd.DataFrame([FNR], columns=["American"])
#                         FN_race = pd.concat([FN_race, FNR_Rac.reindex(FN_race.index)], axis=1)


#             else:
#                 print("False Negative Rate in " + category[c] + " for " + diseases[d] + " is: N\A")

#         i= i+1

#     if category_name == 'insurance':
#         FN_insurance.to_csv("./results/FN_insurance.csv")

#     if category_name == 'race':
#         FN_race.to_csv("./results/FN_race.csv")

#     if category_name == 'gender':
#         FN_sex.to_csv("./results/FN_sex.csv")

#     if category_name == 'age_decile':
#         FN_age.to_csv("./results/FN_age.csv")

    
#     return FNR


# def FN_NF_MIMIC_Inter(df, diseases, category1, category_name1,category2, category_name2 ):
#     df = preprocess_MIMIC(df)
#     map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
#     df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
#     # GAP_total = np.zeros((len(category), len(diseases)))
#     # percentage_total = np.zeros((len(category), len(diseases)))
#     # cate = []

#     if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
#         FN_InsSex = pd.DataFrame(category2, columns=["Insurance"])

#     if (category_name1 == 'gender')  &  (category_name2 == 'race'):
#         FN_RaceSex = pd.DataFrame(category2, columns=["race"])

#     if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
#         FN_InsRace = pd.DataFrame(category2, columns=["race"])
        
#     if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
#         FN_AgeSex = pd.DataFrame(category2, columns=["SexAge"])


#     print("FN in MIMIC====================================")
#     i = 0
#     for c1 in range(len(category1)):
#         FNR_list = []

#         for c2 in range(len(category2)):
#             for d in range(len(diseases)):
#                 pred_disease = "bi_" + diseases[d]
#                 gt =   df.loc[((df[diseases[d]] == 1)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
#                 pred = df.loc[((df[pred_disease] == 0) & (df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]


#                 if len(gt) != 0:
#                     FNR = len(pred) / len(gt)
#                     print(len(pred),'--' ,len(gt))
#                     print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: " + str(FNR))

#                 else:
#                     print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: N\A")


#             FNR_list.append(FNR)
            
#         if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
#             if i == 0:
#                 FNR_SA = pd.DataFrame(FNR_list, columns=["M"])
#                 FN_AgeSex = pd.concat([FN_AgeSex, FNR_SA.reindex(FN_AgeSex.index)], axis=1)

#             if i == 1:
#                 FNR_SA = pd.DataFrame(FNR_list, columns=["F"])
#                 FN_AgeSex = pd.concat([FN_AgeSex, FNR_SA.reindex(FN_AgeSex.index)], axis=1)            
            
            

#         if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
#             if i == 0:
#                 FNR_SIn = pd.DataFrame(FNR_list, columns=["M"])
#                 FN_InsSex = pd.concat([FN_InsSex, FNR_SIn.reindex(FN_InsSex.index)], axis=1)

#             if i == 1:
#                 FNR_SIn = pd.DataFrame(FNR_list, columns=["F"])
#                 FN_InsSex = pd.concat([FN_InsSex, FNR_SIn.reindex(FN_InsSex.index)], axis=1)

#         if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
#             if i == 0:
#                 FNR_RIn = pd.DataFrame(FNR_list, columns=["Medicare"])
#                 FN_InsRace = pd.concat([FN_InsRace, FNR_RIn.reindex(FN_InsRace.index)], axis=1)

#             if i == 1:
#                 FNR_RIn = pd.DataFrame(FNR_list, columns=["Other"])
#                 FN_InsRace = pd.concat([FN_InsRace, FNR_RIn.reindex(FN_InsRace.index)], axis=1)

#             if i == 2:
#                 FNR_RIn = pd.DataFrame(FNR_list, columns=["Medicaid"])
#                 FN_InsRace = pd.concat([FN_InsRace, FNR_RIn.reindex(FN_InsRace.index)], axis=1)


#         if (category_name1 == 'gender')  &  (category_name2 == 'race'):
#             if i == 0:
#                 FNR_SR = pd.DataFrame(FNR_list, columns=["M"])
#                 FN_RaceSex = pd.concat([FN_RaceSex, FNR_SR.reindex(FN_RaceSex.index)], axis=1)

#             if i == 1:
#                 FNR_SR = pd.DataFrame(FNR_list, columns=["F"])
#                 FN_RaceSex = pd.concat([FN_RaceSex, FNR_SR.reindex(FN_RaceSex.index)], axis=1)


#         i = i + 1

#     if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
#         FN_InsSex.to_csv("./results/FN_InsSex.csv")

#     if (category_name1 == 'gender')  &  (category_name2 == 'race'):
#         FN_RaceSex.to_csv("./results/FN_RaceSex.csv")

#     if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
#         FN_InsRace.to_csv("./results/FN_InsRace.csv")
    
#     if (category_name1 == 'gender')  &  (category_name2 == 'age_decile'):
#         FN_AgeSex.to_csv("./results/FN_AgeSex.csv")

#     return FNR


# def FNR_Underdiagnosis():
    

#     #MIMIC data
#     diseases_MIMIC = [ 'No Finding']
#     age_decile_MIMIC = ['60-80', '40-60', '20-40', '80-', '0-20']

#     gender_MIMIC = ['M', 'F']

#     race_MIMIC = ['WHITE', 'BLACK/AFRICAN AMERICAN','HISPANIC/LATINO',
#             'OTHER', 'ASIAN', 'AMERICAN INDIAN/ALASKA NATIVE']
#     insurance_MIMIC = ['Medicare', 'Other', 'Medicaid']

#     pred_MIMIC = pd.read_csv("./results/bipred.csv")
#     factor_MIMIC = [gender_MIMIC, age_decile_MIMIC, race_MIMIC, insurance_MIMIC]
#     factor_str_MIMIC = ['gender', 'age_decile', 'race', 'insurance']

#     # #CXP data
#     # diseases_CXP = ['No Finding']
#     # Age_CXP = ['60-80', '40-60', '20-40', '80-', '0-20']
#     # gender_CXP = ['M', 'F']
#     # pred_CXP = pd.read_csv("./CXP/results/bipred.csv")
#     # factor_CXP = [gender_CXP, Age_CXP]
#     # factor_str_CXP = ['Sex', 'Age']

#     # False positive rates are study on 'No Finding' label of MIMIC-CXR dataset only
    
#     #Subgroup-specific Chronic Underdiagnosis
#     FN_NF_MIMIC(pred_MIMIC, diseases_MIMIC, insurance_MIMIC, 'insurance')
    
#     FN_NF_MIMIC(pred_MIMIC, diseases_MIMIC, age_decile_MIMIC, 'age_decile')
    
#     FN_NF_MIMIC(pred_MIMIC, diseases_MIMIC, race_MIMIC, 'race')
    
#     FN_NF_MIMIC(pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender')
    
#     #Intersectional-specific Chronic Underdiagnosis
    
#     FN_NF_MIMIC_Inter(pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender',race_MIMIC,'race')

#     FN_NF_MIMIC_Inter(pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', insurance_MIMIC, 'insurance')

#     FN_NF_MIMIC_Inter(pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', age_decile_MIMIC, 'age_decile')
    
#     FN_NF_MIMIC_Inter(pred_MIMIC, diseases_MIMIC, insurance_MIMIC,'insurance', race_MIMIC, 'race')




# if __name__ == '__main__':
#     FNR_Underdiagnosis()