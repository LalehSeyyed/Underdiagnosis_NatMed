import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

test_df_path ="/scratch/gobi2/projects/ml4h/datasets/NIH/split/July16/test.csv"
test_df = pd.read_csv(test_df_path)

pred = pd.read_csv("./results/bipred.csv")

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


test_df = preprocess_NIH(test_df)




def FPFN_NF_NIH(TrueWithMeta_df, df, diseases, category, category_name):
   # return FPR and FNR per subgroup and the unber of patients with 0 No-finding in test set.
   
    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")
   
    
    FP_total = []
    NNF_total = []
    
    FN_total = []
    PNF_total = []
    

    if category_name == 'Sex':
        FPR_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'Age':
        FPR_age = pd.DataFrame(diseases, columns=["diseases"])

    print("FP and FNR and #of negative and #of positive NF in NIH")
    
    for c in category:
        FP_y = []        
        NNF_y = []
        
        FN_y = []
        PNF_y = []
               
        for d in diseases:
            pred_disease = "bi_" + d
            
            # number of patient in subgroup with actual NF=0
            gt_fp = df.loc[(df[d] == 0) & (df[category_name] == c), :]
            # number of patient in subgroup with actual NF=1
            gt_fn = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            
            pred_fp = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] == c), :]
            
            pred_fn = df.loc[(df[pred_disease] == 0) & (df[d] == 1) & (df[category_name] == c), :]
            

                
            if len(gt_fp) != 0 :
                FPR = len(pred_fp) / len(gt_fp)
                Percentage = len(gt_fp) # we remove# to number 
                
                FP_y.append(round(FPR,3))
                NNF_y.append(round(Percentage,3))
            else:
                FP_y.append(np.NaN)
                NNF_y.append(0)
                

            if len(gt_fn) != 0 :
                FNR = len(pred_fn) / len(gt_fn)
                Percentage = len(gt_fn) # we remove# to number 
                
                FN_y.append(round(FNR,3))
                PNF_y.append(round(Percentage,3))
                
            else:
                FN_y.append(np.NaN)
                PNF_y.append(0) 
                
        FP_total.append(FP_y)
        NNF_total.append(NNF_y)
        
        FN_total.append(FN_y)
        PNF_total.append(PNF_y)
                

    for  i in range(len(FN_total)):
        
        if category_name == 'Sex':
            if i == 0:
                Perc_S = pd.DataFrame(NNF_total[i], columns=["#NNF_M"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                Perc_S = pd.DataFrame(PNF_total[i], columns=["#PNF_M"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)
                
                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_M"])
                FPR_sex = pd.concat([FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)
                
                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_M"])
                FPR_sex = pd.concat([FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)
                
                

            if i == 1:
                Perc_S = pd.DataFrame(NNF_total[i], columns=["#NNF_F"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)

                Perc_S = pd.DataFrame(PNF_total[i], columns=["#PNF_F"])
                FPR_sex = pd.concat([FPR_sex, Perc_S.reindex(FPR_sex.index)], axis=1)                
                
                FPR_S = pd.DataFrame(FP_total[i], columns=["FPR_F"])
                FPR_sex = pd.concat([FPR_sex, FPR_S.reindex(FPR_sex.index)], axis=1)
                
                FNR_S = pd.DataFrame(FN_total[i], columns=["FNR_F"])
                FPR_sex = pd.concat([FPR_sex, FNR_S.reindex(FPR_sex.index)], axis=1)

            FPR_sex.to_csv("./results/FPRFNR_NF_sex.csv")    

        if category_name == 'Age':
            if i == 0:

                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_40-60"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_40-60"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_40-60"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_40-60"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)                
                
            if i == 1:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_60-80"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_60-80"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_60-80"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_60-80"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1) 

            if i == 2:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_20-40"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)

                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_20-40"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_20-40"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_20-40"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1) 

            if i == 3:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_80-"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_80-"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)                
                                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_80-"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_80-"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1) 

            if i == 4:
                Perc_A = pd.DataFrame(NNF_total[i], columns=["#NNF_0-20"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                Perc_A = pd.DataFrame(PNF_total[i], columns=["#PNF_0-20"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)                
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_0-20"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_0-20"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)
            FPR_age.to_csv("./results/FPRFNR_NF_age.csv")


def FPFN_NF_NIH_Inter(TrueWithMeta_df, df, diseases, category1, category_name1,category2, category_name2 ):
  #  return FPR and FNR for the 'No Finding' per intersection
    
    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")

    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["SexAge"])


    print("Intersectional identity FPR and FNR")
    i = 0
    for c1 in range(len(category1)):
        FPR_list = []
        FNR_list = []

        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                pred_disease = "bi_" + diseases[d]
                
                gt_fp =   df.loc[((df[diseases[d]] == 0)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                pred_fp = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]

                gt_fn =   df.loc[((df[diseases[d]] == 1)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                pred_fn = df.loc[((df[pred_disease] == 0) & (df[diseases[d]] == 1) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]                

                if len(gt_fp) != 0:
                    FPR = len(pred_fp) / len(gt_fp)                    
                else:
                    FPR = np.NaN
                
                if len(gt_fn) != 0:
                    FNR = len(pred_fn) / len(gt_fn)
                else:
                    FNR = np.NaN


            FPR_list.append(round(FPR,3))
            FNR_list.append(round(FNR,3))

        if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)
                
                FPR_SA = pd.DataFrame(FNR_list, columns=["FNR_M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)
                
            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["FPR_F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

                FPR_SA = pd.DataFrame(FNR_list, columns=["FNR_F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)
                
        i = i + 1

    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex.to_csv("./results/FPFN_AgeSex.csv")




    


def FP_NF_NIH_MEMBERSHIP_Num_Inter(TrueWithMeta_df, df, diseases, category1, category_name1,category2, category_name2 ):

    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")

    
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])
    


    print("Number of patient with actual NF=0")
    i = 0
    for c1 in range(len(category1)):
        FPR_list = []

        for c2 in range(len(category2)):
            for d in range(len(diseases)):
            
                # Number of patient with actual NF=0
                SubDivision= df.loc[((df[diseases[d]] == 0)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                

            FPR_list.append(round(len(SubDivision),4))
            
        if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)                  

        i = i + 1


    
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex.to_csv("./results/Num_NNF_AgeSex.csv")

#------------------------------------------------------------------------
# FN, NF. Membership count with actual NF =1
#------------------------------------------------------------------------

def FN_NF_NIH_MEMBERSHIP_Num_Inter(TrueWithMeta_df, df, diseases, category1, category_name1,category2, category_name2 ):
    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")
    
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])

    print("Number of patient with actual NF=1")
    i = 0
    for c1 in range(len(category1)):
        NNF_list = []

        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                
                # Nubmer of patient within intersection with actual NF = 1
                SubDivision= df.loc[((df[diseases[d]] == 1)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                

            NNF_list.append(round(len(SubDivision),4))
            
        if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(NNF_list, columns=["M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(NNF_list, columns=["F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)            

        i = i + 1

    # Number of patients with actual No Finding = 1 within the intersection 
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex.to_csv("./results/Num_PNF_AgeSex.csv")

   #return FPR




def FPR_Underdiagnosis():
    #MIMIC data
    diseases = ['No Finding']
    Age = ['40-60', '60-80', '20-40', '80-', '0-20']
    Sex = ['M', 'F']

    pred_NIH = pd.read_csv("./results/bipred.csv")
    TrueWithMeta = pd.read_csv("./True_withMeta.csv")  
    
   
    
    factor_NIH = [Sex, Age]
    factor_str_NIH = ['Sex', 'Age']

    
#     #Subgroup-specific Chronic Underdiagnosis

    FPFN_NF_NIH(TrueWithMeta, pred_NIH, diseases, Age, 'Age')
    FPFN_NF_NIH(TrueWithMeta, pred_NIH, diseases, Sex, 'Sex')
    
   
    
#     #Intersectional-specific Chronic Underdiagnosis
    
#     FP_NF_NIH_Inter(pred_NIH, diseases, Sex, 'Sex',Age,'Age')
    FPFN_NF_NIH_Inter(TrueWithMeta, pred_NIH, diseases, Sex, 'Sex',Age,'Age')

# #--------------------------------------------------------------    
#     #Intersectional Membership FN_NF_NIH_MEMBERSHIP_Num_Inter
# #-------------------------------------------------------------- 

    FP_NF_NIH_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_NIH, diseases, Sex, 'Sex', Age, 'Age')

    FN_NF_NIH_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_NIH, diseases, Sex, 'Sex', Age, 'Age')



if __name__ == '__main__':
    FPR_Underdiagnosis()