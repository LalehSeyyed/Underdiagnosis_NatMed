 # This code  is the modified version of false_pos.py. The purpose is that adding calculation of FNR per subgroup as well as the number of patients per subgroup and intersections. 

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





def FP_NF_MIMIC(TrueWithMeta_df, df, diseases, category, category_name):
   # return FPR and FNR per subgroup and the unber of patients with 0 No-finding in test set.
   
    df = df.merge(TrueWithMeta_df, left_on="Path", right_on="Path")
   
    
    FP_total = []
    percentage_total = []
    FN_total = []
    

    if category_name == 'Sex':
        FPR_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'Age':
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
        
        if category_name == 'Sex':
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

        if category_name == 'Age':
            if i == 0:

                Perc_A = pd.DataFrame(percentage_total[i], columns=["#40-60"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_40-60"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_40-60"])
                FPR_age = pd.concat([FPR_age, FNR_A.reindex(FPR_age.index)], axis=1)                
                
            if i == 1:
                Perc_A = pd.DataFrame(percentage_total[i], columns=["#60-80"])
                FPR_age = pd.concat([FPR_age, Perc_A.reindex(FPR_age.index)], axis=1)
                
                FPR_A = pd.DataFrame(FP_total[i], columns=["FPR_60-80"])
                FPR_age = pd.concat([FPR_age, FPR_A.reindex(FPR_age.index)], axis=1)
                
                FNR_A = pd.DataFrame(FN_total[i], columns=["FNR_60-80"])
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




def FP_NF_MIMIC_Inter(TrueWithMeta_df, df, diseases, category1, category_name1,category2, category_name2 ):

    df = df.merge(TrueWithMeta_df, left_on="Path", right_on="Path")

    
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])
    
     


    print("FP in MIMIC====================================")
    i = 0
    for c1 in range(len(category1)):
        FPR_list = []

        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                pred_disease = "bi_" + diseases[d]
                gt =   df.loc[((df[diseases[d]] == 0)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                pred = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                
                


                if len(gt) != 0:
                    FPR = len(pred) / len(gt)
                    print(len(pred),'--' ,len(gt))
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: " + str(FPR))

                else:
                    FPR = np.NaN
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: N\A")


            FPR_list.append(round(FPR,3))
            
        if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)            
            

 
                

        i = i + 1

  
    
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex.to_csv("./results/FP_AgeSex.csv")

   #return FPR




   #return FPR
# Return membership number
def FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta_df, df, diseases, category1, category_name1,category2, category_name2 ):

    df = df.merge(TrueWithMeta_df, left_on="Path", right_on="Path")

    
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["Age"])
    


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
            
        if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
            if i == 0:
                FPR_SA = pd.DataFrame(FPR_list, columns=["M"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)

            if i == 1:
                FPR_SA = pd.DataFrame(FPR_list, columns=["F"])
                FP_AgeSex = pd.concat([FP_AgeSex, FPR_SA.reindex(FP_AgeSex.index)], axis=1)            
            

 
                

        i = i + 1


    
    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex.to_csv("./results/Num_AgeSex.csv")

   #return FPR



def FPR_Underdiagnosis():
    

    #MIMIC data
    diseases_MIMIC = ['No Finding']
    Age = ['40-60', '60-80', '20-40', '80-', '0-20']

    Sex = ['M', 'F']


    pred_MIMIC = pd.read_csv("./results/bipred.csv")
    TrueWithMeta = pd.read_csv("./True_withMeta.csv")  

    
    factor_MIMIC = [Sex, Age]
    factor_str_MIMIC = ['Sex', 'Age']


    
    FP_NF_MIMIC(TrueWithMeta, pred_MIMIC, diseases_MIMIC, Age, 'Age')
    
    
    FP_NF_MIMIC(TrueWithMeta, pred_MIMIC, diseases_MIMIC, Sex, 'Sex')

    
#--------------------------------------------------------------    
    #Intersectional-specific Chronic Underdiagnosis
#--------------------------------------------------------------     


    FP_NF_MIMIC_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, Sex, 'Sex', Age, 'Age')
    
 

  

#--------------------------------------------------------------    
    #Intersectional-specific Chronic Underdiagnosis
#-------------------------------------------------------------- 

    FP_NF_MIMIC_MEMBERSHIP_Num_Inter(TrueWithMeta, pred_MIMIC, diseases_MIMIC, Sex, 'Sex', Age, 'Age')
    



if __name__ == '__main__':
    FPR_Underdiagnosis()
    
 