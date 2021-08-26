 # This code provide the results of subgroup-specific underdiagnosis and Intersectional specific chronic underdiagnosis by studing the NoFinding label
# since

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Config import test_df

pred = pd.read_csv("./results/bipred.csv")

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




def FP_NF_ALL(Pred, diseases, category, category_name):
    df = test_df.merge(Pred, left_on='Jointpath', right_on='Jointpath')
    
    GAP_total = np.zeros((len(category), len(diseases)))
    percentage_total = np.zeros((len(category), len(diseases)))
    cate = []

    if category_name == 'Sex':
        FP_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'Age':
        FP_age = pd.DataFrame(diseases, columns=["diseases"])

    print("FP in ALLDAta====================================")
    i=0
    for c in range(len(category)):
        for d in range(len(diseases)):
            pred_disease = "bi_" + diseases[d]
            gt = df.loc[(df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name] == category[c]), :]



            if len(gt) != 0:
                FPR = len(pred) / len(gt)
                print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: " + str(FPR))

                if category_name == 'Sex':
                    if i == 0:
                        FPR_S = pd.DataFrame([FPR], columns=["M"])
                        FP_sex = pd.concat([FP_sex, FPR_S.reindex(FP_sex.index)], axis=1)

                    if i == 1:
                        FPR_S = pd.DataFrame([FPR], columns=["F"])
                        FP_sex = pd.concat([FP_sex, FPR_S.reindex(FP_sex.index)], axis=1)

                # make sure orders are right

                if category_name == 'Age':
                    if i == 0:
                        FPR_A = pd.DataFrame([FPR], columns=["40-60"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 1:
                        FPR_A = pd.DataFrame([FPR], columns=["60-80"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 2:
                        FPR_A = pd.DataFrame([FPR], columns=["20-40"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 3:
                        FPR_A = pd.DataFrame([FPR], columns=["80-"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 4:
                        FPR_A = pd.DataFrame([FPR], columns=["0-20"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)


            else:
                print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: N\A")

        i= i+1

 
    if category_name == 'Sex':
        FP_sex.to_csv("./results/FP_sex.csv")

    if category_name == 'Age':
        FP_age.to_csv("./results/FP_age.csv")

    
    return FPR


def FP_NF_ALL_Inter(Pred, diseases, category1, category_name1,category2, category_name2 ):
    
    
    df = test_df.merge(Pred, left_on='Jointpath', right_on='Jointpath')

    if (category_name1 == 'Sex')  &  (category_name2 == 'Age'):
        FP_AgeSex = pd.DataFrame(category2, columns=["SexAge"])


    print("FP in ALL Data====================================")
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
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: N\A")


            FPR_list.append(FPR)

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




    return FPR


def FPR_Underdiagnosis():
    #MIMIC data
    diseases = ['No Finding']
    Age = ['40-60', '60-80', '20-40', '80-', '0-20']
    Sex = ['M', 'F']

    pred_ALL = pd.read_csv("./results/bipred.csv")
    factor_ALL = [Sex, Age]
    factor_str_ALL = ['Sex', 'Age']

    
    #Subgroup-specific Chronic Underdiagnosis
    FP_NF_ALL(pred_ALL, diseases, Age, 'Age')
    
    FP_NF_ALL(pred_ALL, diseases, Sex, 'Sex')
    

    
    #Intersectional-specific Chronic Underdiagnosis
    
    FP_NF_ALL_Inter(pred_ALL, diseases, Sex, 'Sex',Age,'Age')





if __name__ == '__main__':
    FPR_Underdiagnosis()