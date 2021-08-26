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


test_df_path ="/scratch/gobi2/projects/ml4h/datasets/NIH/split/July16/test.csv"
test_df = pd.read_csv(test_df_path)

pred = pd.read_csv("./results/bipred.csv")

def tpr(df, d, c, category_name):
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
    if len(gt) != 0:
        TPR = len(pred) / len(gt)
        return TPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return -1

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


def add_NF(df, dataset):
    
    if dataset == "test_df":
        diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    else:
        diseases = ['bi_Atelectasis', 'bi_Cardiomegaly', 'bi_Consolidation', 'bi_Edema',
                    'bi_Effusion', 'bi_Emphysema', 'bi_Fibrosis', 'bi_Hernia', 'bi_Infiltration', 'bi_Mass',
                    'bi_Nodule', 'bi_Pleural_Thickening', 'bi_Pneumonia', 'bi_Pneumothorax']
    
    df["NoFinding"] = 0
    
    for disease in diseases:
        df["NoFinding"] = df["NoFinding"] + df[disease]
    
    df.loc[df['NoFinding'] > 0, 'No Finding'] = 0
    df.loc[df['NoFinding'] < 1, 'No Finding'] = 1
    
    if dataset == "pred_NIH":
        df["bi_No Finding"] = df["No Finding"]
        df = df.drop(columns=[u'No Finding', u'NoFinding'])
        
    
    return df
#---------------------------------------------------------
test_df = preprocess_NIH(test_df)
test_df = add_NF(test_df, "test_df")
#------------------------------------------------------


def plot_median(Pred, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = test_df.merge(Pred, left_on='path', right_on='path')
    
    GAP_total = []
    percentage_total = []
    cate = []
    print(diseases)



    if category_name == 'Sex':

        Run155_sex = pd.DataFrame(diseases,columns=["diseases"])

    if category_name == 'Age':

        Run155_age = pd.DataFrame(diseases,columns=["diseases"])



    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)

        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)

        
        
    GAP_total = np.array(GAP_total)
    x = np.arange(len(diseases))
    fig = plt.figure(figsize=(18,9))
    ax = fig.add_subplot(111)
    for item in x:
        mask = GAP_total[:, item] < 50
        ann = ax.annotate('', xy=(item, np.max(GAP_total[:, item][mask])), xycoords='data',
                  xytext=(item, np.min(GAP_total[:, item][mask])), textcoords='data',
                  arrowprops=dict(arrowstyle="<->",
                                  connectionstyle="bar"))

  
    for i in range(len(GAP_total)):
        s = np.multiply(percentage_total[i],1000)
        mask = GAP_total[i] < 50
        plt.scatter(x[mask], GAP_total[i][mask], s=s, marker='o', label=cate[i])


        print("Perc", percentage_total[i])
        print("GAPt", GAP_total[i][mask])

        if category_name == 'Age':

            if i== 0:
                Percent4 = pd.DataFrame(percentage_total[i], columns=["%40-60"])
                Run155_age = pd.concat([Run155_age, Percent4.reindex(Run155_age.index)], axis=1)

                Gap4 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_40-60"])
                Run155_age = pd.concat([Run155_age, Gap4.reindex(Run155_age.index)], axis=1)

            if i == 1:
                Percent6 = pd.DataFrame(percentage_total[i], columns=["%60-80"])
                Run155_age = pd.concat([Run155_age, Percent6.reindex(Run155_age.index)], axis=1)

                Gap6 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_60-80"])
                Run155_age = pd.concat([Run155_age, Gap6.reindex(Run155_age.index)], axis=1)

            if i == 2:
                Percent4 = pd.DataFrame(percentage_total[i], columns=["%20-40"])
                Run155_age = pd.concat([Run155_age, Percent4.reindex(Run155_age.index)], axis=1)

                Gap4 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_20-40"])
                Run155_age = pd.concat([Run155_age, Gap4.reindex(Run155_age.index)], axis=1)

            if i == 3:
                Percent8 = pd.DataFrame(percentage_total[i], columns=["%80-"])
                Run155_age = pd.concat([Run155_age, Percent8.reindex(Run155_age.index)], axis=1)

                Gap8 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_80-"])
                Run155_age = pd.concat([Run155_age, Gap8.reindex(Run155_age.index)], axis=1)

            if i == 4:
                Percent0 = pd.DataFrame(percentage_total[i], columns=["%0-20"])
                Run155_age = pd.concat([Run155_age, Percent0.reindex(Run155_age.index)], axis=1)

                Gap0 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_0-20"])
                Run155_age = pd.concat([Run155_age, Gap0.reindex(Run155_age.index)], axis=1)

            Run155_age.to_csv("./results/Run155_Age.csv")


        if category_name == 'Sex':

            if i == 0:
                MalePercent = pd.DataFrame(percentage_total[i], columns=["%M"])
                Run155_sex = pd.concat([Run155_sex, MalePercent.reindex(Run155_sex.index)], axis=1)

                MaleGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_M"])
                Run155_sex = pd.concat([Run155_sex, MaleGap.reindex(Run155_sex.index)], axis=1)

            else:
                FeMalePercent = pd.DataFrame(percentage_total[i], columns=["%F"])
                Run155_sex = pd.concat([Run155_sex, FeMalePercent.reindex(Run155_sex.index)], axis=1)

                FeMaleGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_F"])
                Run155_sex = pd.concat([Run155_sex, FeMaleGap.reindex(Run155_sex.index)], axis=1)

            Run155_sex.to_csv("./results/Run155_sex.csv")




    plt.xticks(x, [diseases_abbr[k] for k in diseases])
    plt.ylabel("TPR " + ylabel[category_name] + " DISPARITY")
    plt.legend()
    plt.savefig("./results/Median_Diseases_x_GAP_" + category_name + ".pdf")

def plot_sort_median(Pred, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = test_df.merge(Pred, left_on='path', right_on='path')
    
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    GAP_total = np.array(GAP_total)
    percentage_total = np.array(percentage_total)

    difference = {}
    for i in range(GAP_total.shape[1]):
        mask = GAP_total[:, i] < 50
        difference[diseases[i]] = np.max(GAP_total[:, i][mask]) - np.min(GAP_total[:, i][mask])
    sort = [(k, difference[k]) for k in sorted(difference, key=difference.get, reverse=False)]
    diseases = []
    for k, _ in sort:
        diseases.append(k)
    df = Pred

    plot_median(df, diseases, category, category_name)


def TPR_15Label():
    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'] 
    Age = ['40-60', '60-80', '20-40', '80-', '0-20']
    Sex = ['M', 'F']
    
    pred_NIH = pd.read_csv("./results/bipred.csv")
    pred_NIH = add_NF(pred_NIH, "pred_NIH")
    
    factor_NIH = [Sex, Age]
    factor_str_NIH = ['Sex', 'Age']
    
    plot_sort_median(pred_NIH, diseases,  Age, 'Age')
    plot_sort_median(pred_NIH, diseases, Sex, 'Sex')
    
    
if __name__ == '__main__':
    TPR_15Label()    