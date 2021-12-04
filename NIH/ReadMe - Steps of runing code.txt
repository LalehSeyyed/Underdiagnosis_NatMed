A) Trainig and testing the network

1 - Train your network using "MODE = train"  --> the trained model will be saved in Checkpoint

2 - Test your network using "MODE = test" and runing main.py
* The following csv files are generated:
**Eval.csv (contain AUC on validation set and thereshold based on maximizing f1 score on validation set)
**TestEval.csv (contain AUC on test set )  --> drop thereshod here, it is not meaningfull

3 - Rename TestEval.csv to Evel*.csv, where * is the number of run (e.g Evel1.csv for run1).


4 -  Run the code "FPRFNR.py" to calculate:


    Num_PNF_AgeSex.csv: Number of patients with actual No Finding = 1 within the intersection
    Num_NNF_AgeSex.csv: Number of patients with actual No Finding = 0 within the intersection
    FPFN_AgeSex.csv: return FPR and FNR for the 'No Finding' for age-sex intersection
    FPRFNR_NF_age.csv: return FPR and FNR for the 'No Finding' for age and the number of patients with actual No Finding 0 and 1
    FPRFNR_NF_sex.csv: return FPR and FNR for the 'No Finding' for sex and the number of patients with actual No Finding 0 and 1

5 - rename the results forlder followed by the applied random seed for the checkpoint. (e.g. for random seed 31 use results31)

Do the step 2 to 5 for all 5 runs (each run with same hyper parameter but different seed) per dataset.

6 - Create a folder and call it "results" to later save the results that gathered from mean of each run and the confidence interval (CI) using confidence.py 
--------------------------------
7 - Run the code in Confidence.ipynb it will gives:
    a) The satatistics of the dataset that we presented in table 1 such as percentage of each subgroup on NIH, etc.
    b) The mean of AUC per disease over 5 run and the 95% confidence intervals.
    c) plots of subgroup-specific underdiagnosis and overdiagnosis rate for age and sex attributes.
    d) plots of intersectional identity chronic underdiagnosis and overdiagnosis for age and sex attributes.
------------------------------------------------------------------------------

Note: Due to data usage agreement we are not allowed to share the True.csv, Pred.csv or bi_Pred.csv file. However, sine the datasets are enough large if you merge all available data in dataset, make any 80-10-10 train validation and test set split of the dataset you can train your own model using the hyper-parameters that we used in the code, and test it using the prediction.py code and re-generate the results. If you want to test this code exactly on the same test set as our we have provided the Subjrct_ID of the patients in our test set in the TestSubjectid.csv.csv file.