A) Trainig and testing the network

1 - Train your network using "MODE = train"  --> the trained model will be saved in Checkpoint

2 - Test your network using "MODE = test" and runing main.py
* The following csv files are generated:
**Eval.csv (contain AUC on validation set and thereshold based on maximizing f1 score on validation set)
**TestEval.csv (contain AUC on test set )  --> drop thereshod here, it is not meaningfull

3 - Rename TestEval.csv to Evel*.csv, where * is the number of run (e.g Evel1.csv for run1).


4 -  Run the code "false_pos.py" to calculate the false positive rate on "No_Finding" labels, per subgroup and per intersection of different attribiutes. 

 Note: The original classifier that we trained NIH is designed to have 14 disease label similar to the original NIH paper. For adding NF, in file NIH_15, we get the test dataset and add the NF columns such that if all the other disease labels are 0, then the No Finding is 1, otherwise it is 0. This is match to the definition of No Finding in CheXpert and MIMIC-CXR dataset. The same procedure is conducted on the binary predeiction (./results/bipred.csv) to extract the No Finding (NF) label.
 
 
Note: * rename all files properly (e.g. FP_Age.csv to . FP*_Age.csv, where * is 1 for run 1, etc. ) to show thay are the resuls of which run. This is mandatory later as the code of claculating the CI of this results will work with this naming protocol. 


5 -  Run the code "NIH_15.py" to calculate the results of TPR disparity gap and the % of patients per subgroup on 15 labels, including No Finding. We save them with name Run155_Age.csv and Run155_sex.csv in each reults folder.

6 - rename the results forlder followed by the applied random seed for the checkpoint. (e.g. for random seed 31 use results31)

Do the step 2 to 6 for all 5 runs (each run with same hyper parameter but different seed) per dataset.

7 - Create a folder and call it "results" to later save the results that gathered from mean of each run and the confidence interval (CI) using confidence.py 
--------------------------------
8 - Run the code in Confidence.ipynb it will gives:
a) The satatistics of the dataset that we presented in table 1 such as percentage of each subgroup on NIH, etc.
b) The mean of AUC per disease over 5 run and the 95% confidence intervals.
c) Subgroup-specific underdiagnosis rate for age and sex attributes.
d) Intersectional identity chronic underdiagnosis for age and sex attributes.
------------------------------------------------------------------------------

Note: Due to data usage agreement we are not allowed to share the True.csv, Pred.csv or bi_Pred.csv file. However, sine the datasets are enough large if you merge all available data in dataset, make any 80-10-10 train validation and test set split of the dataset you can train your own model using the hyper-parameters that we used in the code, and test it using the prediction.py code and re-generate the results. If you want to test this code exactly on the same test set as our we have provided the Subjrct_ID of the patients in our test set in the TestSubjectid.csv.csv file.