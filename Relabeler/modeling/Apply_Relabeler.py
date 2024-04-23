import os
import time
import yaml
from datetime import datetime
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Read the settings
settings_file = ''
if len(sys.argv) == 2:
    settings_file = sys.argv[1]
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "Run with settings.py file at: " + settings_file)
else:
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "No settings.yaml file found as argument, try to find the file in the project folder.")
    settings_file = "../filter_modeling/settings_apply_relabeler.yaml"
settings_file = open(settings_file)
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

models = settings["models"]
output_folder = settings["output_folder"]
years = settings["years"]
retroBM_datasets = settings["retroBM_datasets"]
retroBM_datasets_original = settings["retroBM_datasets_original"]
DBM_predictions = settings["DBM_prediction_paths"]
fgsi_golden = settings["fgsi_golden"]
prefixes = settings["prefixes"]

# Initialize probability thresholds with a range of values
prob_thresholds = np.concatenate([np.arange(0.5, 0.9, 0.05), np.arange(0.9, 1 + 0.001, 0.005)])
if "prob_thresholds" in settings.keys():
    prob_thresholds = settings["prob_thresholds"]

datasets = []

for i in range(len(years)):
    dataset = {}
    dataset["year"] = years[i]
    dataset["retroBM_dataset"] = retroBM_datasets[i]
    dataset["retroBM_dataset_original"] = retroBM_datasets_original[i]
    dataset["DBM_predictions"] = DBM_predictions[i]
    dataset["fgsi_golden"] = fgsi_golden[i]
    dataset["prefix"] = prefixes[i]
    datasets.append(dataset)

df = pd.DataFrame()

for dataset in datasets:
    print(datasets)
    year = dataset['year']
    retroBM_dataset_path = dataset['retroBM_dataset']
    retroBM_dataset_original_path = dataset['retroBM_dataset_original']
    descriptor_list_full = pd.read_csv(retroBM_dataset_path + os.path.sep + "FN_SCL_selected.csv")
    descriptor_list = list(descriptor_list_full["Descr. UI"])

    # Original FGSI testset, all articles, regardless of any predictions
    # includes pmids and golden/ground-truth values
    test_year = pd.read_csv(retroBM_dataset_original_path + os.path.sep + "test_" + year + ".csv")

    # predictions for the original FGSI dataset
    prediction_pickle_file_DBM = dataset['DBM_predictions']
    fgsi_predictions = None
    with open(prediction_pickle_file_DBM, 'rb') as f:
      fgsi_predictions = pickle.load(f)

    # golden values for the original FGSI dataset
    prediction_pickle_file_golden = dataset['fgsi_golden']
    fgsi_golden = None
    with open(prediction_pickle_file_golden, 'rb') as f:
      fgsi_golden = pickle.load(f)

    # Filtering the CO predictions instead of DBM results
    predictions_CO = pd.read_csv(retroBM_dataset_original_path + os.path.sep + 'label_matrix_test_concept_occurrence_label_' + year + '_filtered.csv')
    predictions_ALO3 = pd.read_csv(retroBM_dataset_original_path + os.path.sep + 'label_matrix_test_minority_voter_' + year + '_filtered.csv')

    # print(descriptor_list)
    for descriptor in descriptor_list:
        index = descriptor_list.index(descriptor)
        # predictions for the original FGSI dataset
        fgsi_label_predictions = fgsi_predictions[:,index]

        # predictions for the original FGSI dataset
        fgsi_label_golden = fgsi_golden[:,index]

        fgsi_label_predictions_CO = predictions_CO[descriptor].values
        fgsi_label_predictions_ALO3 = predictions_ALO3[descriptor].values
        co_cocuments = predictions_CO[descriptor].sum()
        alo3_cocuments = predictions_ALO3[descriptor].sum()

        # confusion matrix (original predictions)
        DBM_original_report = classification_report(fgsi_label_golden, fgsi_label_predictions, output_dict = True)
        CO_original_report = classification_report(fgsi_label_golden, fgsi_label_predictions_CO, output_dict = True)
        ALO3_original_report = classification_report(fgsi_label_golden, fgsi_label_predictions_ALO3, output_dict = True)

        # Filtering dataset (corresponding to an FGSI dataset, but containing only predicted negatives by the weak labeling)
        # Includes pmids, but no predicted values
        descriptors = pd.read_csv(retroBM_dataset_path + os.path.sep + dataset["prefix"] + descriptor + os.path.sep + "test.csv")

        valid_cocuments = len(descriptors[descriptors['valid_SCL'].str.contains(descriptor, na=False)])

        for model in models:
            # FN predicitons on filtering dataset (corrsponding to a FGSI dataset, but containing only predicted negatives by the weak labeling)
            # only values, no pmids
            predictions_probs = []
            prediction_prob_pickle_file = model + os.path.sep + "zs_"+descriptor+"predition_probs.pkl"
            with open(prediction_prob_pickle_file, 'rb') as pf:
                predictions_probs = pickle.load(pf)

            pred_probs = []
            for row in predictions_probs:
              pred_probs.extend(row)

            #Combining pmids and predicted values for FN filtering
            if 'filterProb' in descriptors.columns:
              descriptors = descriptors.drop(['filterProb'], axis=1)
            descriptors['filterProb'] = pred_probs

            for preb_thres in prob_thresholds:

              #Keeping only the (few) positive cases from FN filtering predictions (i.e. articles to be added as additonal positives)
              data_selected = descriptors[(descriptors['filterProb'] > preb_thres)]

              data_selected = data_selected[data_selected['valid_SCL'].str.contains(descriptor, na=False)]
              data_selected.count()

              # Filter to be applied in original predictions for an FGSI test
              fixed_predictions = []
              # for each document/row in the original FGSI testset (all documents)
              for ind in test_year.index:
                  pmid = test_year['pmid'][ind]
                  # print(data_selected["pmid"])
                  if pmid in data_selected["pmid"].values:
                    fixed_predictions.append(1)
                  else:
                    fixed_predictions.append(0)

              # FN-filtered preditcitions
              fgsi_label_predictions_FN_filtered = list(map(max, zip(fgsi_label_predictions, fixed_predictions)))

              # # confusion matrix ( FN-filtered preditcitions)
              DBM_FN_filtered_report = classification_report(fgsi_label_golden, fgsi_label_predictions_FN_filtered, output_dict = True)
              DBM_Fil_tn, DBM_Fil_fp, DBM_Fil_fn, DBM_Fil_tp = confusion_matrix(fgsi_label_golden, fgsi_label_predictions_FN_filtered).ravel()

              # FN-filtered CO preditcitions
              fgsi_label_CO_predictions_FN_filtered = list(map(max, zip(fgsi_label_predictions_CO, fixed_predictions)))
              fgsi_label_ALO3_predictions_FN_filtered = list(map(max, zip(fgsi_label_predictions_ALO3, fixed_predictions)))

              # confusion matrix ( FN-filtered CO preditcitions)
              CO_FN_filtered_report = classification_report(fgsi_label_golden, fgsi_label_CO_predictions_FN_filtered, output_dict = True)
              CO_Fil_tn, CO_Fil_fp, CO_Fil_fn, CO_Fil_tp = confusion_matrix(fgsi_label_golden, fgsi_label_CO_predictions_FN_filtered).ravel()
              ALO3_FN_filtered_report = classification_report(fgsi_label_golden, fgsi_label_ALO3_predictions_FN_filtered, output_dict = True)
              ALO3_Fil_tn, ALO3_Fil_fp, ALO3_Fil_fn, ALO3_Fil_tp = confusion_matrix(fgsi_label_golden, fgsi_label_ALO3_predictions_FN_filtered).ravel()

              overall_results={
                              'model':model,
                              'probability_thresshold':preb_thres,
                              'year':year,
                              'Descritor':descriptor,
                              'valid_docs':valid_cocuments,
                              'co_docs':co_cocuments,
                              "support":DBM_FN_filtered_report["1.0"]['support'],
                              'CO_F1':CO_original_report["1.0"]['f1-score'],
                              'CO_P':CO_original_report["1.0"]['precision'],
                              'CO_R':CO_original_report["1.0"]['recall'],
                              'ALO3_F1':ALO3_original_report["1.0"]['f1-score'],
                              'ALO3_P':ALO3_original_report["1.0"]['precision'],
                              'ALO3_R':ALO3_original_report["1.0"]['recall'],
                              'DBM_F1':DBM_original_report["1.0"]['f1-score'],
                              'DBM_P':DBM_original_report["1.0"]['precision'],
                              'DBM_R':DBM_original_report["1.0"]['recall'],
                              "CO_FNfil_F1":CO_FN_filtered_report["1.0"]['f1-score'],
                              "CO_FNfil_P":CO_FN_filtered_report["1.0"]['precision'],
                              "CO_FNfil_R":CO_FN_filtered_report["1.0"]['recall'],
                              "ALO3_FNfil_F1":ALO3_FN_filtered_report["1.0"]['f1-score'],
                              "ALO3_FNfil_P":ALO3_FN_filtered_report["1.0"]['precision'],
                              "ALO3_FNfil_R":ALO3_FN_filtered_report["1.0"]['recall'],
                              "DBM_FNfil_F1":DBM_FN_filtered_report["1.0"]['f1-score'],
                              "DBM_FNfil_P":DBM_FN_filtered_report["1.0"]['precision'],
                              "DBM_FNfil_R":DBM_FN_filtered_report["1.0"]['recall'],
                              "DBM_Fil_tn":DBM_Fil_tn,
                              "DBM_Fil_fp" :  DBM_Fil_fp,
                              "DBM_Fil_fn" :  DBM_Fil_fn,
                              "DBM_Fil_tp" :  DBM_Fil_tp,
                              "CO_Fil_tn":  CO_Fil_tn,
                              "CO_Fil_fp":  CO_Fil_fp,
                              "CO_Fil_fn":  CO_Fil_fn,
                              "CO_Fil_tp":  CO_Fil_tp,
                              "ALO3_Fil_tn" :  ALO3_Fil_tn,
                              "ALO3_Fil_fp" :  ALO3_Fil_fp,
                              "ALO3_Fil_fn" :  ALO3_Fil_fn,
                              "ALO3_Fil_tp" :  ALO3_Fil_tp
                              }

              print(overall_results)
              df = pd.concat([df, pd.DataFrame([overall_results])], ignore_index=True)

        # print(df)
        df.to_csv(output_folder + os.path.sep + 'FN_Filtered_results.csv')
