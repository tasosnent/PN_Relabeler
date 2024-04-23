import random
import os
import time
import yaml
import json
import codecs
import csv
import pandas as pd
from slugify import slugify
from datetime import datetime

# Read the settings
settings_file = open("settings_PN_Relabeler.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

def handleLine(line):
    '''
    The JSON datasets are JSON Objects with an Array of documents available in the field names "documents"
    However, we follow a convention that "each line is a JSON object representing a single document" to allow reading the files line by line.
    To do so, we need to ignore the comma separating each line to the next one and the closing of the array in the last line "]}".
    :param line:    A line from a JSON dataset
    :return:        The line without the final coma "," or "]}" for the last line, ready to be parsed as JSON.
    '''
    stripped_line = line.strip()
    if stripped_line.endswith(','):
        # Remove the last character
        stripped_line = stripped_line[:len(stripped_line) - 1]
    elif stripped_line.endswith(']}'):
        # Remove the last two characters
        stripped_line = stripped_line[:len(stripped_line) - 2]
    return stripped_line;

def createNoiseDataset(settings, dataset_folder, SCL, SCL_parents, logfile, add_negative_labels= True):
    '''
    Read a JSON-based dataset with golden and weak labels and develop a "PN Relabeling" dataset for these labels.
        Save it as (1) FLAIR-compatible fast test format (2) CSV file with one column per label
    :param settings: A dictionary with all the settings required for developing the PN Relabeling" dataset parsed from setting.yaml file.
    :param dataset_folder:  The folder path to write the specific dataset
    '''
    # inputFile: String path to the original datasets in JSON format
    inputFile = settings["inputFile"]
    # golenLabelField: String name of the field to get the golden labels from (e.g. "newFGDescriptors").
    golenLabelField = settings["golenLabelField"]
    # weakLabelField: String name of the field to get the weak labels from (e.g. "newFGDescriptors")
    weakLabelField = settings["weakLabelField"]
    # datasetType: A flag indicating the task of interest:{"PWL": "Positive Weak Labels" (FP vs TP),  "NWL": "Negative Weak Labels" (TN vs FN)}
    datasetType = settings["datasetType"]
    # keptInstanceRatio: A float value representing the ratio of the dataset to be considered for train/dev/test (e.g. 0.2). for value 1.0 it considers the whole dataset.
    keptInstanceRatio = float(settings["keptInstanceRatio"])
    # testRatio: A float value representing the ratio of the dataset to be considered for test
    testRatio = float(settings["testRatio"])
    # devRatio: A float value representing the ratio of the dataset to be considered for development
    devRatio = float(settings["devRatio"])
    statsOnly = settings["statsOnly"]

    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Start PN Relabeling dataset creation comparing the fields: " + golenLabelField + ", and " + weakLabelField + ". Input file: " + inputFile )
    logfile.write("\n" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
          "-->> Start PN Relabeling dataset creation comparing the fields: " + golenLabelField + ", and " + weakLabelField + ". Input file: " + inputFile + "\n" + str(SCL))
    # Open a file to write in FastText format
    fileFastTextTrain = codecs.open(dataset_folder + "train.txt", "w+", "utf-8")
    fileFastTextDev = codecs.open(dataset_folder + "dev.txt", "w+", "utf-8")
    fileFastTextTest = codecs.open(dataset_folder + "test.txt", "w+", "utf-8")

    # Open CSV file to write with one column per label
    fileCSVtrain = codecs.open(dataset_folder + "train.csv", "w+", "utf-8")
    fileCSVdev = codecs.open(dataset_folder + "dev.csv", "w+", "utf-8")
    fileCSVtest = codecs.open(dataset_folder + "test.csv", "w+", "utf-8")

    # Keep an copy of the settings in the experiment_folder
    with open(dataset_folder + os.path.sep + "settings.yaml", 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)

    # create the csv writer
    writerTrain = csv.writer(fileCSVtrain)
    writerDev = csv.writer(fileCSVdev)
    writerTest = csv.writer(fileCSVtest)
    # write a row to the csv file
    csvHeader = ["pmid", "text", "valid_SCL"] + SCL
    writerTrain.writerow(csvHeader)
    writerDev.writerow(csvHeader)
    writerTest.writerow(csvHeader)

    label_weak = {}
    label_gold = {}
    label_FP = {}
    label_FN = {}
    label_TN = {}
    # Read the dataset file line by line
    random.seed(10)
    with open(settings["workingPath"] + os.path.sep + inputFile, "r", encoding="utf8") as file:
        count = 0;
        considered = 0
        selected = 0
        golden_positive = 0
        golden_fully_negative = 0
        weak_positive = 0
        weak_fully_negative = 0
        weak_filtered = 0 # At least one weak label has been removed due to validity filtering of the weak labels
        invalid = 0 # At least one weak label has been removed due to validity filtering of the weak labels
        FP = 0
        FN = 0
        skipped = 0
        for line in file:
            # Randomly add this line to train, development or test set
            rValue = random.random()
            # If testRatio is 1, we only want a single dataset for testing
            if testRatio == 1:
                rValue = 1
            if rValue <= testRatio: # train
                writer = writerTest
                fileFastText = fileFastTextTest
            elif rValue <= testRatio + devRatio: # development
                writer = writerDev
                fileFastText = fileFastTextDev
            else :
                writer = writerTrain
                fileFastText = fileFastTextTrain
            # read each line
            stripped_line = handleLine(line)
            if not stripped_line.startswith('{"documents":['): # Skipp the first line
                count += 1
                if (count % 10000) == 0:
                    print("lines read: ", count)
                document = json.loads(stripped_line)
                # print(document["pmid"],' - ', document["newFGDescriptors"] )
                goldenLabels = document[golenLabelField]
                # find which SCL are valid for this document
                # i.e. have golden annotation of at least one parent or parent-descendant
                valid_SCL = []
                for currentSCL in SCL:
                    currentSCL_parents = SCL_parents[currentSCL]
                    if set(currentSCL_parents).intersection(goldenLabels):
                        # The intersection of parents and golden labels is not empty
                        # i.e. this is a valid SCL of interest
                        valid_SCL.append(currentSCL)

                # Get weak labels for the document as assigned by MetaMap/SemMedDB.
                weakLabels_unfiltered = document[weakLabelField]
                # filter out invalid weak labels for documents that are not (golden) annotated with any parent-descendant.
                weakLabels = set(weakLabels_unfiltered).intersection(valid_SCL)
                if len(weakLabels) != len(weakLabels_unfiltered):
                    weak_filtered = weak_filtered + 1
                # # Only keep documents with at least one weak SCL label or one SCL golden label
                # if any(item in SCL for item in weakLabels) or any(item in SCL for item in goldenLabels) :
                # # Only keep documents with at least one weak SCL label
                # if any(item in SCL for item in weakLabels) :

                considered += 1
                fastTextLabels = []
                fastTextLabelsSerealized = " "
                csvRow = [ document["pmid"],document["title"]+ " " + document["abstractText"], "~".join(valid_SCL)]
                golden_fully_negative_doc = True # This document has no golden label for any of the SCLs considered
                weak_fully_negative_doc = True # This document has no weak label for any of the SCLs considered
                invalid_doc = False # This document is not valid for any of the SCLs considered
                if len(valid_SCL) < 1:
                    invalid_doc = True
                FP_doc = False # This doc contains (at least one) FP weak label.
                FN_doc = False # This doc contains (at least one) FN weak label.
                for label in SCL:
                    # Confusion matrix and stats
                    FP_label = False
                    FN_label = False
                    TP_label = False
                    TN_label = False
                    if label not in label_weak:
                        label_weak[label] = 0
                        label_gold[label] = 0
                        label_FP[label] = 0
                        label_FN[label] = 0
                        label_TN[label] = 0
                    if label in goldenLabels:  # Golden Positive
                        golden_fully_negative_doc = False
                        label_gold[label] += 1
                        if label in weakLabels: # True Positive
                            TP_label = True
                        else: # Flase Negative
                            label_FN[label] += 1
                            FN_doc = True
                            FN_label = True
                    if label in weakLabels:  # Weak/Predicted Positive
                        weak_fully_negative_doc = False
                        label_weak[label] += 1
                        if not TP_label: # False Positive case
                            label_FP[label] += 1
                            FP_doc = True
                            FP_label = True
                    else: # Weak/Predicted Negative
                        if not FN_label: # True Negative case
                            TN_label = True
                            if label in valid_SCL:
                                label_TN[label] += 1

                    if not statsOnly:
                        # Dataset creation
                        # Caution: This is ONLY meaningful for single-label datasets!
                        if datasetType == 'PWL': # Alternative 1 "PWL" sands for positive weak label : Check for true/false positive weak labels
                            if FP_label:
                                csvRow.append(1) # 1 denotes FP
                                fastTextLabels.append("__label__not" + label)
                            else:
                                csvRow.append(0) # 0 denotes TP (When multiple SCLs are considered TN, or FN are also included in 0)
                            if add_negative_labels:
                                if TP_label:
                                    fastTextLabels.append("__label__" + label)
                        elif datasetType == 'NWL': # Alternative 2 "NWL" sands for negative weak label : Check for true/false negative weak labels
                            if FN_label:
                                csvRow.append(1) # 1 denotes FN
                                fastTextLabels.append("__label__" + label)
                            else:
                                csvRow.append(0) # 0 denotes TN (When multiple SCLs are considered TP, or FP are also included in 0)
                            if add_negative_labels:
                                if TN_label:
                                    fastTextLabels.append("__label__not" + label)
                # Write the dataset
                if not statsOnly and not invalid_doc:
                    # a) For the 'PWL' only keep documents that have some weak label
                    # b) For the 'NWL' only keep documents that have no weak label
                    if (datasetType == 'PWL' and not weak_fully_negative_doc) or (datasetType == 'NWL' and weak_fully_negative_doc):
                        select_to_write = True
                        # Under-sampling for TN only: Randomly select whether to add this instance or not
                        if weak_fully_negative_doc and not FN_doc and (random.random() > keptInstanceRatio):
                            select_to_write = False # This TN line/document is randomly selected to be neglected from the dataset
                        if select_to_write:
                            selected += 1
                            #  write this document in the fastText output dataset
                            fileFastText.write(fastTextLabelsSerealized.join(fastTextLabels) + " " + document["title"].replace('\n', ' ') + " " + document["abstractText"].replace('\n', ' ') + "\n")
                            #  write this document in the CSV output dataset
                            writer.writerow(csvRow)
                        else:
                            skipped += 1

                # Update stats
                if golden_fully_negative_doc:
                    golden_fully_negative = golden_fully_negative + 1
                else:
                    golden_positive = golden_positive + 1
                if weak_fully_negative_doc:
                    weak_fully_negative = weak_fully_negative + 1
                else:
                    weak_positive = weak_positive + 1
                if FP_doc:
                    FP = FP + 1
                if FN_doc:
                    FN = FN + 1
                if invalid_doc:
                    invalid = invalid + 1

    # Close the files
    fileFastTextTrain.close()
    fileFastTextDev.close()
    fileFastTextTest.close()
    fileCSVtrain.close()
    fileCSVdev.close()
    fileCSVtest.close()
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> End data development." + str(considered) + " docs considered, " + str(skipped) + " docs skipped due to undersampling."  )
    logfile.write("\n"+datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') +
          "-->> End data development." + str(considered) + " docs considered, " + str(skipped) + " docs skipped due to undersampling.")
    # Keep some stats
    with open(dataset_folder + 'dataset_description.txt', 'w') as outfile:
        outfile.write( str(considered) + " docs considered. ")
        outfile.write("These docs are all " + str(count) + " docs that have been parsed in total from " + inputFile + " ")
        outfile.write("\n Out of them, " + str(skipped) + " docs have been skipped due to undersampling of True Negative documents. ")
        outfile.write("\n ~~~~~ ")
        outfile.write("\n" + str(selected) + " docs have been selected in total, and written to the output dataset.")
        if statsOnly:
            outfile.write("\n Dataset writing was disabled as this execution aims at stats calculation about label frequencies only. ")
        elif datasetType == 'PWL':
            outfile.write("\n This dataset targets the FP/TP classification task, where the domain is the PWL documens (positive Weakly labelled). ")
            outfile.write("\n These docs are about " + str(
                keptInstanceRatio * 100) + " % of all valid articles with weak labels for any of the " + str(
                len(SCL)) + " SCL labels considered. \n")
        elif datasetType == 'NWL':
            outfile.write("\n This dataset targets the FN/TN classification task, where the domain is the NWL documens (negative Weakly labelled). ")
            outfile.write("\n These docs are about " + str(
                keptInstanceRatio * 100) + " % of all valid articles without weak labels for all " + str(
                len(SCL)) + " SCL labels considered. \n")
        outfile.write("\n ~~~~~ ")
        outfile.write("\nIn " + str(weak_filtered) + " docs at least one weak label was removed as invalid")
        outfile.write("\n" + str(invalid) + " docs are invalid for All the " + str(
                len(SCL)) + " SCL labels considered")
        outfile.write("\n" + str(golden_fully_negative) + " docs are golden_fully_negative")
        outfile.write("\n" + str(golden_positive) + " docs are golden_positive")
        outfile.write("\n" + str(weak_fully_negative) + " docs are weak_fully_negative")
        outfile.write("\n" + str(weak_positive) + " docs are weak_positive")
        outfile.write("\n" + str(FP) + " docs are noisy (FP) for at least one label")
        outfile.write("\n" + str(FN) + " docs are noisy (FN) for at least one label")
        outfile.write("\n WL freq \n\t")
        outfile.write(json.dumps(label_weak))
        outfile.write("\n fpWL freq \n\t")
        outfile.write(json.dumps(label_FP))
    return label_weak, label_FP, label_FN, label_gold, label_TN


def create_dataset_folder(taskName, add_timestamp = True):
    # The folder path to write the specific dataset
    dataset_folder = os.path.join(settings["workingPath"]+os.path.sep, taskName )
    if add_timestamp:
        dataset_folder += "_" + slugify(str(datetime.now()))
    dataset_folder += os.path.sep
    os.mkdir(dataset_folder)
    settings["datasetPath"] = dataset_folder
    # Keep a copy of the settings for reference
    with open(dataset_folder + 'settings.yaml', 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)
    return dataset_folder

def data_stats(inputFile, dataset_folder):
    fileCSV = dataset_folder + os.path.sep + inputFile[:inputFile.rfind(".")] + ".csv"
    fileStats = inputFile[:inputFile.rfind(".")] + "_stats.txt"
    # Save data stats in a log file
    with open(fileStats, 'w') as f:
        # print('Filename:', file=f)  # Python 3.x
        dataset = pd.read_csv(fileCSV)
        detailsfgNL = pd.read_csv(settings["workingPath"] + os.path.sep + settings["detailsSCL"])
        fgNL_UIs = list(detailsfgNL["Descr. UI"])
        # Total size
        print("All documents:",len(dataset), file=f)
        # Label frequency
        print("\nLabel frequency:", file=f)
        for label in fgNL_UIs:
            count = dataset[label].sum()
            print(label, count, file=f)
        # Document multi-labelness
        print("\nLabels per document:", file=f)
        total = dataset.loc[:, fgNL_UIs].sum(axis=1)
        labels_per_doc = total.value_counts()
        print(labels_per_doc, file=f)
        # Label combinations
        print("\nLabel combinations:", file=f)
        combinations = []
        for i in range(len(dataset)):
            s = dataset.iloc[i]
            a = s.index.values[(s == 1)]
            if len(a) > 1:
                combinations.append(" ".join(a))
                # print(a)
        # print(combinations)
        print([[l, combinations.count(l)] for l in set(combinations)], file=f)

label_total_weak = {}
label_noisy_FP = {}
label_noisy_FN = {}
label_total_gold = {}
label_TN = {}

with open(settings["workingPath"]+ os.path.sep + "log.txt", 'w') as outfile:

    detailsSCL = pd.read_csv(settings["workingPath"] + os.path.sep + settings["detailsSCL"])
    SCL = list(detailsSCL["Descr. UI"])
    # "Parents" actually includes all "parent descendants" as well, as anything indexed with them is implicitly indexed with the parent as well.
    SCL_parent_sets = list()
    if "Parents" in detailsSCL.keys():
        SCL_parent_sets = list(detailsSCL["Parents"])
    elif "PHex UIs" in detailsSCL.keys():
        SCL_parent_sets = list(detailsSCL["PHex UIs"])
    else:
        print("Error: No 'Parents' or 'PHex UIs' column available in detailsSCL SCV file!")
    SCL_parents = {}
    for i in range(len(SCL_parent_sets)):
        parents = SCL_parent_sets[i]
        # NaN values are considered unequal to all other values, including themselves.
        if parents == parents: # i.e. not NaN
            parents = parents.split('~')
        child = SCL[i]
        SCL_parents[child] = parents

    # print(SCL_parents)

    chunk_size = settings["chunk_size"]
    for i in range(0, len(SCL), chunk_size):
        SCL_group = SCL[i:i+chunk_size]
        # Convert the datasets into the required formats for model development
        if chunk_size <= 3:
            df = create_dataset_folder(settings["taskName"]+"_"+str("_".join(SCL_group)), False)
        else:
            df = create_dataset_folder(settings["taskName"]+"_"+str(i), False)

        detailsSCL_group = detailsSCL[detailsSCL["Descr. UI"].isin(SCL_group)]
        detailsSCL_group.to_csv(df + os.path.sep + settings["detailsSCL"], index=False)

        total_weak, noisy_FP, noisy_FN, total_gold, total_valid  = createNoiseDataset(settings, df, SCL_group, SCL_parents, outfile)
        label_total_weak.update(total_weak)
        label_noisy_FP.update(noisy_FP)
        label_noisy_FN.update(noisy_FN)
        label_total_gold.update(total_gold)
        label_TN.update(total_valid)
    label_freqs = {
        'label': label_total_weak.keys(),
        'total_weak': label_total_weak.values(),
        'fplabel': label_noisy_FP.keys(),
        'fp': label_noisy_FP.values(),
        'fnlabel': label_noisy_FN.keys(),
        'fn': label_noisy_FN.values(),
        'total_gold': label_total_gold.values(),
        'tn': label_TN.values()
    }
    df = pd.DataFrame(label_freqs)
    df.to_csv(settings["workingPath"]+ os.path.sep + "label_freq.csv", index=False)

