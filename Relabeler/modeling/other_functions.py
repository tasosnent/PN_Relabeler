import json, csv
import pandas as pd

# Auxiliary functions for model development and evaluation

def read_data(training_csv, target_label, label_name):
    '''
    Read data from a CSV file and return prepare a dataset for the target_labels as a dataframe.
    :param training_csv:
    :param target_labels:
    :return:
    '''
    prompts = pd.read_csv(training_csv)
    prompts.head()

    # input_data = prompts[['pmid', target_label, 'text']]
    fields = ['pmid', 'text']

    # print(fields)
    input_data = prompts[fields]
    input_data['target'] = prompts[target_label]
    input_data['label'] = [label_name] * len(prompts)


    # make sure all values are string
    input_data.loc[:, 'pmid'] = input_data['pmid'].astype("str")
    input_data.loc[:, 'text'] = input_data['text'].astype("str")
    input_data.loc[:, 'label'] = input_data['label'].astype("str")

    # make sure there are no missing values
    input_data = input_data[~input_data['pmid'].isin([pd.NA])]
    # for target_label in target_labels:
    #     input_data = input_data[~input_data[target_label].isin([pd.NA])]
    # input_data = input_data[~input_data[target_label].isin([pd.NA])]
    input_data = input_data[~input_data['text'].isin([pd.NA])]
    input_data = input_data[~input_data['label'].isin([pd.NA])]
    # remove data for out-of-focus labels
    # print("dataset fields: ",fields)
    # print("dataset head: ")
    # print(input_data.head(2))
    # print("dataset first sample: ")

    return input_data

def save_evaluation_report(report_json, report_json_file, report_csv_file, matrix = None):
    # open the file in the write mode
    if report_json_file is not None:
        with open(report_json_file, 'w') as outfile:
            json.dump(report_json, outfile)

    new_json = {}

    if report_csv_file is not None:
        with open(report_csv_file, 'w', newline='', encoding='utf-8') as f:
            # create the csv writer
            writer = csv.writer(f)
            header_row = ["label"] + list(report_json['macro avg'].keys())
            if matrix is not None:
                header_row = header_row + ["tn","fp","fn","tp"]
            writer.writerow(header_row)
            for row in report_json:
                new_row = report_json[row]
                if "avg" in row:
                    # This is a row like "micro avg", "macro avg" etc
                    content_row = [row] + list(report_json[row].values())
                    if matrix is not None:
                        content_row = content_row + [" "," "," "," "]
                elif "accuracy" in row:
                    # This is an "accuracy" row shown in binary classification
                    content_row = [row, " "," "] + [report_json[row]] + [" "]
                    if matrix is not None:
                        content_row = content_row + [" "," "," "," "]
                else:
                    # This is a label row
                    label_name = row # for binary cases "0.0" and "0.1" for negative and positive instances
                    content_row = [label_name] + list(report_json[row].values())
                    if matrix is not None:
                        index = 1
                        content_row = content_row + [matrix[index][0][0], matrix[index][0][1],matrix[index][1][0],matrix[index][1][1]]
                        new_row['tn'] = int(matrix[index][0][0])
                        new_row['fp'] = int(matrix[index][0][1])
                        new_row['fn'] = int(matrix[index][1][0])
                        new_row['tp'] = int(matrix[index][1][1])
                        new_json[row] = new_row
                # print("content_row ", content_row)
                # print("new_json", new_json)
                # write a row to the csv file
                writer.writerow(content_row)
        if report_json_file is not None:
            with open(report_json_file, 'w') as outfile:
                json.dump(new_json, outfile)
