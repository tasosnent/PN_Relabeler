""" Script for selecting Fine Grained Labels (FGLs) for PN Relabeling dataset development. """
import yaml
from datetime import datetime
import pandas as pd

# Read the settings
settings_file = open("settings_FGL_selection.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

dataset_path = settings['dataset_path']
# Read FGL
descriptors = pd.read_csv(settings['detailsSCL'])
print(len(descriptors))

print(descriptors.columns)

# Read freq
freqs = pd.read_csv(settings['label_freq'])
print(len(freqs))

print(freqs.columns)
print(freqs.head)

descriptors.count()

freqs.count()

"""Merge the data"""

data = pd.merge(freqs, descriptors, how='left', left_on='label', right_on='Descr. UI')
data.count()

# Find check tags
data[data['TreeNumbers'].isnull()]

# Delete check tags
data= data[~data['TreeNumbers'].isnull()]
data.count()

# Find publication characteristics (Category V is not used as MeSH tags)
data[data['Categories'].str.contains('V')]

# Delete publication characteristics
data= data[~data['Categories'].str.contains('V')]

data.count()

threshold = 50
print("PP>=1 ", len(data[data['total_weak'] >= 1]))
print("GP>=1 ", len(data[data['total_gold'] >= 1]))
print("FP>=1 ", len(data[data['fp'] >= 1]))
print("FN>=1 ", len(data[data['fn'] >= 1]))
print("PP>="+str(threshold)+" ", len(data[data['total_weak'] >= threshold]))
print("GP>="+str(threshold)+" ", len(data[data['total_gold'] >= threshold]))
print("FP>="+str(threshold)+" ", len(data[data['fp'] >= threshold]))
print("FN>="+str(threshold)+" ", len(data[data['fn'] >= threshold]))
print("FP>="+str(threshold)+" & FN>="+str(threshold)+": ", len(data[(data['fp'] >= threshold) & (data['fn'] >= threshold) ]))

data['fp_ratio'] = data['fp']/data['total_weak']
data['precision'] = 1- data['fp_ratio']
data['fn_ratio'] = data['fn']/data['total_gold']
data['recall'] = 1- data['fn_ratio']
data['precise'] = round(data['precision'])
data['low_recall'] = 1- round(data['recall'])
data['tp'] = data['total_weak']-data['fp']
data['f1'] = (2 * data['precision'] * data['recall'])/ (data['precision'] + data['recall'])

print("Precision in [0.4, 0.6]: ", len(data[(data['precision'] >= 0.4) & (data['precision'] <= 0.6)]))
print("Recall in [0.4, 0.6]: ", len(data[(data['recall'] >= 0.4) & (data['recall'] <= 0.6)]))
print("Precision and Recall in [0.4, 0.6]: ", len(data[(data['precision'] >= 0.4) & (data['precision'] <= 0.6) & (data['recall'] >= 0.4) & (data['recall'] <= 0.6)]))

FP_dataset = data[(data['precision'] >= 0.4) & (data['precision'] <= 0.6) & (data['fp'] >= threshold) ]
FP_dataset.count()

FN_dataset = data[(data['recall'] >= 0.4) & (data['recall'] <= 0.6) & (data['fn'] >= threshold)]
FN_dataset.count()

"""Store selected SCL sets"""

FP_dataset.to_csv(dataset_path + '/FP_SCL_selected.csv', index=False)
FN_dataset.to_csv(dataset_path + '/FN_SCL_selected.csv', index=False)
