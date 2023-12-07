import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import math

unk_label = 'unknown'

SCENARIO_TO_RUN = sys.argv[1]
MODEL_TO_LOAD = sys.argv[2]
OUTPUT= sys.argv[3]

FRAGMENTS_EXPERT_DATA = sys.argv[4]
FRAGMENTS_EXPERT_LABELS = sys.argv[5]

ALPHA = sys.argv[6]
THRESHOLD = sys.argv[7]

if os.path.exists(OUTPUT) == False:
  os.makedirs(OUTPUT)

BATCH_SIZE = 128

maxlen=512

def acc_calc(row):
  if row.name == 0: return '-'
  correct = int(row[row.name])
  total = sum([int(x) for x in row[1:]])
  try:
    return (correct / total) * 100.0
  except:
    pass
  return 0

def openmax(probabilities, alpha=10):
  open_probabilities = np.zeros((probabilities.shape[0], probabilities.shape[1] + 1))
  for i in range(probabilities.shape[0]):
    p_known = probabilities[i]
    sum_known = np.sum(p_known)
    p_unknown = np.maximum(0, (p_known - alpha * (p_known / sum_known)))
    p_unknown /= np.sum(p_unknown)
    open_probabilities[i, :-1] = p_known
    open_probabilities[i, -1] = np.max(p_unknown)
  return open_probabilities

def load(scenario=1, block_size='4k', subset='train'):
  if block_size not in ['512', '4k']:
    raise ValueError('Invalid block size!')
  if scenario not in range(1, 7):
    raise ValueError('Invalid scenario!')
  if subset not in ['train', 'val', 'test']:
    raise ValueError('Invalid subset!')

  data_dir = os.path.join('../', '{:s}_{:1d}'.format(block_size, scenario))
  data = np.load(os.path.join(data_dir, '{}.npz'.format(subset)))

  if os.path.isfile('../classes.json'):
    with open('../classes.json') as json_file:
      classes = json.load(json_file)
      labels = classes[str(scenario)]
  else:
    raise FileNotFoundError('Please download classes.json to the current directory!')

  return data['x'], data['y'], labels

_, _, labels = load(1, '512', 'train')


x_test_init = pd.read_csv(FRAGMENTS_EXPERT_DATA, header=None, sep=',')
y_test_init = pd.read_csv(FRAGMENTS_EXPERT_LABELS, header=None, sep=',')

new_dataset_labels = set([row[0].lower() for index, row in y_test_init.iterrows()])

x_test=[]
y_test=[]

lab_check = set(labels)
unkown_counter = 0
for index, row in x_test_init.iterrows():
  if sum(row.to_list()) > 0:

    lab = y_test_init.iloc[index].to_list()[0].lower()

    if '-' in lab:
      lab = lab.split('-')[0]

    if 'png' in lab:
      lab = 'png'

    if 'jpeg' in lab:
      lab = 'jpg'

    if lab in labels:
      x_test.append(row.to_list())
      y_test.append( lab )
    else:
      x_test.append(row.to_list())
      y_test.append(unk_label)
      unkown_counter += 1

print("Samples total: {}".format(len(y_test)))
print("Samples unknown: {}".format(unkown_counter))

x_test = np.array(x_test)
y_test_base = np.array(y_test)

model = tf.keras.models.load_model(MODEL_TO_LOAD)
#model.summary()
logits_output = []
print("Predicting...")
y_pred_probs = model.predict(x_test, batch_size=BATCH_SIZE)
print("Predicted")
labels = labels + [unk_label]

open_probabilities = openmax(y_pred_probs, alpha=ALPHA)

predictions_class = np.argmax(open_probabilities[:, :-1], axis=-1)
predictions_class[open_probabilities[:, -1] > THRESHOLD] = -1

y_pred = predictions_class

print("Converting")
y_pred = [ labels[a] if a >= 0 else unk_label for a in y_pred ]
cor_labs = list(set(y_test_base))

labels_val =  list(set(y_test_base)) +[unk_label] 

print("Running report ...")

report = classification_report(y_test, y_pred, output_dict=True)
clas_report = pd.DataFrame(report)
clas_report.to_excel(OUTPUT + 'classification_report_fragmexpert_th{}.xlsx'.format(THRESHOLD))

print("ALPHA: {}".format(ALPHA))
print("Threshold: {}".format(THRESHOLD))

print("acc: {}".format(accuracy_score(y_test, y_pred)))
print("f1_score: {}".format(f1_score(y_test, y_pred, average='micro')))
print("precision_score: {}".format(precision_score(y_test, y_pred, average='micro')))
print("recall_score: {}".format(recall_score(y_test, y_pred, average='micro')))


conf_matrix = confusion_matrix(y_test,y_pred, labels=labels )

# save as excel
labels_left = np.array(labels).reshape(-1,1)
conf_matrix = np.concatenate([labels_left, conf_matrix], axis=1)

conf_matrix = np.vstack([['-'] + labels, conf_matrix])

conf_matrix = pd.DataFrame(conf_matrix)
conf_matrix['acc'] = conf_matrix.apply(lambda row: acc_calc(row), axis=1)
conf_matrix.to_excel(OUTPUT + 'fifty_confusion_matrix_fragmexpert_th{}.xlsx'.format(THRESHOLD) )

