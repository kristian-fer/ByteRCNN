import os
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from bytercnn_models import byte_rcnn_model

SCENARIO_TO_RUN = sys.argv[1]
maxlen=512
LR = sys.argv[2]

embed_dim = 16  
BATCH_SIZE =64
kernels = [9,27,40,65]
CNN_SIZE = 64
RNN_SIZE = 64
EPOCHS = sys.argv[3]

OUTPUT = sys.argv[4]

if os.path.exists(OUTPUT) == False:
  os.makedirs(OUTPUT)

def acc_calc(row):
  if row.name == 0: return '-'
  correct = int(row[row.name])
  total = sum([int(x) for x in row[1:]])
  try:
    return (correct / total) * 100.0
  except:
    pass
  return 0

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


print("\n\n")
print("="*90)
print("="*90)
x_train, y_train, labels = load(SCENARIO_TO_RUN, '4k' if maxlen == 4096 else str(maxlen), 'train')
print("Loaded data: x.shape={}, y.shape={}".format(x_train.shape, y_train.shape))

print("Exampel label: {}".format(y_train[0]))

X_val, y_val, labels_val = load(SCENARIO_TO_RUN, '4k' if maxlen == 4096 else str(maxlen), 'val')
print("Validation data: x.shape={}, y.shape={}".format(X_val.shape,y_val.shape))

print("TRAIN: {}".format( len(x_train) ))
print("VAL: {}".format( len(X_val) ))

output_cnt = len( labels )
print("LABELS CNT: {}".format( output_cnt ))
print("="*90)
print("="*90)
print("\n\n")

model = byte_rcnn_model(maxlen, embed_dim, RNN_SIZE, CNN_SIZE, kernels, output_cnt, OUTPUT, LR)

history = model.fit(
    x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS , validation_data=(X_val, y_val)
    , callbacks=[
            keras.callbacks.ModelCheckpoint(OUTPUT+"best_model_", save_best_only=True)
    ]
)

# plot history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.savefig(OUTPUT+'train_hist_acc_rcnn.png', dpi=400)
plt.clf()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig(OUTPUT+'train_hist_loss_rcnn.png', dpi=400)
plt.clf()

x_test, y_test, labels_val = load(SCENARIO_TO_RUN, '4k' if maxlen == 4096 else str(maxlen), 'test')
results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("test loss, test acc:", results)

y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred, axis=-1)


y_pred = np.argmax(y_pred, axis=-1)
y_pred = [ labels_val[a] for a in y_pred]
y_test = [ labels_val[a] for a in y_test]

report = classification_report(y_test, y_pred, target_names=labels_val, output_dict=True)
clas_report = pd.DataFrame(report)
clas_report.to_excel(OUTPUT + 'classification_report_bytercnn_sc{}.xlsx'.format(SCENARIO_TO_RUN))

conf_matrix = confusion_matrix(y_test, y_pred, labels=labels_val)
# save as excel
labels_left = np.array(labels_val).reshape(-1, 1)
conf_matrix = np.concatenate([labels_left, conf_matrix], axis=1)

conf_matrix = np.vstack([['-'] + labels_val, conf_matrix])

conf_matrix = pd.DataFrame(conf_matrix)
conf_matrix['acc'] = conf_matrix.apply(lambda row: acc_calc(row), axis=1)
conf_matrix.to_excel(OUTPUT + 'bytercnn_sc{}_confusion_matrix.xlsx'.format(SCENARIO_TO_RUN) )

print("acc: {}".format(accuracy_score(y_test, y_pred)))
print("f1_score: {}".format(f1_score(y_test, y_pred, average='micro')))
print("precision_score: {}".format(precision_score(y_test, y_pred, average='micro')))
print("recall_score: {}".format(recall_score(y_test, y_pred, average='micro')))

report = classification_report(y_test, y_pred, target_names=labels_val, output_dict=True)
clas_report = pd.DataFrame(report).transpose()
clas_report.to_excel('classification_report_bytercnn.xlsx')


model.save(OUTPUT+'_bytercnn_len{}_sc{}_model_save'.format(maxlen, SCENARIO_TO_RUN))
