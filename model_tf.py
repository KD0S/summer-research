import tensorflow as tf 
import tensorflow
import matplotlib.pyplot as plt
import preprocessor
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM,Bidirectional,ConvLSTM1D,MaxPooling1D,Conv1D
from tensorflow.compat.v1.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

preprocess = preprocessor.Preprocessing('./data/aug_datasets/aug_data_80_protected_words.csv')
preprocess.load_data()
x_train = preprocess.x_train
y_train = preprocess.y_train
x_test = preprocess.x_test
y_test = preprocess.y_test


model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))

model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(64,return_sequences=False)))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(2e-4),
              metrics=['accuracy'])

model.build([8, 1, 256])
print(model.summary())

filepath="./weights_best_cnn.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
callbacks_list = [checkpoint]
history_embedding = model.fit(x_train, y_train, epochs=10, batch_size=16,verbose = 1,callbacks = callbacks_list, validation_data=(x_test,y_test))

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
mc = ModelCheckpoint('./modelbigru.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score

# plt.plot(history_embedding.history['accuracy'],c='b',label='train accuracy')
# plt.plot(history_embedding.history['loss'],c='r',label='loss')
# plt.legend(loc='lower right')
# plt.show()

# plt.plot(history_embedding.history['accuracy'],c='b',label='train accuracy')
# plt.plot(history_embedding.history['val_accuracy'],c='r',label='validation accuracy')
# plt.legend(loc='lower right')
# plt.show()

# plt.plot(history_embedding.history['val_accuracy'],c='b',label='val accuracy')
# plt.plot(history_embedding.history['val_loss'],c='r',label='val loss')
# plt.legend(loc='lower right')
# plt.show()

y_pred = model.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# np.savetxt('no_aug_fpr_1.txt', fpr)
# np.savetxt('no_aug_tpr_1.txt', tpr)

import time

start_time=time.time()
y_pred = model.predict(x_test)
end_time=time.time()
duration= end_time-start_time
print("inference time : ", duration)

auc = roc_auc_score(y_test, model.predict(x_test))
print(auc)


plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print(model.evaluate(x_test, y_test))
y_pred = np.where(model.predict(x_test)>.5,1,0)
from sklearn import metrics
print(metrics.classification_report(y_pred, y_test))


print("Precision Score : ", precision_score(y_test, y_pred))
print("Recall Score : ", recall_score(y_test, y_pred))
print("F1 Score : ", f1_score(y_test, y_pred))
print("Accuracy Score : ", accuracy_score(y_test, y_pred))


