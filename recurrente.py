import numpy as np 
import tensorflow as tf
from laberinto import laberinto

from datetime import datetime
import io
import itertools
from packaging import version

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

#--------------------------------------------------------------------------
def plot_confusion_matrix(cm):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

logdir = "logs/plots5/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')




def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(X_trainf)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(Y_train, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
#------------------------------------------------------------------------------------------------


Y_trainf=[]
X_trainf=[]
def arrayentero(array,dim, num=None):
    Y_array=[]
    X_array=[]
    #trasformamos el array en float
    array=np.array(array,dtype='float16')
    #guardamos la dimension de la ventana en una variable
    ventana=dim
    if num is not None:
        #recorremos todos los elementos del array hasta N-longitud de la ventana y lo guardamos en un nuevo array y generamos el array de etiquetas con el num
        for i in range(0,len(array)-ventana):
            X_array.append(array[i:ventana+i])
            Y_array.append([num])
        Y_array=np.array(Y_array)
        return X_array,Y_array
    else:
        #recorremos todos los elementos del array hasta N-longitud de la ventana y lo guardamos en un nuevo array
        for i in range(0,len(array)-ventana):
            X_array.append(array[i:ventana+i,:])
        return X_array

#leyemos los datos
#-------------------------------------------------------------------------------------------------
ventana=600
X_train=np.load("datos2/1.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])


X_train=np.load("datos2/2.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])



X_train=np.load("datos2/3.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/5.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/6.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/7.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/8.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/9.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/10.npy")
aux1,aux2=arrayentero(X_train,ventana,0)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])

"""
X_train=np.load("datos2/1.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/2.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])

X_train=np.load("datos2/3.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/4.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/5.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/6.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/7.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/8.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/9.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/10.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/11.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/12.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/13.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/14.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/15.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/16.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/17.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
X_train=np.load("datos2/18.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
"""
#-------------------------------------------------------------------------------------------------
X_train=np.load("datos2/1001.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1002.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1003.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1004.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1005.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1006.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1007.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1008.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1009.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/1010.npy")
aux1,aux2=arrayentero(X_train,ventana,1)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
"""
X_train=np.load("datos2/1001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])


X_train=np.load("datos2/1002.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1003.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1004.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1005.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1006.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1007.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1008.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1009.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])

X_train=np.load("datos2/1010.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])
"""
#---------------------------------------------------------------------------------------------
X_train=np.load("datos2/2001.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2002.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2003.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2004.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2005.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2006.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2007.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2008.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2009.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/2010.npy")
aux1,aux2=arrayentero(X_train,ventana,2)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
"""
print(len(X_train))
X_train=np.load("datos2/2001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
X_train=np.load("datos2/2002.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))

X_train=np.load("datos2/2003.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
X_train=np.load("datos2/2004.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))

X_train=np.load("datos2/2005.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
X_train=np.load("datos2/2006.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))

X_train=np.load("datos2/2007.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
X_train=np.load("datos2/2008.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))

X_train=np.load("datos2/2009.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
X_train=np.load("datos2/2010.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
"""
#---------------------------------------------------------------------
X_train=np.load("datos2/3001.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3002.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3003.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3004.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3005.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3006.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3007.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3008.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3009.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/3010.npy")
aux1,aux2=arrayentero(X_train,ventana,3)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
"""
X_train=np.load("datos2/3001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3002.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3003.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3004.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3005.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3006.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3007.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3008.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3009.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3010.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
"""
#------------------------------------------------------------------------------------
X_train=np.load("datos2/4001.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4002.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4003.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4004.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4005.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4006.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4007.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4008.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4009.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
X_train=np.load("datos2/4010.npy")
aux1,aux2=arrayentero(X_train,ventana,4)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
"""
X_train=np.load("datos2/4001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4002.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4003.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4004.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4005.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4006.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4007.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4008.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4009.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4010.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
"""
#------------------------------------------------------------------------------
"""
X_train=np.load("datos2/ventana01.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana02.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana03.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana04.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana05.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana06.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana07.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana08.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/ventana09.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))"""
#-------------------------------------------------------------------------


X_train=np.load("datos2/6001.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6002.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6003.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6004.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6005.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6006.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6007.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6008.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6009.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
X_train=np.load("datos2/6010.npy")
aux1,aux2=arrayentero(X_train,ventana,5)
#aux=np.zeros(shape=[4000,2])
#aux[0:len(X_train)]=X_train
X_trainf.append(aux1[0])
Y_trainf.append(aux2[0])
print(len(X_train))
"""
X_train=np.load("datos2/6002.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6003.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6004.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6005.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6006.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6007.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6008.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6009.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6010.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
"""
#-------------------------------------------------------------------
"""
X_train=np.load("datos2/7000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([7])
print(len(X_train))
X_train=np.load("datos2/7001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([7])
print(len(X_train))
X_train=np.load("datos2/8000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([8])
print(len(X_train))
X_train=np.load("datos2/8001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([8])
print(len(X_train))
X_train=np.load("datos2/9000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([9])
print(len(X_train))
X_train=np.load("datos2/9001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([9])
print(len(X_train))
"""
"""
X_train=np.load("datos2/10000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
print(len(X_train))
X_train=np.load("datos2/10001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
"""
#capturamos los datos del usuario habitual 
#-----------------------------------------------------------------------
"""
datos=laberinto()
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
datos=laberinto()
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([0])
"""
print(len(X_trainf))


X_trainf=np.array(X_trainf)
Y_train=np.array(Y_trainf)

#print(X_trainf)
X_trainf=X_trainf/640
#print(X_trainf)

#modelo de deep learning
model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(ventana,2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GRU(units=16),
    tf.keras.layers.Dense(7,activation=tf.nn.softmax)])

"""
    tf.keras.layers.Flatten(input_shape=(ventana,2)),
    tf.keras.layers.Dense(ventana, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(ventana,activation=tf.nn.relu),
    tf.keras.layers.Dense(6,activation=tf.nn.softmax)])
"""
"""tf.keras.layers.Input(shape=(4000,2)),
    tf.keras.layers.GRU(units=32),
    tf.keras.layers.Dense(7,activation=tf.nn.softmax)

"""
"""tf.keras.layers.Input(shape=[300,2]),#el  1 es por que solo vamos a utilizar el blanco y negro si usasemos RGB 
                                                    #tendriamos que poner 3 ya que son tres capas
    tf.keras.layers.GRU(units=32),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax),#esto es para redes de clasificacion ya que nos coge el maximo y
    #se convierte en uno y  10  ya que tenemos 10 categorias"""

#compilamos el modelo 
model.compile(optimizer='adam'
               , loss='sparse_categorical_crossentropy'
               , metrics=["accuracy"]
               )
X_trainf=tf.random.shuffle(X_trainf, seed=1234)
Y_train=tf.random.shuffle(Y_train, seed=1234)
#lo entrenamos
model.fit(X_trainf,Y_train,
           epochs=100,
           #callbacks=[tensorboard_callback, cm_callback],
           validation_batch_size=0.2
           )


model.save('path_to_my_model.h5')

datos=laberinto()
datos=arrayentero(datos,ventana)
"""aux=np.zeros(shape=[4000,2])
aux[0:len(datos)]=datos"""
datos=np.array(datos)
datos=datos/640

predict=model.predict(datos)
cad=np.bincount(np.argmax(predict, axis=1)).argmax()
print(cad)
