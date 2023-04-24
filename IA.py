import numpy as np 
import tensorflow as tf
from laberinto import laberinto



Y_train=[]
X_trainf=[]

#leyemos los datos

X_train=np.load("datos2/1000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])


X_train=np.load("datos2/1001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([1])


print(len(X_train))
X_train=np.load("datos2/2000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
X_train=np.load("datos2/2001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([2])
print(len(X_train))
X_train=np.load("datos2/3000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/3001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([3])
print(len(X_train))
X_train=np.load("datos2/4000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/4001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([4])
print(len(X_train))
X_train=np.load("datos2/5000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/5001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([5])
print(len(X_train))
X_train=np.load("datos2/6000.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([6])
print(len(X_train))
X_train=np.load("datos2/6001.npy")
aux=np.zeros(shape=[4000,2])
aux[0:len(X_train)]=X_train
X_trainf.append(aux)
Y_train.append([6])
print(len(X_train))
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




X_trainf=np.array(X_trainf)
Y_train=np.array(Y_train)
print(np.shape(X_trainf))
print(np.shape(Y_train))

print(X_trainf)
X_trainf=X_trainf/640
print(X_trainf)

#modelo de deep learning
modelo=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4000,2)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
	

])

"""tf.keras.layers.Input(shape=[300,2]),#el  1 es por que solo vamos a utilizar el blanco y negro si usasemos RGB 
                                                    #tendriamos que poner 3 ya que son tres capas
    tf.keras.layers.GRU(units=32),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax),#esto es para redes de clasificacion ya que nos coge el maximo y
    #se convierte en uno y  10  ya que tenemos 10 categorias"""

#compilamos el modelo 
modelo.compile(optimizer='adam'
               , loss='sparse_categorical_crossentropy'
               , metrics=["accuracy"])
X_trainf=tf.random.shuffle(X_trainf, seed=1234)
Y_train=tf.random.shuffle(Y_train, seed=1234)
#lo entrenamos
modelo.fit(X_trainf,Y_train,
           epochs=70
           )

#modelo.save('path_to_my_model.h5')

"""datos=laberinto()
aux=np.zeros(shape=[4000,2])
aux[0:len(datos)]=datos
aux=aux/640
aux=np.array([aux])
predict=modelo.predict(aux)
print(np.argmax(predict, axis=1))
print(predict)"""