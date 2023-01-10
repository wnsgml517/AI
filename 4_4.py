import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MSE = tf.keras.losses.MeanSquaredError()
def mse_loss():
    y = tf.matmul(x_batch,W)+b
    return MSE(y, t_batch)
##    return tf.reduce_mean(tf.square(y - t_batch))

train_data = np.array([ [-4,1,-1,1],
                        [-2,0,-1,1],
                        [-1,1,-1,4],
                        [0,0,1,-3],
                        [1,1,1,0],
                        [2,0,1,-1],
                        [3,1,1,2],
                        [4,0,1,1]], dtype= np.float32)

X=train_data[:,:-1]
t=train_data[:,-1:]

tf.random.set_seed(1)
W = tf.Variable(tf.random.normal(shape=[3,1]))
b = tf.Variable(tf.random.normal(shape=[1]))

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
##opt = tf.keras.optimizers.Adagrad(0.01)
##opt = tf.keras.optimizers.Adam(0.01) 
##opt = tf.keras.optimizers.RMSprop(0.01)

train_size = 8
batch_size = 4
K = train_size// batch_size

loss_list = [ ]
for epoch in range(1000):
    batch_loss = 0.0
    for  step  in range(K):
        mask = np.random.choice(train_size, batch_size)
        x_batch = X[mask]
        t_batch = t[mask]
        
        opt.minimize(mse_loss, var_list= [W, b])
        loss = mse_loss().numpy()
        batch_loss += loss
        
    batch_loss /= K # average loss
    loss_list.append(batch_loss)
##    if not epoch % 100:
##            print ("epoch={}: batch_loss={:.5f}".format(epoch, batch_loss))	

                    
print ("W={}, b={}, loss={}".format(
        W.numpy(), b.numpy(), batch_loss))	
plt.plot(loss_list)
plt.show()

x1=5
x2=1
x3=-1

y=x1*W[0].numpy()+x2*W[1].numpy()+ x3*W[2].numpy()+b.numpy()

print ("y={}, loss={}".format(
        y, batch_loss))	


