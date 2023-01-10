import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MSE = tf.keras.losses.MeanSquaredError()
def mse_loss():
    y = W*x_batch**2 + b
    return MSE(y, t_batch)
##    return tf.reduce_mean(tf.square(y - t_batch))

train_data = np.array([
     [-0.5,4],
     [-0.4,3],
     [-0.3,2.2],
     [-0.2,1.6],
     [0,1],
     [0.1,1],
     [0.2,1.2],
     [0.3,1.6],
     [0.4,2.2],
     [0.5,3]], dtype= np.float32)
X=train_data[:,:-1]
t=train_data[:,-1:]

tf.random.set_seed(1)
W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
##opt = tf.keras.optimizers.Adagrad(0.01)
##opt = tf.keras.optimizers.Adam(0.01) 
##opt = tf.keras.optimizers.RMSprop(0.01)

train_size = 10
batch_size = 2
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

plt.scatter(X, t) # 주어진 training data 점으로 표
w_pred, b_pred = W.numpy(),b.numpy()
t_pred = tf.pow(X,2) * w_pred + b_pred
plt.plot(X, t_pred, '-r') # 예측 값 선으로 잇기
plt.show()


X= -1
t_pred =  W.numpy() * X**2 + b.numpy()
print("t_pred = ", t_pred)
