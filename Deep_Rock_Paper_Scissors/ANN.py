import tensorflow as tf
import cv2
import numpy as np
import time



# Variables

def weight_init(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_init(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def avg_pool_4x4(x):
    return tf.nn.avg_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')


X = tf.placeholder(tf.float32,[None,640,480,3])
pool1 = avg_pool_4x4(X)
pool2 = avg_pool_4x4(pool1)

WF1 = tf.Variable(tf.truncated_normal([3600,900],stddev=0.1), name = 'ANN_WF1')
BF1 = tf.Variable(tf.constant(0.1,[900]),name = 'ANN_BF1')
flatten = tf.reshape(pool2,[-1,3600])
fc1 = tf.nn.relu(tf.matmul(flatten, WF1) + BF1)
fc1_drop = tf.nn.dropout(fc1, 0.75)

WF2 = tf.Variable(tf.truncated_normal([900,3], stddev=0.1), name = 'ANN_WF2')
BF2 = tf.Variable(tf.constant(0.1,[3]),name = 'ANN_BF2')
result = tf.nn.softmax(tf.matmul(fc1_drop, WF2) + BF2)


# Now evaluation
Y = tf.placeholder(tf.float32,[None,3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(result), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
comp_pred = tf.equal(tf.arg_max(result,1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype=tf.float32))



# Create Saver

sess = tf.InteractiveSession()
cost_set = []
accuracy_set = []

print("Start loading...")
inputs = []
val = []
for i in range(1,1301):
    inputs.append((cv2.imread("D:/data/rock (%d).jpg"%i),(1,0,0)))
    if i%100 == 0:
        print("Loading Rock(%d/1400)"%i)
print("Loading Rock done")
for i in range(1,1301):
    inputs.append((cv2.imread("D:/data/paper (%d).jpg"%i),(0,1,0)))
    if i%100 == 0:
        print("Loading Paper(%d/1400)"%i)
print("Loading Paper done")
for i in range(1,1301):
    inputs.append((cv2.imread("D:/data/scissors (%d).jpg"%i),(0,0,1)))
    if i%100 == 0:
        print("Loading Scissors(%d/1400)"%i)
print("Loading Scissors done")
    
print("Loading validation set")
for i in range(1301,1401):
    val.append((cv2.imread("D:/data/rock (%d).jpg"%i),(1,0,0)))
    val.append((cv2.imread("D:/data/paper (%d).jpg"%i),(0,1,0)))
    val.append((cv2.imread("D:/data/scissors (%d).jpg"%i),(0,0,1)))
print("Loading done")

print("Start training..")
sess.run(tf.global_variables_initializer())

param_list=[WC1,BC1,WC2,BC2,WF1,BF1,WF2,BF2]
saver = tf.train.Saver()

for i in range(3001):
    in_X = []
    in_Y = []
    val_X = []
    val_Y = []
    for step in range(32):
        index = random.randint(0,3899)
        in_X.append(inputs[index][0])
        in_Y.append(inputs[index][1])
        index = random.randint(0,99)
        val_X.append(val[index][0])
        val_Y.append(val[index][1])

    sess.run(train_step,feed_dict={X:in_X,Y:in_Y})
    
    cost,accu = sess.run([cross_entropy,accuracy],feed_dict={X:val_X,Y:val_Y})

    accuracy_set.append(accu)
    cost_set.append(cost)
    if(i % 20 == 0):
        print("step %d, accuracy = %f, cost = %f"%(i,accu,cost))
        if(i%100 ==0):
            saver.save(sess,'./save/ANN_model',global_step = i)
            if i%1000==0:
                plt.figure()
                plt.plot(accuracy_set)
                plt.show()
                plt.figure()
                plt.plot(cost_set)
                plt.show()
                accuracy_set.clear()
                cost_set.clear()

