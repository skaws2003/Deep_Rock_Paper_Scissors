import tensorflow as tf
import cv2


# Camera initialization
cap = cv2.VideoCapture(1)

# Make variables
X = tf.placeholder(tf.float32,[None,480,640,3])

WC1 = tf.Variable(tf.truncated_normal([3,3,3,6], stddev = 0.1),name='WC1')
BC1 = tf.Variable(tf.ones([6])/10, name='BC1')
WC2 = tf.Variable(tf.truncated_normal([3,3,6,12], stddev = 0.1), name='WC2')
BC2 = tf.Variable(tf.ones([12])/10, name='BC2')

WF1 = tf.Variable(tf.truncated_normal([40*30*12,225], stddev = 0.1), name='WF1')
BF1 = tf.Variable(tf.ones([225])/10, name = 'BF1')
WF2 = tf.Variable(tf.truncated_normal([225,3], stddev=0.1), name='WF2')
BF2 = tf.Variable(tf.ones([3])/10, name='BF2')

# Operations
conv1 = tf.nn.relu(tf.nn.conv2d(X,WC1, strides=[1,2,2,1], padding='SAME') + BC1)
conv2 = tf.nn.relu(tf.nn.conv2d(conv1,WC2,strides=[1,2,2,1],padding='SAME')+BC2)
pooling1 = tf.nn.max_pool(conv2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
flatten = tf.reshape(pooling1,[-1,40*30*12])
fc1 = tf.nn.relu(tf.matmul(flatten,WF1)+BF1)
fc1_drop = tf.nn.dropout(fc1,0.6)
semi_result = tf.matmul(fc1_drop,WF2)+BF2
result = tf.nn.softmax(semi_result)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load Model
print("Loading model...")
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./save'))
print("Load done")

while True:
    if cv2.waitKey(10)>0 : break;
    _,frame = cap.read();
    res = sess.run(result,feed_dict={X:[frame]})
    print(res)
    cv2.imshow("test",frame)
    
    
