import numpy as np

for i in range(2,21):
    print('u'+str(i)+'=u'+str(i-1)+'-(u'+str(i-1)+'-u'+str(i-1)+'**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau'+str(i-1)+',deltaW[:,'+str(i-1)+',:]),axis=1)')
    print('x=x+np.sqrt(2.0)*deltaW[:,'+str(i-1)+',:]')
    print('xx'+str(i)+'=tf.reshape(x,[batch_size,dims])')
    print('digits1'+str(i)+'=tf.matmul(xx'+str(i)+',W1['+str(i-1)+'])')
    print('bat1'+str(i)+'=tf.layers.batch_normalization(digits1'+str(i)+',axis=-1,training=True)')
    print('y1'+str(i)+'=tf.nn.relu(digits1'+str(i)+')')
    print('digits2'+str(i)+'=tf.matmul(y1'+str(i)+',W2['+str(i-1)+'])')
    print('bat2'+str(i)+'=tf.layers.batch_normalization(digits2'+str(i)+',axis=-1,training=True)')
    print('y2'+str(i)+'=tf.nn.relu(digits2'+str(i)+')')
    print('y3'+str(i)+'=tf.matmul(y2'+str(i)+',W3['+str(i-1)+'])+B['+str(i-1)+']')
    print('deltau'+str(i)+'=y3'+str(i))
    print('\n')
    