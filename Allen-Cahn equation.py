import tensorflow as tf
import numpy as np

dims=100
T=0.3
N=20
deltaT=T/N
std=0.0  #biaozhuicha

x=tf.zeros([dims],dtype=tf.float64)
u0=tf.Variable(1.0,dtype=tf.float64)
deltau0=tf.Variable(tf.random_normal([dims],stddev=std,dtype=tf.float64))
W1=tf.Variable(tf.random_normal([N-1,dims,dims+10],stddev=std,dtype=tf.float64))
W2=tf.Variable(tf.random_normal([N-1,dims+10,dims+10],stddev=std,dtype=tf.float64))
W3=tf.Variable(tf.random_normal([N-1,dims+10,dims],stddev=std,dtype=tf.float64))
B=tf.Variable(tf.random_normal([N-1,dims],stddev=std,dtype=tf.float64))


batch_size=64



deltaW=tf.random_normal([batch_size,N,dims],stddev=np.sqrt(0.3/20),dtype=tf.float64)
u1=u0-(u0-u0**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau0,deltaW[:,0,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,0,:]
xx1=tf.reshape(x,[batch_size,dims])
digits11=tf.matmul(xx1,W1[0])
bat11=tf.layers.batch_normalization(digits11,axis=-1,training=True)
y11=tf.nn.relu(bat11)
digits21=tf.matmul(y11,W2[0])
bat21=tf.layers.batch_normalization(digits21,axis=-1,training=True)
y21=tf.nn.relu(bat21)
y31=tf.matmul(y21,W3[0])+B[0]
deltau1=y31



u2=u1-(u1-u1**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau1,deltaW[:,1,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,1,:]
xx2=tf.reshape(x,[batch_size,dims])
digits12=tf.matmul(xx2,W1[1])
bat12=tf.layers.batch_normalization(digits12,axis=-1,training=True)
y12=tf.nn.relu(digits12)
digits22=tf.matmul(y12,W2[1])
bat22=tf.layers.batch_normalization(digits22,axis=-1,training=True)
y22=tf.nn.relu(digits22)
y32=tf.matmul(y22,W3[1])+B[1]
deltau2=y32


u3=u2-(u2-u2**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau2,deltaW[:,2,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,2,:]
xx3=tf.reshape(x,[batch_size,dims])
digits13=tf.matmul(xx3,W1[2])
bat13=tf.layers.batch_normalization(digits13,axis=-1,training=True)
y13=tf.nn.relu(digits13)
digits23=tf.matmul(y13,W2[2])
bat23=tf.layers.batch_normalization(digits23,axis=-1,training=True)
y23=tf.nn.relu(digits23)
y33=tf.matmul(y23,W3[2])+B[2]
deltau3=y33


u4=u3-(u3-u3**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau3,deltaW[:,3,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,3,:]
xx4=tf.reshape(x,[batch_size,dims])
digits14=tf.matmul(xx4,W1[3])
bat14=tf.layers.batch_normalization(digits14,axis=-1,training=True)
y14=tf.nn.relu(digits14)
digits24=tf.matmul(y14,W2[3])
bat24=tf.layers.batch_normalization(digits24,axis=-1,training=True)
y24=tf.nn.relu(digits24)
y34=tf.matmul(y24,W3[3])+B[3]
deltau4=y34


u5=u4-(u4-u4**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau4,deltaW[:,4,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,4,:]
xx5=tf.reshape(x,[batch_size,dims])
digits15=tf.matmul(xx5,W1[4])
bat15=tf.layers.batch_normalization(digits15,axis=-1,training=True)
y15=tf.nn.relu(digits15)
digits25=tf.matmul(y15,W2[4])
bat25=tf.layers.batch_normalization(digits25,axis=-1,training=True)
y25=tf.nn.relu(digits25)
y35=tf.matmul(y25,W3[4])+B[4]
deltau5=y35


u6=u5-(u5-u5**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau5,deltaW[:,5,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,5,:]
xx6=tf.reshape(x,[batch_size,dims])
digits16=tf.matmul(xx6,W1[5])
bat16=tf.layers.batch_normalization(digits16,axis=-1,training=True)
y16=tf.nn.relu(digits16)
digits26=tf.matmul(y16,W2[5])
bat26=tf.layers.batch_normalization(digits26,axis=-1,training=True)
y26=tf.nn.relu(digits26)
y36=tf.matmul(y26,W3[5])+B[5]
deltau6=y36


u7=u6-(u6-u6**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau6,deltaW[:,6,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,6,:]
xx7=tf.reshape(x,[batch_size,dims])
digits17=tf.matmul(xx7,W1[6])
bat17=tf.layers.batch_normalization(digits17,axis=-1,training=True)
y17=tf.nn.relu(digits17)
digits27=tf.matmul(y17,W2[6])
bat27=tf.layers.batch_normalization(digits27,axis=-1,training=True)
y27=tf.nn.relu(digits27)
y37=tf.matmul(y27,W3[6])+B[6]
deltau7=y37


u8=u7-(u7-u7**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau7,deltaW[:,7,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,7,:]
xx8=tf.reshape(x,[batch_size,dims])
digits18=tf.matmul(xx8,W1[7])
bat18=tf.layers.batch_normalization(digits18,axis=-1,training=True)
y18=tf.nn.relu(digits18)
digits28=tf.matmul(y18,W2[7])
bat28=tf.layers.batch_normalization(digits28,axis=-1,training=True)
y28=tf.nn.relu(digits28)
y38=tf.matmul(y28,W3[7])+B[7]
deltau8=y38


u9=u8-(u8-u8**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau8,deltaW[:,8,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,8,:]
xx9=tf.reshape(x,[batch_size,dims])
digits19=tf.matmul(xx9,W1[8])
bat19=tf.layers.batch_normalization(digits19,axis=-1,training=True)
y19=tf.nn.relu(digits19)
digits29=tf.matmul(y19,W2[8])
bat29=tf.layers.batch_normalization(digits29,axis=-1,training=True)
y29=tf.nn.relu(digits29)
y39=tf.matmul(y29,W3[8])+B[8]
deltau9=y39


u10=u9-(u9-u9**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau9,deltaW[:,9,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,9,:]
xx10=tf.reshape(x,[batch_size,dims])
digits110=tf.matmul(xx10,W1[9])
bat110=tf.layers.batch_normalization(digits110,axis=-1,training=True)
y110=tf.nn.relu(digits110)
digits210=tf.matmul(y110,W2[9])
bat210=tf.layers.batch_normalization(digits210,axis=-1,training=True)
y210=tf.nn.relu(digits210)
y310=tf.matmul(y210,W3[9])+B[9]
deltau10=y310


u11=u10-(u10-u10**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau10,deltaW[:,10,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,10,:]
xx11=tf.reshape(x,[batch_size,dims])
digits111=tf.matmul(xx11,W1[10])
bat111=tf.layers.batch_normalization(digits111,axis=-1,training=True)
y111=tf.nn.relu(digits111)
digits211=tf.matmul(y111,W2[10])
bat211=tf.layers.batch_normalization(digits211,axis=-1,training=True)
y211=tf.nn.relu(digits211)
y311=tf.matmul(y211,W3[10])+B[10]
deltau11=y311


u12=u11-(u11-u11**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau11,deltaW[:,11,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,11,:]
xx12=tf.reshape(x,[batch_size,dims])
digits112=tf.matmul(xx12,W1[11])
bat112=tf.layers.batch_normalization(digits112,axis=-1,training=True)
y112=tf.nn.relu(digits112)
digits212=tf.matmul(y112,W2[11])
bat212=tf.layers.batch_normalization(digits212,axis=-1,training=True)
y212=tf.nn.relu(digits212)
y312=tf.matmul(y212,W3[11])+B[11]
deltau12=y312


u13=u12-(u12-u12**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau12,deltaW[:,12,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,12,:]
xx13=tf.reshape(x,[batch_size,dims])
digits113=tf.matmul(xx13,W1[12])
bat113=tf.layers.batch_normalization(digits113,axis=-1,training=True)
y113=tf.nn.relu(digits113)
digits213=tf.matmul(y113,W2[12])
bat213=tf.layers.batch_normalization(digits213,axis=-1,training=True)
y213=tf.nn.relu(digits213)
y313=tf.matmul(y213,W3[12])+B[12]
deltau13=y313


u14=u13-(u13-u13**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau13,deltaW[:,13,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,13,:]
xx14=tf.reshape(x,[batch_size,dims])
digits114=tf.matmul(xx14,W1[13])
bat114=tf.layers.batch_normalization(digits114,axis=-1,training=True)
y114=tf.nn.relu(digits114)
digits214=tf.matmul(y114,W2[13])
bat214=tf.layers.batch_normalization(digits214,axis=-1,training=True)
y214=tf.nn.relu(digits214)
y314=tf.matmul(y214,W3[13])+B[13]
deltau14=y314


u15=u14-(u14-u14**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau14,deltaW[:,14,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,14,:]
xx15=tf.reshape(x,[batch_size,dims])
digits115=tf.matmul(xx15,W1[14])
bat115=tf.layers.batch_normalization(digits115,axis=-1,training=True)
y115=tf.nn.relu(digits115)
digits215=tf.matmul(y115,W2[14])
bat215=tf.layers.batch_normalization(digits215,axis=-1,training=True)
y215=tf.nn.relu(digits215)
y315=tf.matmul(y215,W3[14])+B[14]
deltau15=y315


u16=u15-(u15-u15**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau15,deltaW[:,15,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,15,:]
xx16=tf.reshape(x,[batch_size,dims])
digits116=tf.matmul(xx16,W1[15])
bat116=tf.layers.batch_normalization(digits116,axis=-1,training=True)
y116=tf.nn.relu(digits116)
digits216=tf.matmul(y116,W2[15])
bat216=tf.layers.batch_normalization(digits216,axis=-1,training=True)
y216=tf.nn.relu(digits216)
y316=tf.matmul(y216,W3[15])+B[15]
deltau16=y316


u17=u16-(u16-u16**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau16,deltaW[:,16,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,16,:]
xx17=tf.reshape(x,[batch_size,dims])
digits117=tf.matmul(xx17,W1[16])
bat117=tf.layers.batch_normalization(digits117,axis=-1,training=True)
y117=tf.nn.relu(digits117)
digits217=tf.matmul(y117,W2[16])
bat217=tf.layers.batch_normalization(digits217,axis=-1,training=True)
y217=tf.nn.relu(digits217)
y317=tf.matmul(y217,W3[16])+B[16]
deltau17=y317


u18=u17-(u17-u17**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau17,deltaW[:,17,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,17,:]
xx18=tf.reshape(x,[batch_size,dims])
digits118=tf.matmul(xx18,W1[17])
bat118=tf.layers.batch_normalization(digits118,axis=-1,training=True)
y118=tf.nn.relu(digits118)
digits218=tf.matmul(y118,W2[17])
bat218=tf.layers.batch_normalization(digits218,axis=-1,training=True)
y218=tf.nn.relu(digits218)
y318=tf.matmul(y218,W3[17])+B[17]
deltau18=y318


u19=u18-(u18-u18**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau18,deltaW[:,18,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,18,:]
xx19=tf.reshape(x,[batch_size,dims])
digits119=tf.matmul(xx19,W1[18])
bat119=tf.layers.batch_normalization(digits119,axis=-1,training=True)
y119=tf.nn.relu(digits119)
digits219=tf.matmul(y119,W2[18])
bat219=tf.layers.batch_normalization(digits219,axis=-1,training=True)
y219=tf.nn.relu(digits219)
y319=tf.matmul(y219,W3[18])+B[18]
deltau19=y319


u20=u19-(u19-u19**3.0)*deltaT+np.sqrt(2.0)*tf.reduce_sum(tf.multiply(deltau19,deltaW[:,19,:]),axis=1)
x=x+np.sqrt(2.0)*deltaW[:,19,:]



y_=1.0/(2.0+0.4*tf.reduce_sum(tf.square(x),axis=1))
cost=tf.losses.mean_squared_error(u20,y_)
optimizer=tf.train.AdamOptimizer(0.005).minimize(cost)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

loss=np.array([])
ans=np.array([])
for i in range(2000):
    sess.run(optimizer)
    ans=np.append(ans,sess.run(u0))
    loss=np.append(loss,sess.run(cost))
    print(i,sess.run(u0))
