import tensorflow as tf
import numpy as np

dims=100
T=1.0
N=40
deltaT=T/N
std=0.0  #biaozhuicha

x=100*tf.ones([100],dtype=tf.float64)
u0=tf.Variable(58.5,dtype=tf.float64)
deltau0=tf.Variable(tf.random_normal([dims],stddev=std,dtype=tf.float64))
W1=tf.Variable(tf.random_normal([N-1,dims,dims+10],stddev=std,dtype=tf.float64))
B1=tf.Variable(tf.random_normal([N-1,dims+10],stddev=std,dtype=tf.float64))
W2=tf.Variable(tf.random_normal([N-1,dims+10,dims+10],stddev=std,dtype=tf.float64))
B2=tf.Variable(tf.random_normal([N-1,dims+10],stddev=std,dtype=tf.float64))
W3=tf.Variable(tf.random_normal([N-1,dims+10,dims],stddev=std,dtype=tf.float64))
B3=tf.Variable(tf.random_normal([N-1,dims],stddev=std,dtype=tf.float64))


batch_size=64


def g(x):
    return tf.where(tf.less(x,50.0),0.2*tf.ones_like(x),tf.where(tf.greater(x,70.0),0.02*tf.ones_like(x),0.2+(x-50.0)*(0.02-0.2)/(70.0-50.0)))

#deltaW=tf.random_normal([batch_size,N,dims],stddev=std)   #mean=0,biaozhuicha=stddev
deltaW=tf.random_normal([batch_size,N,dims],stddev=np.sqrt(deltaT),dtype=tf.float64)
u1=u0+(g(u0)/3.0+0.02)*u0*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(deltau0,x),deltaW[:,0,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,0,:])
xx1=tf.reshape(x,[batch_size,dims])
digits11=tf.matmul(xx1,W1[0])
###remember to add batch normalizatino
y11=tf.nn.relu(digits11)
digits21=tf.matmul(y11,W2[0])
y21=tf.nn.relu(digits21)
y31=tf.matmul(y21,W3[0])
deltau1=y31


u2=u1+(g(u1)/3.0+0.02)*u1*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau1,(64,100)),x),deltaW[:,1,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,1,:])
xx2=tf.reshape(x,[batch_size,-1,dims])
digits12=tf.matmul(xx2,W1[1])+B1[1]
###remember to add batch normalizatino
y12=tf.nn.relu(digits12)
digits22=tf.matmul(y12,W2[1])+B2[1]
y22=tf.nn.relu(digits22)
y32=tf.matmul(y22,W3[1])+B3[1]
deltau2=y32


u3=u2+(g(u2)/3.0+0.02)*u2*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau2,(64,100)),x),deltaW[:,2,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,2,:])
xx3=tf.reshape(x,[batch_size,-1,dims])
digits13=tf.matmul(xx3,W1[2])+B1[2]
###remember to add batch normalizatino
y13=tf.nn.relu(digits13)
digits23=tf.matmul(y13,W2[2])+B2[2]
y23=tf.nn.relu(digits23)
y33=tf.matmul(y23,W3[2])+B3[2]
deltau3=y33


u4=u3+(g(u3)/3.0+0.02)*u3*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau3,(64,100)),x),deltaW[:,3,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,3,:])
xx4=tf.reshape(x,[batch_size,-1,dims])
digits14=tf.matmul(xx4,W1[3])+B1[3]
###remember to add batch normalizatino
y14=tf.nn.relu(digits14)
digits24=tf.matmul(y14,W2[3])+B2[3]
y24=tf.nn.relu(digits24)
y34=tf.matmul(y24,W3[3])+B3[3]
deltau4=y34


u5=u4+(g(u4)/3.0+0.02)*u4*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau4,(64,100)),x),deltaW[:,4,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,4,:])
xx5=tf.reshape(x,[batch_size,-1,dims])
digits15=tf.matmul(xx5,W1[4])+B1[4]
###remember to add batch normalizatino
y15=tf.nn.relu(digits15)
digits25=tf.matmul(y15,W2[4])+B2[4]
y25=tf.nn.relu(digits25)
y35=tf.matmul(y25,W3[4])+B3[4]
deltau5=y35


u6=u5+(g(u5)/3.0+0.02)*u5*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau5,(64,100)),x),deltaW[:,5,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,5,:])
xx6=tf.reshape(x,[batch_size,-1,dims])
digits16=tf.matmul(xx6,W1[5])+B1[5]
###remember to add batch normalizatino
y16=tf.nn.relu(digits16)
digits26=tf.matmul(y16,W2[5])+B2[5]
y26=tf.nn.relu(digits26)
y36=tf.matmul(y26,W3[5])+B3[5]
deltau6=y36


u7=u6+(g(u6)/3.0+0.02)*u6*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau6,(64,100)),x),deltaW[:,6,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,6,:])
xx7=tf.reshape(x,[batch_size,-1,dims])
digits17=tf.matmul(xx7,W1[6])+B1[6]
###remember to add batch normalizatino
y17=tf.nn.relu(digits17)
digits27=tf.matmul(y17,W2[6])+B2[6]
y27=tf.nn.relu(digits27)
y37=tf.matmul(y27,W3[6])+B3[6]
deltau7=y37


u8=u7+(g(u7)/3.0+0.02)*u7*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau7,(64,100)),x),deltaW[:,7,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,7,:])
xx8=tf.reshape(x,[batch_size,-1,dims])
digits18=tf.matmul(xx8,W1[7])+B1[7]
###remember to add batch normalizatino
y18=tf.nn.relu(digits18)
digits28=tf.matmul(y18,W2[7])+B2[7]
y28=tf.nn.relu(digits28)
y38=tf.matmul(y28,W3[7])+B3[7]
deltau8=y38


u9=u8+(g(u8)/3.0+0.02)*u8*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau8,(64,100)),x),deltaW[:,8,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,8,:])
xx9=tf.reshape(x,[batch_size,-1,dims])
digits19=tf.matmul(xx9,W1[8])+B1[8]
###remember to add batch normalizatino
y19=tf.nn.relu(digits19)
digits29=tf.matmul(y19,W2[8])+B2[8]
y29=tf.nn.relu(digits29)
y39=tf.matmul(y29,W3[8])+B3[8]
deltau9=y39


u10=u9+(g(u9)/3.0+0.02)*u9*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau9,(64,100)),x),deltaW[:,9,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,9,:])
xx10=tf.reshape(x,[batch_size,-1,dims])
digits110=tf.matmul(xx10,W1[9])+B1[9]
###remember to add batch normalizatino
y110=tf.nn.relu(digits110)
digits210=tf.matmul(y110,W2[9])+B2[9]
y210=tf.nn.relu(digits210)
y310=tf.matmul(y210,W3[9])+B3[9]
deltau10=y310


u11=u10+(g(u10)/3.0+0.02)*u10*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau10,(64,100)),x),deltaW[:,10,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,10,:])
xx11=tf.reshape(x,[batch_size,-1,dims])
digits111=tf.matmul(xx11,W1[10])+B1[10]
###remember to add batch normalizatino
y111=tf.nn.relu(digits111)
digits211=tf.matmul(y111,W2[10])+B2[10]
y211=tf.nn.relu(digits211)
y311=tf.matmul(y211,W3[10])+B3[10]
deltau11=y311


u12=u11+(g(u11)/3.0+0.02)*u11*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau11,(64,100)),x),deltaW[:,11,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,11,:])
xx12=tf.reshape(x,[batch_size,-1,dims])
digits112=tf.matmul(xx12,W1[11])+B1[11]
###remember to add batch normalizatino
y112=tf.nn.relu(digits112)
digits212=tf.matmul(y112,W2[11])+B2[11]
y212=tf.nn.relu(digits212)
y312=tf.matmul(y212,W3[11])+B3[11]
deltau12=y312


u13=u12+(g(u12)/3.0+0.02)*u12*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau12,(64,100)),x),deltaW[:,12,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,12,:])
xx13=tf.reshape(x,[batch_size,-1,dims])
digits113=tf.matmul(xx13,W1[12])+B1[12]
###remember to add batch normalizatino
y113=tf.nn.relu(digits113)
digits213=tf.matmul(y113,W2[12])+B2[12]
y213=tf.nn.relu(digits213)
y313=tf.matmul(y213,W3[12])+B3[12]
deltau13=y313


u14=u13+(g(u13)/3.0+0.02)*u13*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau13,(64,100)),x),deltaW[:,13,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,13,:])
xx14=tf.reshape(x,[batch_size,-1,dims])
digits114=tf.matmul(xx14,W1[13])+B1[13]
###remember to add batch normalizatino
y114=tf.nn.relu(digits114)
digits214=tf.matmul(y114,W2[13])+B2[13]
y214=tf.nn.relu(digits214)
y314=tf.matmul(y214,W3[13])+B3[13]
deltau14=y314


u15=u14+(g(u14)/3.0+0.02)*u14*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau14,(64,100)),x),deltaW[:,14,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,14,:])
xx15=tf.reshape(x,[batch_size,-1,dims])
digits115=tf.matmul(xx15,W1[14])+B1[14]
###remember to add batch normalizatino
y115=tf.nn.relu(digits115)
digits215=tf.matmul(y115,W2[14])+B2[14]
y215=tf.nn.relu(digits215)
y315=tf.matmul(y215,W3[14])+B3[14]
deltau15=y315


u16=u15+(g(u15)/3.0+0.02)*u15*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau15,(64,100)),x),deltaW[:,15,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,15,:])
xx16=tf.reshape(x,[batch_size,-1,dims])
digits116=tf.matmul(xx16,W1[15])+B1[15]
###remember to add batch normalizatino
y116=tf.nn.relu(digits116)
digits216=tf.matmul(y116,W2[15])+B2[15]
y216=tf.nn.relu(digits216)
y316=tf.matmul(y216,W3[15])+B3[15]
deltau16=y316


u17=u16+(g(u16)/3.0+0.02)*u16*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau16,(64,100)),x),deltaW[:,16,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,16,:])
xx17=tf.reshape(x,[batch_size,-1,dims])
digits117=tf.matmul(xx17,W1[16])+B1[16]
###remember to add batch normalizatino
y117=tf.nn.relu(digits117)
digits217=tf.matmul(y117,W2[16])+B2[16]
y217=tf.nn.relu(digits217)
y317=tf.matmul(y217,W3[16])+B3[16]
deltau17=y317


u18=u17+(g(u17)/3.0+0.02)*u17*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau17,(64,100)),x),deltaW[:,17,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,17,:])
xx18=tf.reshape(x,[batch_size,-1,dims])
digits118=tf.matmul(xx18,W1[17])+B1[17]
###remember to add batch normalizatino
y118=tf.nn.relu(digits118)
digits218=tf.matmul(y118,W2[17])+B2[17]
y218=tf.nn.relu(digits218)
y318=tf.matmul(y218,W3[17])+B3[17]
deltau18=y318


u19=u18+(g(u18)/3.0+0.02)*u18*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau18,(64,100)),x),deltaW[:,18,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,18,:])
xx19=tf.reshape(x,[batch_size,-1,dims])
digits119=tf.matmul(xx19,W1[18])+B1[18]
###remember to add batch normalizatino
y119=tf.nn.relu(digits119)
digits219=tf.matmul(y119,W2[18])+B2[18]
y219=tf.nn.relu(digits219)
y319=tf.matmul(y219,W3[18])+B3[18]
deltau19=y319


u20=u19+(g(u19)/3.0+0.02)*u19*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau19,(64,100)),x),deltaW[:,19,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,19,:])
xx20=tf.reshape(x,[batch_size,-1,dims])
digits120=tf.matmul(xx20,W1[19])+B1[19]
###remember to add batch normalizatino
y120=tf.nn.relu(digits120)
digits220=tf.matmul(y120,W2[19])+B2[19]
y220=tf.nn.relu(digits220)
y320=tf.matmul(y220,W3[19])+B3[19]
deltau20=y320


u21=u20+(g(u20)/3.0+0.02)*u20*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau20,(64,100)),x),deltaW[:,20,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,20,:])
xx21=tf.reshape(x,[batch_size,-1,dims])
digits121=tf.matmul(xx21,W1[20])+B1[20]
###remember to add batch normalizatino
y121=tf.nn.relu(digits121)
digits221=tf.matmul(y121,W2[20])+B2[20]
y221=tf.nn.relu(digits221)
y321=tf.matmul(y221,W3[20])+B3[20]
deltau21=y321


u22=u21+(g(u21)/3.0+0.02)*u21*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau21,(64,100)),x),deltaW[:,21,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,21,:])
xx22=tf.reshape(x,[batch_size,-1,dims])
digits122=tf.matmul(xx22,W1[21])+B1[21]
###remember to add batch normalizatino
y122=tf.nn.relu(digits122)
digits222=tf.matmul(y122,W2[21])+B2[21]
y222=tf.nn.relu(digits222)
y322=tf.matmul(y222,W3[21])+B3[21]
deltau22=y322


u23=u22+(g(u22)/3.0+0.02)*u22*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau22,(64,100)),x),deltaW[:,22,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,22,:])
xx23=tf.reshape(x,[batch_size,-1,dims])
digits123=tf.matmul(xx23,W1[22])+B1[22]
###remember to add batch normalizatino
y123=tf.nn.relu(digits123)
digits223=tf.matmul(y123,W2[22])+B2[22]
y223=tf.nn.relu(digits223)
y323=tf.matmul(y223,W3[22])+B3[22]
deltau23=y323


u24=u23+(g(u23)/3.0+0.02)*u23*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau23,(64,100)),x),deltaW[:,23,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,23,:])
xx24=tf.reshape(x,[batch_size,-1,dims])
digits124=tf.matmul(xx24,W1[23])+B1[23]
###remember to add batch normalizatino
y124=tf.nn.relu(digits124)
digits224=tf.matmul(y124,W2[23])+B2[23]
y224=tf.nn.relu(digits224)
y324=tf.matmul(y224,W3[23])+B3[23]
deltau24=y324


u25=u24+(g(u24)/3.0+0.02)*u24*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau24,(64,100)),x),deltaW[:,24,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,24,:])
xx25=tf.reshape(x,[batch_size,-1,dims])
digits125=tf.matmul(xx25,W1[24])+B1[24]
###remember to add batch normalizatino
y125=tf.nn.relu(digits125)
digits225=tf.matmul(y125,W2[24])+B2[24]
y225=tf.nn.relu(digits225)
y325=tf.matmul(y225,W3[24])+B3[24]
deltau25=y325


u26=u25+(g(u25)/3.0+0.02)*u25*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau25,(64,100)),x),deltaW[:,25,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,25,:])
xx26=tf.reshape(x,[batch_size,-1,dims])
digits126=tf.matmul(xx26,W1[25])+B1[25]
###remember to add batch normalizatino
y126=tf.nn.relu(digits126)
digits226=tf.matmul(y126,W2[25])+B2[25]
y226=tf.nn.relu(digits226)
y326=tf.matmul(y226,W3[25])+B3[25]
deltau26=y326


u27=u26+(g(u26)/3.0+0.02)*u26*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau26,(64,100)),x),deltaW[:,26,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,26,:])
xx27=tf.reshape(x,[batch_size,-1,dims])
digits127=tf.matmul(xx27,W1[26])+B1[26]
###remember to add batch normalizatino
y127=tf.nn.relu(digits127)
digits227=tf.matmul(y127,W2[26])+B2[26]
y227=tf.nn.relu(digits227)
y327=tf.matmul(y227,W3[26])+B3[26]
deltau27=y327


u28=u27+(g(u27)/3.0+0.02)*u27*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau27,(64,100)),x),deltaW[:,27,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,27,:])
xx28=tf.reshape(x,[batch_size,-1,dims])
digits128=tf.matmul(xx28,W1[27])+B1[27]
###remember to add batch normalizatino
y128=tf.nn.relu(digits128)
digits228=tf.matmul(y128,W2[27])+B2[27]
y228=tf.nn.relu(digits228)
y328=tf.matmul(y228,W3[27])+B3[27]
deltau28=y328


u29=u28+(g(u28)/3.0+0.02)*u28*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau28,(64,100)),x),deltaW[:,28,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,28,:])
xx29=tf.reshape(x,[batch_size,-1,dims])
digits129=tf.matmul(xx29,W1[28])+B1[28]
###remember to add batch normalizatino
y129=tf.nn.relu(digits129)
digits229=tf.matmul(y129,W2[28])+B2[28]
y229=tf.nn.relu(digits229)
y329=tf.matmul(y229,W3[28])+B3[28]
deltau29=y329


u30=u29+(g(u29)/3.0+0.02)*u29*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau29,(64,100)),x),deltaW[:,29,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,29,:])
xx30=tf.reshape(x,[batch_size,-1,dims])
digits130=tf.matmul(xx30,W1[29])+B1[29]
###remember to add batch normalizatino
y130=tf.nn.relu(digits130)
digits230=tf.matmul(y130,W2[29])+B2[29]
y230=tf.nn.relu(digits230)
y330=tf.matmul(y230,W3[29])+B3[29]
deltau30=y330


u31=u30+(g(u30)/3.0+0.02)*u30*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau30,(64,100)),x),deltaW[:,30,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,30,:])
xx31=tf.reshape(x,[batch_size,-1,dims])
digits131=tf.matmul(xx31,W1[30])+B1[30]
###remember to add batch normalizatino
y131=tf.nn.relu(digits131)
digits231=tf.matmul(y131,W2[30])+B2[30]
y231=tf.nn.relu(digits231)
y331=tf.matmul(y231,W3[30])+B3[30]
deltau31=y331


u32=u31+(g(u31)/3.0+0.02)*u31*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau31,(64,100)),x),deltaW[:,31,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,31,:])
xx32=tf.reshape(x,[batch_size,-1,dims])
digits132=tf.matmul(xx32,W1[31])+B1[31]
###remember to add batch normalizatino
y132=tf.nn.relu(digits132)
digits232=tf.matmul(y132,W2[31])+B2[31]
y232=tf.nn.relu(digits232)
y332=tf.matmul(y232,W3[31])+B3[31]
deltau32=y332


u33=u32+(g(u32)/3.0+0.02)*u32*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau32,(64,100)),x),deltaW[:,32,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,32,:])
xx33=tf.reshape(x,[batch_size,-1,dims])
digits133=tf.matmul(xx33,W1[32])+B1[32]
###remember to add batch normalizatino
y133=tf.nn.relu(digits133)
digits233=tf.matmul(y133,W2[32])+B2[32]
y233=tf.nn.relu(digits233)
y333=tf.matmul(y233,W3[32])+B3[32]
deltau33=y333


u34=u33+(g(u33)/3.0+0.02)*u33*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau33,(64,100)),x),deltaW[:,33,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,33,:])
xx34=tf.reshape(x,[batch_size,-1,dims])
digits134=tf.matmul(xx34,W1[33])+B1[33]
###remember to add batch normalizatino
y134=tf.nn.relu(digits134)
digits234=tf.matmul(y134,W2[33])+B2[33]
y234=tf.nn.relu(digits234)
y334=tf.matmul(y234,W3[33])+B3[33]
deltau34=y334


u35=u34+(g(u34)/3.0+0.02)*u34*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau34,(64,100)),x),deltaW[:,34,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,34,:])
xx35=tf.reshape(x,[batch_size,-1,dims])
digits135=tf.matmul(xx35,W1[34])+B1[34]
###remember to add batch normalizatino
y135=tf.nn.relu(digits135)
digits235=tf.matmul(y135,W2[34])+B2[34]
y235=tf.nn.relu(digits235)
y335=tf.matmul(y235,W3[34])+B3[34]
deltau35=y335


u36=u35+(g(u35)/3.0+0.02)*u35*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau35,(64,100)),x),deltaW[:,35,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,35,:])
xx36=tf.reshape(x,[batch_size,-1,dims])
digits136=tf.matmul(xx36,W1[35])+B1[35]
###remember to add batch normalizatino
y136=tf.nn.relu(digits136)
digits236=tf.matmul(y136,W2[35])+B2[35]
y236=tf.nn.relu(digits236)
y336=tf.matmul(y236,W3[35])+B3[35]
deltau36=y336


u37=u36+(g(u36)/3.0+0.02)*u36*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau36,(64,100)),x),deltaW[:,36,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,36,:])
xx37=tf.reshape(x,[batch_size,-1,dims])
digits137=tf.matmul(xx37,W1[36])+B1[36]
###remember to add batch normalizatino
y137=tf.nn.relu(digits137)
digits237=tf.matmul(y137,W2[36])+B2[36]
y237=tf.nn.relu(digits237)
y337=tf.matmul(y237,W3[36])+B3[36]
deltau37=y337


u38=u37+(g(u37)/3.0+0.02)*u37*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau37,(64,100)),x),deltaW[:,37,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,37,:])
xx38=tf.reshape(x,[batch_size,-1,dims])
digits138=tf.matmul(xx38,W1[37])+B1[37]
###remember to add batch normalizatino
y138=tf.nn.relu(digits138)
digits238=tf.matmul(y138,W2[37])+B2[37]
y238=tf.nn.relu(digits238)
y338=tf.matmul(y238,W3[37])+B3[37]
deltau38=y338


u39=u38+(g(u38)/3.0+0.02)*u38*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau38,(64,100)),x),deltaW[:,38,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,38,:])
xx39=tf.reshape(x,[batch_size,-1,dims])
digits139=tf.matmul(xx39,W1[38])+B1[38]
###remember to add batch normalizatino
y139=tf.nn.relu(digits139)
digits239=tf.matmul(y139,W2[38])+B2[38]
y239=tf.nn.relu(digits239)
y339=tf.matmul(y239,W3[38])+B3[38]
deltau39=y339


u40=u39+(g(u39)/3.0+0.02)*u39*deltaT+0.2*tf.reduce_sum(tf.multiply(tf.multiply(tf.reshape(deltau39,(64,100)),x),deltaW[:,39,:]),axis=1)
x=x+0.02*deltaT*x+0.2*tf.multiply(x,deltaW[:,39,:])



y_=tf.reduce_min(x,reduction_indices=1)
cost=tf.losses.mean_squared_error(u40,y_)
optimizer=tf.train.AdamOptimizer(0.008).minimize(cost)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

loss=np.array([])
ans=np.array([])
for i in range(2000):
#    print(sess.run(cost))
    sess.run(optimizer)
    loss=np.append(loss,sess.run(cost))
    ans=np.append(ans,sess.run(u0))
    print(i,sess.run(u0))
    