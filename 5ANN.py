import numpy as np

x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def der_sigmoid(x):
    return x*(1-x)
    
epoc=10
lr=0.1
inputlayer_nureons=2
hiddenlayer_nureons=3
output_nureons=1

wh=np.random.uniform(size=(inputlayer_nureons,hiddenlayer_nureons))
bh=np.random.uniform(size=(1,hiddenlayer_nureons))
wout=np.random.uniform(size=(hiddenlayer_nureons,output_nureons))
bout=np.random.uniform(size=(1,output_nureons))

hinp1=np.dot(x,wh)
hinp=hinp1+bh
hlayer=sigmoid(hinp)

outinp1=np.dot(hlayer,wout)
outinp=outinp1+bout
output=sigmoid(outinp)

print("Input: ",str(x))
print("Actual Output: ",str(y))
print("Bout: ",bout)
print("Predicted Output: ",output)
