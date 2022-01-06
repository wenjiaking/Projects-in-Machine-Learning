import sys
import numpy as np
from environment import MountainCar

def state_mode(mode,s_dict,n_state):
    if (mode=="raw"):
        s=np.array(list(s_dict.values()))
        return(s)
    else:
        s=np.zeros((n_state))
        s[list(s_dict.keys())]=1
        return(s)


def q_learning (mode,w_out,r_out,epis,max_iter,eps,gamma,lr):
    epis=int(epis)
    max_iter=int(max_iter)
    eps=float(eps)
    gamma=float(gamma)
    lr=float(lr)
    env=MountainCar(mode)
    n_state=env.state_space
    n_action=env.action_space
    w=np.zeros((n_state,n_action),dtype=np.longdouble)
    b=0
    rewards_sum=np.zeros((epis,1),dtype=np.longdouble)
    
    for i in np.arange(epis):
        reward_cum=0
        for j in np.arange(max_iter):
            s_dict=env.transform(env.state)
            s=state_mode(mode,s_dict,n_state)
            q=np.dot(s,w)+b
            rand=np.random.binomial(1,eps,1)[0]
            if (rand==0):
                a=np.argmax(q)
            else:
                a=np.random.randint(n_action, size=1)[0]
            
            s1_dict,reward,terminate=env.step(a)
            s1=state_mode(mode,s1_dict,n_state)
            q1=np.dot(s1,w)+b
            w[:,a] -= lr*(q[a]-reward-gamma*np.max(q1))*s
            b -= lr*(q[a]-reward-gamma*np.max(q1))
            reward_cum += reward
            if (terminate==True):
                break
        
        s_dict=env.reset()   
           
        rewards_sum[i,0]=reward_cum
    
    pars=np.insert(w.reshape((n_state*n_action,1)),0,b,axis=0)
    np.savetxt(w_out,pars,fmt="%f")
    np.savetxt(r_out,rewards_sum,fmt="%f")
    #return(rewards_sum)

#raw_rewards=q_learning ("raw","weight_out","returns_out",2000,200,0.05,0.999,0.001)
#tile_rewards=q_learning ("tile","weight_out","returns_out",400,200,0.05,0.99,0.00005)
def rolling_mean(a, n=25) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#raw_roll=rolling_mean(raw_rewards)
#tile_roll=rolling_mean(tile_rewards)
#import matplotlib.pyplot as plt
#plt. clf()
#plt.plot(np.arange(2000),raw_rewards,"r--",label="returns")
#plt.plot(np.arange(2000)[24:],raw_roll,"b--",label="rolling mean")
#plt.xlabel("episode")
#plt.title("raw features")
#plt.ylabel("sum of rewards")
#plt.legend()
#plt.savefig("Q1.4.1a.pdf")

#plt. clf()
#plt.plot(np.arange(400),tile_rewards,"r--",label="returns")
#plt.plot(np.arange(400)[24:],tile_roll,"b--",label="rolling mean")
#plt.xlabel("episode")
#plt.title("tile features")
#plt.ylabel("sum of rewards")
#plt.legend()
#plt.savefig("Q1.4.1b.pdf")


  

          
if __name__ =='__main__':
    mode=sys.argv[1]
    w_out=sys.argv[2]
    r_out=sys.argv[3]
    epis=sys.argv[4]
    max_iter=sys.argv[5]
    eps=sys.argv[6]
    gamma=sys.argv[7]
    lr=sys.argv[8]
    q_learning (mode,w_out,r_out,epis,max_iter,eps,gamma,lr)
