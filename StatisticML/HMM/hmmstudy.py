import math

import numpy as np
from hmmlearn import hmm

start_probability = np.array([0.2, 0.4, 0.4])
transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])
emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

def bulidmultiHMM(n_state):
    model = hmm.MultinomialHMM(n_components=n_state, n_iter=20, tol=0.001)
    model.startprob_=start_probability
    model.transmat_=transition_probability
    model.emissionprob_=emission_probability
    return model

def Question10_1(states,observations):
    start_probability = np.array([0.2, 0.4, 0.4])
    transition_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
        ])
    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
        ])
    o_see=seen = np.array([[0,1,0,1]]).T
    model10_1=bulidmultiHMM(len(states))
    boxlist=model10_1.predict(o_see)
    prediction10_1=model10_1.score(o_see)
    print("10.1,10.3:")
    print("维特比算法最优路径为：",list(map(lambda x: states[x], boxlist)))
    print("模型下观测序列概率为：",math.exp(prediction10_1))

def Question10_2(states,observations):
    start_probability = np.array([0.2, 0.3, 0.5])
    transition_probability = np.array([
        [0.5, 0.1, 0.4],
        [0.3, 0.5, 0.2],
        [0.2, 0.2, 0.6]
        ])
    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
        ])
    o_see=seen = np.array([[0,1,0,0,1,0,1,1]]).T
    model10_2=bulidmultiHMM(len(states))
    prediction10_2=model10_2.predict_proba(o_see)
    print("10.2:")
    print("模型下观测序列概率为：",prediction10_2[3][2])

def main():
    states = ["box 1", "box 2", "box3"]
    observations = ["red", "white"]
    Question10_1(states,observations)
    Question10_2(states,observations)

if __name__ == '__main__':
    main()
