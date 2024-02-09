import matplotlib.pyplot as plt 
import numpy as np

percs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.42]

for num in range(4):
    S = [np.loadtxt(f'./log/{num}_{i}.log') for i in range(2)]
    S = np.array(S)
    plt.plot(np.arange(num+1, len(percs)+1) , S.mean(0), 'o-', alpha = (1/len(percs))*(num+1), color='b')

plt.grid()
plt.xticks(np.arange(1,len(percs)+1), [f'{int(i*100)} %' for i in percs])
plt.ylabel('RMSE')
plt.show()
