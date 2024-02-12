import matplotlib.pyplot as plt 
import numpy as np


def visualize_exp1():
    infos = {
        'california': True,
        'black-friday': True,
        'churn': False,
        'adult': False,
        'otto': False,
    }

    def load(num, data_name, is_regression):
        arr = np.loadtxt(f'./log/exp1/{num}/{data_name}.log')
        if is_regression:
            return np.mean(arr)
        else:
            return arr
    print('=================================')
    print('===== RESULT : EXPERIMENT 1 ========')
    print('=================================')
    for data_name, is_regression in infos.items():
        log = [load(num, data_name, is_regression) for num in range(5)]
        if is_regression:
            print(f'Regression | {data_name} \nloss: {np.sqrt(np.mean(log))}')
        else:
            acc, l = np.max(log,0)
            print(f'Classif | {data_name} \nloss: {l}\naccuracy: {acc}')
        print("==============================")
            
def visualize_exp2():
    print('=================================')
    print('===== RESULT : EXPERIMENT 2 ========')
    print('=================================')


    percs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.42]

    max_num = 5
    for num in range(max_num):
        S = [np.loadtxt(f'./log/exp2/{num}_{i}.log') for i in range(2)]
        S = np.array(S)
        plt.plot(np.arange(num+1, len(percs)+1) , S.mean(0), 'o-', alpha = (1/max_num)*(num+1))

    plt.grid()
    plt.xticks(np.arange(1,len(percs)+1), [f'{int(i*100)} %' for i in percs])
    plt.ylabel('RMSE', fontsize=18)
    plt.xlabel('%'+" de la base d'entraînement utilisée comme candidats pour la prédiction", fontsize=18)
    plt.savefig("./result_img/sample_augmentation.png")
    plt.show()


if __name__ == '__main__':
    visualize_exp1()
    #visualize_exp2()

