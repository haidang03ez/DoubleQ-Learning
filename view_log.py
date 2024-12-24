import matplotlib.pyplot as plt
import numpy as np

file = 'logs/old_ddqn_final.txt'

EPISODES = 50000
AVG_PER = 100


def read_log(file_name):
    with open(file_name, 'r') as f:
        logs = f.readlines()
        logs = [c.strip() for c in logs]

        list = []
        for log in logs:
            log = [float(x) for x in log.split(',')]
            list.append(log)

        list = np.array(list, dtype='float64')
        list = list.transpose()
    return list

def find_avg(list):
    length = list[2].shape[0] - list[2].shape[0] % AVG_PER
    length = length if length < EPISODES else EPISODES
    return np.sum(list[2][:length].reshape((-1, AVG_PER)), axis=1) / AVG_PER


list = read_log(file)
avg_list = find_avg(list)


plt.plot(avg_list)
plt.xlabel('Average food eaten per {} Episode'.format(str(AVG_PER)))
plt.show()