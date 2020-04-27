import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy

fontP = FontProperties()
fontP.set_size('small')
MLAs = ['MLPClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'LinearSVC', 'AutoShardedClassifier']
unlearnings = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0]
SHARDS = [1, 5, 10, 20, 50]


def plot_for_one_dataset(_a, data_name):
    _shards = SHARDS
    _shards.sort()
    _mlas = MLAs
    _tickvalues = list(range(16))

    for sh in SHARDS:
        _fig, _ax = plt.subplots()
        for _m in MLAs:
            _ax.plot(_tickvalues, list(_a[sh][_m].values()))
        _box = _ax.get_position()
        _ax.set_position([_box.x0, _box.y0, _box.width * 0.8, _box.height])
        _ax.legend(_mlas, loc='center left', bbox_to_anchor=(1, 0.5))
        _ax.set_xticks(_tickvalues)
        _ax.set_xticklabels(unlearnings)
        plt.xlabel('% points unlearned')
        plt.ylabel('Accuracy')
        if sh > 1:
            plt.title('{} shards {} dataset'.format(sh, data_name))
        else:
            plt.title('{} shard {} dataset'.format(sh, data_name))
        plt.ylim(bottom=0, top=1)
        _fig.tight_layout()
        plt.show()


with open('cifar10', 'rb') as f:
    a1 = pickle.load(f)
with open('covtype', 'rb') as f:
    a2 = pickle.load(f)
with open('mnist', 'rb') as f:
    a3 = pickle.load(f)
with open('synth', 'rb') as f:
    a4 = pickle.load(f)

plot_for_one_dataset(a1, 'CIFAR-10')
plot_for_one_dataset(a2, 'covertype')
plot_for_one_dataset(a3, 'MNIST')
plot_for_one_dataset(a4, 'Synthetic')

tickvalues = list(range(16))
A = [a1, a2, a3, a4]
max_acc = {sh: {u: {a_i: None for a_i in range(len(A))} for u in unlearnings} for sh in SHARDS}
for sh in SHARDS:
    for u in unlearnings:
        for a in range(len(A)):
            max_acc[sh][u][a] = -1
            for m in MLAs:
                max_acc[sh][u][a] = max(max_acc[sh][u][a], A[a][sh][m][u])

for sh in SHARDS:
    fig, ax = plt.subplots()
    all_a = {m: [] for m in MLAs}
    for u in unlearnings:
        for m in MLAs:
            agg = 0
            for a in range(len(A)):
                agg += ((max_acc[sh][u][a] - A[a][sh][m][u])/max_acc[sh][u][a])
            all_a[m].append((100*agg)/len(A))
    for m in MLAs:
        plt.plot(tickvalues, all_a[m])
    plt.xticks(tickvalues, unlearnings)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(MLAs, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('% points unlearned')
    plt.ylabel('% Accuracy Less than Maximum')
    plt.title('Average Relative Accuracy {} shards'.format(sh))
    plt.ylim(top=40.0)
    plt.show()

for sh in SHARDS:
    fig, ax = plt.subplots()
    all_a = {m: [] for m in MLAs}
    for u in unlearnings:
        avg_rank = [0]*len(MLAs)
        for a in A:
            order = []
            for m in MLAs:
                order.append(a[sh][m][u])
            order = numpy.argsort(order)
            rank = len(MLAs)
            for i in order:
                avg_rank[i] += rank
                rank -= 1
        for i in range(len(avg_rank)):
            avg_rank[i] = avg_rank[i]/len(A)
        for i in range(len(MLAs)):
            all_a[MLAs[i]].append(avg_rank[i])
    for m in MLAs:
        plt.plot(tickvalues, all_a[m])
    plt.xticks(tickvalues, unlearnings)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(MLAs, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('% points unlearned')
    plt.ylabel('Average Rank')
    plt.title('Average Rank {} shards'.format(sh))
    plt.show()
