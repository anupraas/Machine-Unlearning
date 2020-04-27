import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')
MLAs = ['MLPClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'LinearSVC', 'Mixture']


def plot_for_one_dataset(_a, data_name):
    _fig, _ax = plt.subplots()
    _shards = list(_a.keys())
    _shards.sort()
    _mlas = MLAs
    _tickvalues = np.asarray([5, 10, 15])
    in_accs = {m: [] for m in _mlas}
    ens_accs = {m: [] for m in _mlas}
    for _sh in _shards:
        for m in _mlas:
            in_accs[m].append(_a[_sh][m][0])
            ens_accs[m].append(_a[_sh][m][1])
    width = 0.8
    offset = -(width * 2)
    for m in _mlas:
        _ax.bar(_tickvalues + offset, in_accs[m], width)
        offset += width
    offset = -(width * 2)
    for m in _mlas:
        _ax.bar(_tickvalues + offset, np.asarray(ens_accs[m]) - np.asarray(in_accs[m]), width, bottom=in_accs[m],
                color='black')
        offset += width
    for _sh in range(len(_shards)):
        offset = -(width * 2)
        for m in _mlas:
            increase = (ens_accs[m][_sh] - in_accs[m][_sh]) / in_accs[m][_sh]
            increase *= 100
            _ax.annotate('{}%'.format(round(increase, 1)),
                         xy=(_tickvalues[_sh] + offset, ens_accs[m][_sh]),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', size=6)
            offset += width
    _box = _ax.get_position()
    _ax.set_position([_box.x0, _box.y0, _box.width * 0.8, _box.height])
    _ax.legend(_mlas, loc='center left', bbox_to_anchor=(1, 0.5))
    _ax.set_xticks(_tickvalues)
    _ax.set_xticklabels(_shards)
    plt.xlabel('Shards')
    plt.ylabel('Accuracy')
    plt.title('Ensemble Selection on {} dataset'.format(data_name))
    plt.ylim(bottom=0, top=1)
    _fig.tight_layout()
    plt.show()


with open('compare_ensembles_cifar', 'rb') as f:
    a1 = pickle.load(f)
with open('compare_ensembles_covtype', 'rb') as f:
    a2 = pickle.load(f)
with open('compare_ensembles_mnist10', 'rb') as f:
    a3 = pickle.load(f)
with open('compare_ensembles_synth', 'rb') as f:
    a4 = pickle.load(f)

for a in [a1, a2, a3, a4]:
    for sh in a.keys():
        a[sh]['Mixture'] = a[sh]['ens']
        del a[sh]['ens']

plot_for_one_dataset(a1, 'CIFAR-10')
plot_for_one_dataset(a2, 'covertype')
plot_for_one_dataset(a3, 'MNIST')
plot_for_one_dataset(a4, 'Synthetic')

shards = list(a1.keys())
shards.sort()
mlas = MLAs
tickvalues = np.asarray([5, 10, 15])
final_to_plot = {m: [] for m in mlas}
fig, ax = plt.subplots()
for sh in shards:
    for m in mlas:
        change = 0
        change += (a1[sh][m][1] - a1[sh][m][0]) / a1[sh][m][0]
        change += (a2[sh][m][1] - a2[sh][m][0]) / a2[sh][m][0]
        change += (a3[sh][m][1] - a3[sh][m][0]) / a3[sh][m][0]
        change += (a4[sh][m][1] - a4[sh][m][0]) / a4[sh][m][0]
        final_to_plot[m].append((change / 3) * 100)
for m in mlas:
    plt.plot(tickvalues, final_to_plot[m], label=m, linewidth=3)
plt.xticks(tickvalues, shards)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(MLAs, loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Shards')
plt.ylabel('% Increase in accuracy')
plt.title('Impact of ensemble selection on accuracy')
plt.show()
