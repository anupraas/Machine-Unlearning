import pickle
import numpy as np

with open('covtype_compare_ens5_part_1', 'rb') as f:
	a=pickle.load(f)
# print(a)
for num_shard in a.keys():
	print(num_shard)
	for mla in a[num_shard].keys():
		print('\t{}:'.format(mla))
		print(a[num_shard][mla])
