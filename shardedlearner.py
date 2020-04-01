import copy


class ShardedClassifier:
    num_shards = 1
    # X_train = None
    # y_train = None
    init_num_samples = None
    init_shard_size = None
    shard_dict = {}
    ml_algorithm = None
    model = None
    shards = None
    map_point_to_shard = {}

    def __init__(self, ml_algorithm, num_shards=1):
        self.num_shards = num_shards
        self.ml_algorithm = ml_algorithm

    def fit(self, X, y):
        # self.X_train = {i: copy.deepcopy(X[i]) for i in range(len(X))}
        # self.y_train = {i: copy.deepcopy(y[i]) for i in range(len(y))}
        self.init_num_samples = len(y)
        self.init_shard_size = len(y)//self.num_shards
        shard_num = 0
        for i in range(0, len(X), self.init_shard_size):
            X_cur_list = X[i:i + self.init_shard_size]
            y_cur_list = y[i:i + self.init_shard_size]
            self.shard_dict[self.getShardId(i)] = {'X': X_cur_list, 'y': y_cur_list}
            self.map_point_to_shard[i] = shard_num

    def predict(self, X):
        return self.model.predict(X)

    def unlearn(self, X):
        pass

    def get_shard_id(self, idx):
        return idx//self.init_shard_size
