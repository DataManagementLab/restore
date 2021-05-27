import logging
from time import perf_counter

import nmslib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ANNIndex:

    def __init__(self, sampling_threshold):
        self.labels = None
        self.sampling_threshold = sampling_threshold
        self.imputer = None
        self.scaler = None
        self.assigned_idx = set()

    def transform(self, data):
        data = data.astype('float32')
        if self.imputer is None:
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(data)
        data = self.imputer.transform(data)

        if self.scaler is None:
            self.scaler = StandardScaler().fit(data)
        data = self.scaler.transform(data)
        data = np.ascontiguousarray(data)
        return data

    def fit(self, labels, data):
        if len(labels) > self.sampling_threshold:
            idx = np.random.choice(len(labels), self.sampling_threshold)
            labels = labels[idx]
            data = data[idx]

        self.labels = labels
        start_t = perf_counter()
        data = self.transform(data)

        self.fit_index(data)
        assert len(data) == len(labels)

        logger.info(f"Created index for {len(labels)} tuples in {perf_counter() - start_t:.2f} secs")

    def query_batch(self, data, n=15):
        data = self.transform(data)

        index, distance = self.query_index(data, n)
        if n > 1:
            index_ = index[:, 0]
            distance_ = distance[:, 0]

            for i in range(index.shape[0]):
                if index_[i] not in self.assigned_idx:
                    self.assigned_idx.add(index_[i])
                    continue
                for j in range(index.shape[1]):
                    if index[i, j] not in self.assigned_idx:
                        index_[i] = index[i, j]
                        distance_[i] = distance[i, j]
                        self.assigned_idx.add(index_[i])
                        break

            index = index_
            distance = distance_

        return self.labels[index].reshape(-1), distance.reshape(-1)

    def query_index(self, data, n):
        raise NotImplementedError

    def fit_index(self, data):
        raise NotImplementedError


class NMSLibIndex(ANNIndex):

    def __init__(self, method='hnsw', indexparams=None, sampling_threshold=20000, verbose=False):
        ANNIndex.__init__(self, sampling_threshold)
        if indexparams is None:
            # M = 5-100
            # efSearch 100-200
            # ['M=32', 'post=0', 'efConstruction=800']
            indexparams = ['M=24', 'post=0', 'efConstruction=90']
        self.method = method
        self.indexparams = indexparams
        self.verbose = verbose

    def fit_index(self, data):
        self.index = nmslib.init(method=self.method, space='l2')
        self.index.addDataPointBatch(data)
        self.index.createIndex(self.indexparams, print_progress=self.verbose)

    def query_index(self, data, n):
        index = self.index.knnQueryBatch(data, n)
        return np.array([i[0] for i in index]), np.array([i[1] for i in index])
