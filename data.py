import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset



class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = len(data_X)
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:
            self.addConflict(index, ratio_conflict)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass

def PIE():
    # dims of views: 484 256 279
    data_path = "data/PIE_face_10.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("PIE", data_X, data_Y)

def sources():
    data_path = "data/3sources.mat"
    data = sio.loadmat(data_path)
    data_X = {0: data['X1'], 1: data['X2'], 2: data['X3']}
    data_Y = data['truth']
    return MultiViewDataset("3Sources", data_X, data_Y)

def Scene():
    # dims of views: 20 59 40
    data_path = "data/scene15_mtv.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("Scene", data_X, data_Y)

def Webkb():
    data_path = "data/webkb.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    data_Y = data_Y.T
    return MultiViewDataset("Webkb",data_X,data_Y)

def MSRCV():
    data_path = "data/MSRCV1_6views.mat"
    data = sio.loadmat(data_path)
    data_X = [None, None, None, None, None, None]
    data_X[0] = data['x1']
    data_X[1] = data['x2']
    data_X[2] = data['x3']
    data_X[3] = data['x4']
    data_X[4] = data['x5']
    data_X[5] = data['x6']
    data_Y = data['gt']
    return MultiViewDataset("MSRCV", data_X, data_Y)

def COIL100():
    data_path = "data/COIL100.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']

    return MultiViewDataset("COIL100", data_X, data_Y)












