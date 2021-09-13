from dataset import AutoVpDataset
dataset_train = AutoVpDataset("/media/sunhuibo/data/Work/data/3rdparty/CULane/test_heatmap.txt")
dataset_train.__getitem__(100)
