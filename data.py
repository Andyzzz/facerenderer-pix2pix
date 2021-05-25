from os.path import join

from dataset import DatasetFromFolder


def get_training_set(root_dir, direction, csvfile):
    train_dir = root_dir  # join(root_dir, "train")

    return DatasetFromFolder(train_dir, direction, csvfile)


def get_test_set(root_dir, direction, csvfile):
    test_dir = root_dir  # join(root_dir, "test")

    return DatasetFromFolder(test_dir, direction, csvfile)
