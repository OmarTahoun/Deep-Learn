from NN import *
import h5py

def load_images_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x = np.array(train_dataset["train_set_x"][:]) # your training set features
    train_set_y = np.array(train_dataset["train_set_y"][:]) # your training set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def main():
    train_x, train_y, test_x, test_y, classes = load_images_data()

    train_x = train_x.reshape(train_x.shape[0], -1).T
    test_x = test_x.reshape(test_x.shape[0], -1).T

    train_x = train_x / 255
    test_x = test_x / 255

    n_x = train_x.shape[0]
    n_y = test_y.shape[0]

    layers_size = [12288, 20, 7, 5, 1]
    parameters = model(train_x, train_y,  layers_size, iterations=2500, print_cost=True)

    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)


if __name__ == '__main__':
    main()
