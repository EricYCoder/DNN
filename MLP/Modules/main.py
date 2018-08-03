import numpy as np

from new_dataio import prepare_data, Spectrum
from architecture import MLP
from trainer import Trainer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import settings

if __name__ == "__main__":
    for key in settings.file_pair.keys():
        test_path = settings.file_pair[key][0]
        train_path = settings.file_pair[key][1]

        # prepare data
        train_data = prepare_data(train_path)
        test_data = prepare_data(test_path)
        feature_len = len(train_data[0]) - 1

        print(len(train_data))
        print(len(test_data))

        train_data = train_data[:17174,]

        train_dataset = Spectrum(data=train_data)
        test_dataset = Spectrum(data=test_data)

        # Model parameters
        cuda = True
        in_channels = feature_len
        out_channels = 3
        hidden_num = 6
        epoch = 500

        print(len(train_data))
        for label_index in range(out_channels):
            print(np.sum(train_data[:, -1] == label_index))
        print(len(test_data))
        for label_index in range(out_channels):
            print(np.sum(test_data[:, -1] == label_index))

        batch_size = 256
        lr_rate = 1e-4
        optimizer_func = "Adam"
        model_name = key + ".pkl"

        model_info = (
            "test_path:"
            + test_path
            + " model_name:"
            + model_name
            + " batch_size:"
            + str(batch_size)
            + " optimizer_func:"
            + str(optimizer_func)
            + " learning_rate:"
            + str(lr_rate)
        )
        print(model_info)
        net = MLP(in_channels, out_channels, hidden_num)
        trainer = Trainer(net=net, cuda=cuda, model_name=model_name)
        print("Begining")
        trainer.train_model(
            train_dataset, test_dataset, batch_size, epoch, optimizer_func, lr_rate
        )

        shape = test_data.shape
        input_data = test_data[:, 0 : shape[1] - 1]
        pred = trainer.predict(input_data)
        truth = test_data[:, -1]
        pred = pred.astype(np.int)
        truth = truth.astype(np.int)

        f = open("res.txt", "a")
        class_report = classification_report(truth, pred)
        confu_matrix = confusion_matrix(truth, pred)
        f.write(model_info + "\n\n")
        f.write(class_report + "\n")
        f.write(str(confu_matrix) + "\n\n\n\n")
        f.close()
        print(class_report)
        print(confu_matrix)
