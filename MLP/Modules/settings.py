import os

hidden_layer_params = {
    3: [256, 128, 64],
    4: [256, 128, 128, 64],
    5: [256, 256, 128, 128, 64],
}

hidden_num_list = [3, 4, 5]
batch_size_list = [32, 64, 128, 256]
# init_func = ["random", "truncated", "gaussian"]

# acti_func = ["ReLU", "LeakyReLU", "tanh", "softsign"]
opti_func_list = ["SGD", "RMSprop", "AdaGrad", "Adam"]
learning_rate_list = [1e-1, 1e-3, 1e-6]

file_pair = {
    "1001-33-2-CoCoSoSpWiDuOt-L": [
        os.path.expanduser("~")
        + "/data_pool/waterfall_data/pretrain_result/0501_1001_33_2_CoCoSoSpWiDuOt_L_REG_TEST_17.npz",
        os.path.expanduser("~")
        + "/data_pool/waterfall_data/pretrain_result/0501_1001_33_2_CoCoSoSpWiDuOt_L_REG_TRAIN_141516.npz",
    ],
    "1001-17-1-CoCoSoSpWiDuOt-L": [
        os.path.expanduser("~")
        + "/data_pool/waterfall_data/pretrain_result/0501_1001_17_1_CoCoSoSpWiDuOt_L_REG_TEST_17.npz",
        os.path.expanduser("~")
        + "/data_pool/waterfall_data/pretrain_result/0501_1001_17_1_CoCoSoSpWiDuOt_L_REG_TRAIN_141516.npz",
    ],
}
