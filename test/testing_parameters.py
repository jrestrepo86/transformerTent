# model parameters
target_dim = 1
source_dim = 1
model_dim = 16
history_len = 5
prediction_len = 30
heads = 1
embed_max_len = 5000
dropout = 0.0
transf_activation = "elu"
transf_fordward_expansion = 2  # d_ff =  transf_fordward_expansion * model_dim

# training parameters
batch_size = 1024
max_epochs = 4
lr = 1e-4
treet_block_randsamp_draws = 15
source_history_len = 3

treetModel_parameters = {
    "model_params": {
        "target_dim": target_dim,
        "source_dim": source_dim,
        "model_dim": model_dim,
        "heads": heads,
        "history_len": history_len,
        "prediction_len": prediction_len,
        "attn_dropout": dropout,
        "embed_max_len": embed_max_len,
        "embed_dropout": dropout,
        "transf_activation": transf_activation,
        "transf_dropout": dropout,
        "transf_fordward_expansion": transf_fordward_expansion,
        "treet_block_randsamp_draws": treet_block_randsamp_draws,
    },
    "train_params": {
        "batch_size": batch_size,
        "max_epochs": 80,
        "lr": lr,
        "weight_decay": 5e-5,
        "train_size": 0.8,
        "val_size": 0.2,
        "normalize_dataset": None,
        "calc_tent": True,
        "source_history_len": source_history_len,
        "verbose": True,
    },
}


oriTREETargs = {
    # "training"
    # Training related parameters
    "is_training": 1,  # status
    "train_epochs": max_epochs,  # train epochs
    "batch_size": batch_size,  # batch size of train input data
    "patience": 20,  # early stopping patience
    "learning_rate": lr,  # optimizer learning rate
    "loss": "dv",  # loss function
    "lradj": "type1_0.95",  # adjust learning rate
    "use_amp": False,  # use automatic mixed precision training
    "optimizer": "adam",  # optimizer name, options: [adam, rmsprop]
    "n_draws": treet_block_randsamp_draws,  # number of draws for DV potential calculation
    "exp_clipping": "inf",  # exponential clipping for DV potential calculation
    "alpha_dv_reg": 0.0,  # alpha for DV regularization on C constant
    "num_workers": 0,  # data loader num workers
    "itr": 1,  # experiments times
    "log_interval": 5,  # training log print interval
    # "model"
    # /* Model related parameters */
    "model": "Decoder_Model",  # model name, options: [Transformer_Encoder, LSTM, Autoformer, Informer, Transformer]
    "seq_len": 0,  # input sequence length
    "label_len": history_len,  # start token length. pre-prediction sequence length for the encoder
    "pred_len": prediction_len,  # prediction sequence length
    "y_dim": target_dim,  # y input size - exogenous values
    "x_dim": source_dim,  # x input size - endogenous values
    "c_out": 1,  # output size
    "d_model": model_dim,  # dimension of model
    "n_heads": heads,  # num of heads
    "time_layers": 1,  # num of attention layers
    "ff_layers": 1,  # num of ff layers
    "d_ff": transf_fordward_expansion * model_dim,  # dimension of fcn
    "factor": 1,  # attn factor (c hyper-parameter)
    "distil": True,  # whether to use distilling in encoder, using this argument means not using distilling
    "dropout": dropout,  # dropout
    "embed": "fixed",  # time features encoding, options:[timeF, fixed, learned]
    "activation": transf_activation,  # activation - must be elu to work with NDG
    "output_attention": False,  # whether to output attention in encoder
    # "process_channel"
    # /* Process and Channel related parameters */
    "use_ndg": False,  # use NDG instead of previous created dataset.
    "process_info": {
        "type": "Apnea",
        "x": "breath",  # TE(heart->breath) < TE(breath->heart)
        "x_length": source_history_len,  # -1 means all = label length (the last x is trimmed due to synchronization of the processes)
        "memory_cut": True,  # reset states of the model, and taking data without stride
    },
}
