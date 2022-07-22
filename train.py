import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import numpy as np
import pickle as pkl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from argparse import ArgumentParser
from model.MyDataset import get_batch_data
from model.GTAE import GTAEModel

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

seed_everything(27, workers=True)

def train():
    # parser
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GTAEModel.setting_model_args(parser)
    args = parser.parse_args()
    print(args)

    train_data = get_batch_data('./dataset/{}/train.pkl'.format(args.data_name), batch_size=args.batch_size, type='train')
    valid_data = get_batch_data('./dataset/{}/val.pkl'.format(args.data_name), batch_size=args.batch_size, type='valid')
    test_data = get_batch_data('./dataset/{}/test.pkl'.format(args.data_name), batch_size=args.batch_size, type='test')

    with open('./dataset/{}/nodes.pkl'.format(args.data_name), 'rb') as f:
        nodes = pkl.load(f)
        N = len(nodes)
        args.N = N

    GtaeModel = GTAEModel(args.d_model, args.hidden_dim, args.aggregators.split(','), args.scalers.split(','),
                          args.dgn_layers, args.avg_d, args.N, args.dropout_rate, args.k, args.seq_len, args.batch_size,
                          # args.nhead, args.ffn_dim, args.act, args.num_layers, args.is_ge, False, False,
                          args.nhead, args.ffn_dim, args.act, args.num_layers, args.is_ge, args.is_vae, args.is_reg,
                          args.classes, args.lam)
    print('total params:', sum(p.numel() for p in GtaeModel.parameters()))
    print(GtaeModel)
    checkpoints_callback = ModelCheckpoint(monitor='val_acc', filename=args.data_name + '-{epoch:03d}-{val_acc:.5f}',
                                           save_top_k=1, mode='min', save_last=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=7, mode='max')

    trainer = pl.Trainer(callbacks=[checkpoints_callback, LearningRateMonitor(logging_interval='epoch'), early_stopping],
                         max_epochs=args.total_epochs, gpus=args.gpu_lst,
                         deterministic=True)
    trainer.from_argparse_args(args)
    trainer.fit(model=GtaeModel, train_dataloaders=train_data, val_dataloaders=valid_data)
    res = trainer.test(model=GtaeModel, dataloaders=test_data, ckpt_path='best')
    print(res)
train()