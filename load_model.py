# -*- coding: utf-8 -*-
# time: 2024/5/31 16:23
# file: load_model.py
# author: Shuai


from model.LSTM import LSTM
from model.unet import U_Net
# from model.mamba_reshape import mamba_reshape
from model.tst.transformer import Transformer
from model.TTTmodel import TTT


def load_model(args=None):
    if args.model_name == 'LSTM':
        model = LSTM(args.inputs, args.outputs)

    elif args.model_name == 'unet':
        model = U_Net(in_ch=args.inputs, out_ch=args.outputs)

    # elif args.model_name == 'mamba2' or args.model_name == 'mamba':
    #     model = mamba_reshape(inputs=args.inputs, model_name=args.model_name, outputs=args.outputs)

    elif args.model_name == 'transformer':
        model = Transformer(d_input=args.inputs, d_model=args.d_model, d_output=args.outputs, q=8, v=8, h=8, N=4,
                            attention_size=12, dropout=0.2,
                            chunk_mode=None, pe='regular')

    elif args.model_name == 'TTT':
        model = TTT(inputs=args.inputs, outputs=args.outputs)

    return model
