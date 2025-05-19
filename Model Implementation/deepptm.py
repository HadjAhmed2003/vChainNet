import torch
from torch import nn
import math
from torch.autograd import Variable
from typing import Tuple


class deepPTM(nn.Module):
    def __init__(self, in_feat=12,
                 lstm_config= {"width":[200, 100], "keep_prob": 1, 'bidirectional': True, 'dropout': 0.0},
                 attn_config={"dim": 64, "num_heads": 3, "out_dim": 32, 'dropout': 0.0},
                 time_steps=42, device = "cpu", *args, **kwargs) -> None:
        """
        """
        super().__init__(*args, **kwargs)
        # Save configs
        self.lstm_bidirectional = lstm_config["bidirectional"]
        self.attn_embbed_dim = attn_config["dim"]
        self.num_attn_heads = attn_config["num_heads"]
        self.time_steps = time_steps
        self.device = device
        # TODO: Add initialization
        # Input LSTM
        self.in_lstm_fw = self._get_multi_layer_lstm(
                in_feat=in_feat, hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
            
        # Use type 1 biderictonal lstm where both paths are independent across all alyers
        if lstm_config["bidirectional"]:
            self.in_lstm_bw = self._get_multi_layer_lstm(
                in_feat=in_feat, hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
        
        # KQV encoder
        self.val_enc = self._get_encoder_layers(lstm_config["width"][-1], attn_config["dim"], attn_config["num_heads"])
        self.key_enc = self._get_encoder_layers(lstm_config["width"][-1], attn_config["dim"], attn_config["num_heads"])
        self.query_enc = self._get_encoder_layers(lstm_config["width"][-1], attn_config["dim"], attn_config["num_heads"])

        # Multihead attention
        self.attn = nn.MultiheadAttention(embed_dim=attn_config["dim"] * attn_config["num_heads"], num_heads=attn_config["num_heads"],
                                          dropout=attn_config["dropout"], batch_first=True)
        self.attn_out = torch.nn.Linear(attn_config["dim"] * attn_config["num_heads"], attn_config["out_dim"])
        
        # LSTM out
        self.out_lstm_fw = self._get_multi_layer_lstm(
                in_feat=attn_config["out_dim"], hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
            
        # Use type 1 biderictonal lstm where both paths are independent across all alyers
        if lstm_config["bidirectional"]: 
            self.out_lstm_bw = self._get_multi_layer_lstm(
                in_feat=attn_config["out_dim"], hidden_size=lstm_config["width"], dropout=lstm_config["dropout"])
        
        self.dec_layer = nn.Linear(lstm_config["width"][-1], 1)

    def _get_multi_layer_lstm(self, in_feat, hidden_size=[200, 100], dropout=0.):
        lstm = nn.ModuleList(
            [nn.LSTM(input_size=in_feat, hidden_size=hidden_size[0],
                     num_layers=1, batch_first=True, dropout=dropout)]+
                     [nn.LSTM(input_size=hidden_size[i], hidden_size=width, 
                        num_layers=1, batch_first=True, dropout=dropout) 
                        for i, width in enumerate(hidden_size[1:])]
                        )
        return lstm


    def _get_encoder_layers(self, in_dim, out_dim, num_heads):
        return nn.ModuleList([
            torch.nn.Linear(in_dim, out_dim).to(self.device) for i in range(num_heads)])

    # run type 1 biderictonal LSTM
    def _run_multi_layer_bi_directional(self, x, lstm_fw, lstm_bw):
        
        x_f = lstm_fw[0](x)[0]
        x_b = lstm_bw[0](x.flip(-1))[0]
        for l_f, l_b in zip(lstm_fw[1:], lstm_bw[1:]):
            x_f = l_f(x_f)[0]
            x_b = l_b(x_b)[0]

        x = x_f + x_b
        return x

    # run type 2 biderictonal LSTM
    def _run_type_2_multi_layer_bi_directional(self, x, lstm_fw, lstm_bw):
        x_f = lstm_fw[0](x)[0]
        x_b = lstm_bw[0](x.flip(-1))[0]
        for l_f, l_b in zip(lstm_fw[1:], lstm_bw[1:]):
            x = x_f + x_b
            x_f = l_f(x)[0]
            x_b = l_b(x.flip())[0]

        x = x_f + x_b
        return x


    def forward(self, packet_batch):
        b = len(packet_batch)
        # TODO: Check dimensionalit and in and output codes
        x = packet_batch

        if self.lstm_bidirectional:
            x = self._run_multi_layer_bi_directional(x, self.in_lstm_fw, self.in_lstm_bw)
        else:
            x = self.in_lstm_fw(x)
        # x = self.positional_encoder(x)
        v = torch.zeros([b, self.time_steps, self.num_attn_heads, self.attn_embbed_dim],device = self.device)
        k = torch.zeros([b, self.time_steps,  self.num_attn_heads, self.attn_embbed_dim], device = self.device)
        q = torch.zeros([b, self.time_steps, self.num_attn_heads, self.attn_embbed_dim],device = self.device)

        for i, (val_enc, key_enc, query_enc) in enumerate(zip(self.val_enc, self.key_enc, self.query_enc)):
            v[:,:, i] = val_enc(x)
            k[:,:, i] = key_enc(x)
            q[:,:, i] = query_enc(x)

        v = v.reshape(b, self.time_steps, -1)
        k =k.reshape(b, self.time_steps, -1)
        q = q.reshape(b, self.time_steps, -1)

        att = self.attn(q, k, v)[0]
        x = self.attn_out(att)

        if self.lstm_bidirectional:
            x = self._run_multi_layer_bi_directional(x, self.out_lstm_fw, self.out_lstm_bw)
        else:
            for l in self.out_lstm_fw:
                x = l(x)[0]

        lstm_out = x
        
        t_pred = self.dec_layer(lstm_out)

        return t_pred
