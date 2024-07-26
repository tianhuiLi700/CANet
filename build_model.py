import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

def build_models(args, name):

    up = UpSample(args.num_para, args.hidden_dim, dropout=args.dropout)
    mlp = MLP(args.hidden_dim, args.hidden_dim, 4, 3)
    if name == 'lstm':
        model = build_lstm(args, mlp)
        return model
    elif name == 'gru':
        model = build_gru(args, mlp)
        return model
    elif name =="transformer":
        model = build_transformer(args, mlp, up)
        return model
    elif name =="conv":
        model = build_conv(args, mlp)
        return model
    elif name == "ours":
        model = build_ours(args, mlp)
        return model
    else:
        return None


def build_transformer(args, mlp, up):
    transformer = AttChange(d_model=args.hidden_dim,
                              nhead=8,
                              num_decoder_layers=1,
                              num_encoder_layers=1,
                              dim_feedforward=2048,
                              dropout=args.dropout)
    # transformer = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=1, num_decoder_layers=1, dropout=args.dropout)

    return AddTransformer(transformer, mlp, args.seq_len, up)

def build_gru(args, mlp):
    gru = torch.nn.GRU(input_size=args.num_para,
                       hidden_size=args.hidden_dim,
                       num_layers=args.lstm_layers,
                       batch_first=True,
                       dropout=args.dropout,)
    return AddModel(gru, mlp)

def build_lstm(args, mlp):
    lstm = torch.nn.LSTM(input_size=args.num_para,
                         hidden_size=args.hidden_dim,
                         num_layers=args.lstm_layers,
                         batch_first=True,
                         dropout=args.dropout,
                         )
    return AddModel(lstm, mlp)

def build_conv(args, mlp):
    return Conv()


def build_ours(args, mlp):
    return Ours()

class Ours(nn.Module):
    def __init__(self, d_model=128, nhead=8,
                 dim_feedforward=512, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(Ours, self).__init__()
        self.conv = nn.Conv1d(4, 128, 5, padding=2)
        num_m = 5
        self.convs = nn.ModuleList([nn.Conv1d(128, 128, 5, padding=2) for i in range(num_m)])

        self.atts = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout) for i in range(num_m)])
        self.mlps = nn.ModuleList([MLP(128, 246, 128, 3) for i in range(num_m)])
        self.ffns = nn.ModuleList([nn.Linear(128, 128) for i in range(num_m)])
        self.norms = nn.ModuleList([nn.LayerNorm(128) for i in range(num_m)])
        self.mlps_1 = nn.ModuleList([MLP(128, 246, 128, 3) for i in range(num_m)])
        self.last_conv = nn.Conv1d(128, 256, 50, stride=1)
        self.mlp = MLP(256, 512, 4, 3)

    def forward(self, x):

        for i in range(len(self.convs)):
            x = x.permute(0, 2, 1)
            if i == 0:
                x = self.conv(x)
            x = self.convs[i](x)
            x = x.permute(0, 2, 1)
            tgt = self.mlps[i](x)
            tgt1 = self.atts[i](tgt, tgt, tgt)[0]
            tgt1 = self.ffns[i](tgt1)
            x = tgt + tgt1
            x = self.norms[i](x)
            x = self.mlps_1[i](x)
        x = x.permute(0, 2, 1)
        x = self.last_conv(x).permute(0, 2, 1)
        x = self.mlp(x)
        return x[:, 0, :]



class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(nn.Conv1d(4, 16, 5, padding=2),
                                   nn.Conv1d(16, 64, 5, padding=2),
                                   nn.Conv1d(64, 256, 5, padding=2),
                                   nn.Conv1d(256, 1024, 50, stride=1),)
        self.mlp = MLP(1024, 1024, 4, 4)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.convs(x).transpose(1, 2)
        x = self.mlp(x)
        return x[:, 0, :]


class AddTransformer(nn.Module):
    def __init__(self, model, mlp, length, up):
        super(AddTransformer, self).__init__()
        self.transformer = model
        self.tgt = torch.nn.Embedding(1, 512)
        self.pos_emb = torch.nn.Embedding(length, 512)
        self.mlp = mlp
        self.up = up

    def pos2posemb(self, c, num_pos_feats=512, temperature=10000):
        scale = 2 * math.pi
        pos = c * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        posemb = pos[..., None] / dim_t
        posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
        return posemb

    def forward(self, x):
        bs, _, _ = x.shape
        # src = self.pos2posemb(x, 512//4)
        src = self.up(x)
        tgt = self.tgt.weight.data
        tgt = tgt.unsqueeze(0).repeat(bs, 1, 1)
        pos_emb = self.pos_emb.weight.data
        pos_emb = pos_emb.unsqueeze(0).repeat(bs, 1, 1)
        # batch first is False so we should change the channel
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        pos_emb = pos_emb.permute(1, 0, 2)
        pos_emb = None
        x = self.transformer(src, tgt, pos_emb)
        # x = self.mlp(x).permute(1, 0, 2)
        x = self.mlp(x)
        return x[:, 0, :]

class AddModel(nn.Module):
    def __init__(self, model, mlp):
        super(AddModel, self).__init__()
        self.model = model
        self.attn = nn.MultiheadAttention(512, 8, dropout=0.2, batch_first=True)
        self.mlp = mlp

    def forward(self, x):
        x = self.model(x)[0]
        x = self.attn(x, x, x)[0]
        x = self.mlp(x)

        return x[:, -1, :]



class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(UpSample, self).__init__()
        self.ffns = nn.ModuleList([nn.Linear(in_dim, out_dim//4),
                                   nn.Linear(out_dim//4, out_dim),
                                   nn.Linear(out_dim, out_dim),])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout),
                                       nn.Dropout(dropout),])

    def forward(self, x):
        for idx in range(len(self.ffns)):
            if idx == len(self.ffns) - 1:
                x = self.ffns[idx](x)
            else:
                x = F.relu(self.ffns[idx](x))
                x = self.dropouts[idx](x)
        return x




class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class AttChange(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super(AttChange, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, pos_embed):
        # flatten NxCxHxW to HWxNxC

        # memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, src)
        return hs[0].permute(1, 0, 2)







