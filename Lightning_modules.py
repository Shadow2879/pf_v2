import lightning as L
from torchmetrics.regression import MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError
from torch import nn
import torch
from collections import OrderedDict
import math
from fastai.torch_core import to_np
import gc
import warnings
import numpy as np
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_transformation import get_data_loaders

class ShapeCheck(nn.Module):
    '''Class for troubleshooting
        gives you input shape of data (singular).
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x):
        print(x.shape)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self,d_model: int,dropout: float=0.1,max_len: int=5000):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self,x: torch.Tensor):
        
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class ConvLogEmb(nn.Module):

    def __init__(self,resolution: int,min: int=-1,max: int=1):

        super().__init__()
        self.resolution=resolution
        self.min=min
        self.max=max

    def forward(self,x: torch.Tensor):

        x=torch.clamp(x,min=self.min,max=self.max)
        x=(x+1)*self.resolution+1
        return x

class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float=0.1,
        max_len: int=50000,
        pad_token: float=0.0,
        resolution: int=1000,
        emb_min: int=-1,
        emb_max: int=1,
        ):

        super(PositionalEmbedding,self).__init__()
        self.position=PositionalEncoding(d_model,dropout,max_len)
        self.embedding=nn.Embedding(num_embeddings=resolution*math.ceil(emb_max-emb_min)+1,embedding_dim=d_model)
        self.pad_token,self.resolution=pad_token,resolution
        self.converter=ConvLogEmb(resolution=resolution)

    def forward(self,x: torch.Tensor):

        emb=self.embedding(self.converter(x.int()))
        pos=self.position(emb)
        mask=(x!=0).float()
        logits=pos*mask[:,:,None]
        return logits
    
class PadInputs(nn.Module):

    def __init__(self,target_length: int,dim_to_pad: int):

        super(PadInputs, self).__init__()
        self.target_length = target_length
        self.dim_to_pad = dim_to_pad

    def forward(self,x: torch.Tensor):

        padding_size =self.target_length-x.size(self.dim_to_pad)
        # Compute how much padding is needed
        # Create a tuple of slice objects to slice along all dimensions
        slices = [slice(None)] * len(x.shape)
        # Set the slice for the dimension to pad
        slices[self.dim_to_pad] = slice(0,1,1)
        pad_tensor=torch.zeros_like(x[slices]).repeat_interleave(padding_size,dim=self.dim_to_pad)
        # Pad along the specified dimension
        padded_tensor = torch.cat([x, pad_tensor], dim=self.dim_to_pad)
        return padded_tensor

# class TFmodel(nn.Module):
class TFModel(nn.Module):
    def __init__(
            self,
            d_model: int=64,
            dropout: float=0.1,
            max_len: int=50000,
            pad_token: int=0,
            n_head: int=4,
            num_layers: int=2,
            resolution: int=1000,
            input_duration: int=1,
            pred_duration: int=1,
            files: list=torch.arange(100,requires_grad=False),
            y_inp: bool=True,
            reduce_dims: bool=True,
            emb_max: int=1,
            emb_min: int=-1,
            n_features: int=1,
        ):
        '''
        Initalizes a transformer model for autoregressive time series prediction.
        '''
        super().__init__()

        self.pred_duration=pred_duration
        self.files=files
        self.input_duration=input_duration
        self.y_inp=y_inp
        self.dim=(lambda x:pred_duration if x is True else input_duration)(y_inp)
        self.n_head=n_head
        self.num_layers=num_layers
        self.dropout=dropout
        self.pad_token=pad_token
        self.d_model=d_model
        self.max_len=max_len
        self.resolution=resolution
        self.emb_max=emb_max
        self.emb_min=emb_min
        self.n_features=n_features
        self.reduce_dims=reduce_dims
        
        if self.reduce_dims:
            self.reshape_shape=(self.input_duration*len(self.files),self.n_features*self.d_model)
        else:
            self.reshape_shape=(self.input_duration,len(self.files)*self.n_features*self.d_model)

        self.gen_enc()
        self.gen_dec()
        self.gen_ff()

    def gen_ff(self):
        
        layers=[(f'gelu_dec',nn.GELU()),
                (f'compressor_0',nn.Linear(in_features=self.reshape_shape[-1],out_features=self.reshape_shape[-1])),
                (f'gelu_0',nn.GELU()),
                (f'compressor_1',nn.Linear(in_features=self.reshape_shape[-1],out_features=1)),
                (f'gelu_1',nn.GELU()),
                (f'flatten_layer',nn.Flatten())]
        for i in range(len(self.reshape_shape)-2,0,-1):
            layers.append((f'compressor_{2+(2-i)}',nn.Linear(in_features=self.reshape_shape[i],out_features=1)))
            layers.append((f'flatten_layer',nn.Flatten()))
        layers.append(('bias_elim',nn.Linear(in_features=self.reshape_shape[0],out_features=len(self.files))))
        layers.append(('unflatten_layer',nn.Unflatten(dim=-1,unflattened_size=(len(self.files),1))))
        self.ff=nn.Sequential(OrderedDict(layers))

    def gen_enc(self):

        self.enc=nn.Sequential(OrderedDict([
            ('input flatten',nn.Flatten()),
            ('embedding layer',PositionalEmbedding(
                d_model=self.d_model,dropout=self.dropout,max_len=self.max_len,
                pad_token=self.pad_token,resolution=self.resolution,
                emb_max=self.emb_max,emb_min=self.emb_min
            )),
            ('tf flatten',nn.Flatten()),
            ('reshape layer',nn.Unflatten(1,self.reshape_shape)),
            ('encoder layers',nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=self.reshape_shape[-1],nhead=self.n_head,batch_first=True),self.num_layers
            )),
            ('enc_gelu',nn.GELU()),
        ]))      

    def gen_dec(self):
        emb_dims=self.d_model*self.n_features if self.y_inp else self.d_model
        self.dec_input=nn.Sequential(OrderedDict([
            ('input flatten',nn.Flatten()),
            ('embedding layer',PositionalEmbedding(
                d_model=emb_dims,dropout=self.dropout,max_len=self.max_len,
                pad_token=self.pad_token,resolution=self.resolution,
                emb_max=self.emb_max,emb_min=self.emb_min
            )),
            ('tf flatten',nn.Flatten()),
            ('reshape_layer',nn.Unflatten(1,self.reshape_shape)),
        ]))

        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=self.reshape_shape[-1],nhead=self.n_head,batch_first=True
        ),self.num_layers)

    def forward(self,x):

        if self.y_inp:
            self.y_shape=torch.tensor(x.shape[1:]).prod()//self.n_features
            outputs=torch.zeros((x.shape[0],self.y_shape,1),device=x.device)# for y inp
        else:
            outputs=x
        x=self.enc(x)

        for i in range(self.pred_duration):
            yin=self.dec_input(torch.clamp(outputs,self.emb_min,self.emb_max))
            y=self.decoder(yin,x)
            y=self.ff(y)

            if i==0:
                outputs=y
            else:
                outputs=torch.concat((outputs,y),2)

        return outputs


class LitTFmodel(L.LightningModule):
    def __init__(self,
                 model:nn.Module,
                 stocks:list,
                 loss=nn.MSELoss,
                 metrics=[MeanAbsoluteError(),MeanAbsolutePercentageError(),MeanSquaredError()],
                 reduction='none',
                 opt=torch.optim.Adam):
        super().__init__()
        self.model=model
        self.loss=loss(reduction=reduction)
        self.reduction=reduction
        self.metrics=metrics
        self.opt=opt
        self.stocks=stocks

    def forward(self,batch,batch_idx):
        x,_=batch
        return self.model(x)
    
    def predict_step(self,batch,batch_idx):
        x,_=batch
        return self.model(x)
    
    def configure_optimizers(self):
        return self.opt(self.model.parameters(),lr=1e-5)
        

    def log_loss_metrics(self,outputs,y):
        loss=self.loss(outputs,y)
        dev_metrics=[]
        for i in self.metrics:
            dev_metrics.append(i.to(y))

        self.log(f'avg_{self.loss._get_name()}',loss.mean(),prog_bar=True,logger=True)
        for i in dev_metrics:
            self.log(f"avg_{i._get_name()}",(i(outputs,y)).mean(),logger=True,prog_bar=True)

        if self.reduction=='none':
            for stock in enumerate(self.stocks):
                self.log(f'{stock[1]}_{self.loss._get_name()}',loss[stock[0]],logger=True)
                for metric in dev_metrics:
                    self.log(f'{stock[1]}_{metric._get_name()}',metric(outputs,y),logger=True)
        return loss

    def training_step(self,batch,batch_idx):
        x,y=batch
        outputs=self.model(x)
        return self.log_loss_metrics(outputs,y)
    
    def validation_step(self,batch,batch_idx):
        x,y=batch
        outputs=self.model(x)
        return self.log_loss_metrics(outputs,y)    

    def test_step(self,batch,batch_idx):
        x,y=batch
        outputs=self.model(x)
        return self.log_loss_metrics(outputs,y)
    
class LitDataModule(L.LightningDataModule):

    def __init__(self,x,y,input_duration,pred_duration,data_block_size,batch_size:int=32,shuffle=True,val_data_split=0.1):
        super().__init__()
        self.x=x
        self.y=y
        self.input_duration=input_duration
        self.data_block_size=data_block_size
        self.pred_duration=pred_duration
        self.batch_size = batch_size
        self.val_data_split=val_data_split
        self.shuffle=shuffle

    def setup(self, stage: str="fit"):
        self.stockvalloader,self.stocktrainloader,self.stocktestloader,self.data_splits=get_data_loaders(
            self.x,self.y,
            input_duration=self.input_duration,batch_size=self.batch_size,
            pred_duration=self.pred_duration,data_block_size=self.data_block_size,
            val_data_split=self.val_data_split,shuffle=self.shuffle,
        )
        print(self.data_splits)

    def train_dataloader(self):
        return self.stocktrainloader

    def val_dataloader(self):
        return self.stockvalloader
    
    def test_dataloader(self):
        return self.stocktestloader

    def predict_dataloader(self):
        return self.stocktestloader

