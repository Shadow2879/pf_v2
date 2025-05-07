'''
This file contains code in order to load and transform the stockdata into a format which can be used by a model for training.
TODO: 
Remove unused functions and features.
Add documentation for future ref.
'''

import os
import pandas as pd
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor
from stockstats import  wrap,unwrap
from itertools import repeat
import torch
from torch.utils.data import Dataset,DataLoader
from collections import namedtuple
import multiprocessing

class stockDataset(Dataset):
  '''
  Creates a torch dataset which can handle multiple tensors of the same length.

  '''
  def __init__(self,
               *args: np.ndarray | torch.Tensor | list,
               ) -> None:
    '''
    Initializes the torch dataset.

    Parameters:
      *args (list of arrays): different arrays containg model data.

    Example:
      >>> data=stockDataset(x,y,z))
      
    Note:
      The arrays need to be of the same size along dimension 0.
    '''
    self.vars=[torch.tensor(i) for i in args]

  def __len__(self) -> int:
    '''
    Returns:
      int: The length of the dataset.

    Example:
        >>> data=stockDataset(x,y,z)
        >>> len(data)
        50
    '''

    return [i.shape[0] for i in self.vars][0]
  
  def getshapes(self) -> list[torch.Size,torch.Size]:
    '''
    Gives the user the shapes of the tensors that are part of the dataset.

    Returns:
      list (torch.Size): The shape of each Tensor.  

    Examples:
        >>> data=stockDataset(x,y,z)
        >>> data.getshapes()
        [torch.Size(50,15,5),torch.Size(50,1),torch.Size(50,5,3)]
    '''

    return [i.shape for i in self.vars]
  
  def __getitem__(self,idx: int | range | list
                  ) -> list[torch.Tensor]:
    '''
    Retrives item at a certain index from the dataset.

    Parameters:
      idx (int|range|list): index for which data is to be retrieved.

    Returns:
      list(torch.Tensor): list of tensors at idx.

    Example:
        >>> data=stockDataset(np.arange(100).reshape(50,2),np.arange(100,200).reshape(50,2))
        >>> data[0]
        [[0,1],[100,101]]
    '''

    if torch.is_tensor(idx):
      idx=idx.tolist()

    return [i[idx] for i in self.vars]
    
class StoreVal():
  def __init__(self,val: int) -> None:

    self.value=val

  def add(self,_value: int) -> int:
    
    self.value+=_value
    return self.value

def get_indicators(df: pd.DataFrame,
                   indicators: list[str] | None =[],
                   drop_cols: list[str] =[]
                   ) -> pd.DataFrame:
  '''
  Calculate a set of indicators for a given dataframe.

    Parameters:
        df (dataframe): The input dataframe.
        indicators (list): Financial Indicators to calculate.
        drop_cols (list): Columns to be dropped after the calculation of indicators.

    Returns:
        dataframe: Contains respective indicators.

    Example:
        >>> get_indicators(df,['rsi'])
          log_ret_close rsi
        0 0.00000       0.0
        ...

    Notes:
        This function assumes that a date column is present which is used an index for data transformation and later dropped. 
        Log return of close price is always added to the list of indicators.
  '''
  df=wrap(df)
  if indicators is None:
    df.init_all()
  else:
    df['log-ret']
    indicators=list(set(indicators))
    if len(indicators):
      df[indicators]
  df=unwrap(df).reset_index(drop=True)
  if 'log-ret' in df.columns:
    df['log_ret_close']=df['log-ret']
    df.drop(columns=['log-ret'],axis=1,inplace=True)
  
  df.drop(columns=drop_cols,axis=1,inplace=True)
  return df

def drop_na_dfs(dfs: dict[pd.DataFrame],
                 stocknames: list[str]
                 ) -> list[pd.DataFrame]:
  '''
  Drop rows based on the number of NAs and sizes of the dataframes.
  
  Parameters:
    dfs (dict): dictionary of dataframes.
    stocknames (list): the names of relevant stocks present in the dictionary.

  Returns:
    list(dataframe): list of dataframes with a certain number of rows dropped.
    
  Note:
    This function drops rows from the start based on the number of NA counts as well as from the end based on the size of the smallest dataframe.
  '''

  na_counts=[i.isna().sum() for i in dfs]
  df_len=np.min([i.shape[0] for i in dfs])

  for i,(k,j) in zip(dfs,enumerate(stocknames)):
    print(f'{i.shape}',f'{j}\n rows dropped due to extra length:{i.shape[0]-df_len}',
          f"\n rows dropped due to NA counts: {na_counts[k]['log_ret_close']}\n")
    
  dfs=[i[np.max(na_counts):df_len] for i in dfs]
  return dfs

def load_data_indicators(df: pd.DataFrame,
                   indicators: list[str],
                   drop_cols: list[str]
                   ) -> tuple[pd.DataFrame,list[int]]:
  '''
  loads a csv file at path, foward fills NA values and caluculates indicators.

  Parameters:
    path (str): The location of the file.
    indicators (list): Financial indicators to compute.
    drop_cols (list): columns to be dropped after the calculation of inidcators.
  
  Returns:
    a tuple containing:
      dataframe: loaded from the file and with respective indicators.
      nacounts: total NA counts of each dataframe.
  '''
  nacounts=df.isna().sum().sum()
  
  if nacounts>0:
    df.ffill(inplace=True)
  return get_indicators(df,indicators,drop_cols),nacounts

def get_dataframes(
                   path:str,
                   indicators: list[str] =[],
                   drop_cols: list[str] =[],
                   circ_limit: tuple[int,int]=(0.2,-0.2)
                   ) -> tuple[dict[pd.DataFrame],list[str]]:
  '''
  loads all .csvs in a directory, calculates indicators and imputes data.

  Parameters:
    indicators (list): Financial indicators to compute.
    drop_cols (list): columns to be dropped after the calculation of inidcators.
    circ_limit (tuple(int,int)): Maximum and Minimum acceptable values for log return. 

  Returns:
    tuple(dict[dataframes],list):  a dictionary of all dataframes loaded and transformed from the directory along with a list of file names which were loaded.
  '''

  from data_process import load_df_dict_from_npz
  dfs=load_df_dict_from_npz(path)
  files=list(dfs.keys())
  with ProcessPoolExecutor() as executor:
    res=list(executor.map(load_data_indicators,list(dfs.values()),repeat(indicators),repeat(drop_cols)))
  res,nas=[i[0] for i in res],{j:i[1] for i,j in zip(res,files)}
  print(f'filled datapoints due to NAs:\n{nas}')
  
  res=drop_na_dfs(res,files)

  with ProcessPoolExecutor() as executor:
    res=list(executor.map(impute_data,res,repeat(circ_limit),files))
  
  res={j:i for i,j in zip(res,files)}

  print(f'NAcounts in the dict:{sum([res[i].isna().sum().sum() for i in res.keys()])}')
  return res,files

def impute_data(df: pd.DataFrame,
                circ_limit: tuple[int,int],
                name: str
                ) -> pd.DataFrame:
  '''
  Imputes data that is beyond a limit to its stockwise mean.

  Parameters:
    df (dataframe): Dataframe to impute.
    circ_limit (tuple(int,int)): Maximum and Minimum acceptable values for log return.
    name (str): Stockname.
  
  Returns:
    dataframe: which is imputed based on log-return values.  
  '''
  indices=df[(df['log_ret_close']>=circ_limit[0])  | (df['log_ret_close']<circ_limit[1])].index
  print(f'imputed indices:{list(indices)}',end=' ')
  print(f'imputed values:{list(df["log_ret_close"][indices])} for {name}\n',end='')
  df[(df['log_ret_close']>=circ_limit[0])  | (df['log_ret_close']<circ_limit[1])]=df.mean()
  return df

def conv_df(inputs: pd.DataFrame,
            x_col: list[str] | None,
            y_col: list[str] | None,
            input_duration: int,
            pred_duration: int,
            window_inc: int,
            # *args: list[str] | None
            ) -> tuple[np.ndarray,np.ndarray]:
    '''
    converts a dataframe into x and y data that can be used for model training.
  
    Parameters:
      inputs (dataframe): dataframe to convert into x,y
      x_col (list(str)): columns of the dataframe to be included in x (defaults to all columns in the dataframe).
      y_col (list(str)): columns of the dataframe to be included iin y (defaults to all columns in the dataframe).
      input_duration (int): Number of data points of the stock to be passed into the model.
      pred_duration (int): Number of data points of the stock which is to be predicted by the model.
      window_inc (int): Number by which to move the sliding window.
    
    Returns:
      tuple(array,array): tuple containing (x,y) data.

    '''

    if x_col is None:
        x_col=inputs.columns

    if y_col is None:
        y_col=inputs.columns
        
    res=[[] for _ in range(2)] # len(args)
    
    if input_duration:
        for i in range(input_duration):
            res[0].append(inputs[x_col].iloc[i:inputs.shape[0]-input_duration-pred_duration+i+1])

    if pred_duration:
        for j in range(pred_duration):
            res[1].append(inputs[y_col].iloc[j+input_duration:inputs.shape[0]-pred_duration+j+1])

    counts=np.repeat(
      np.max([res[i][-1].shape[0] 
              if len(res[i]) 
              else 0 
              for i in range(len(res))]),len(res)
              )
    res=[np.concatenate([
      np.expand_dims(i[:counts[j]],axis=1) 
      for i in res[j]],axis=1) 
      if len(res[j]) 
      else [] 
      for j in range(len(res))
      ]
    res=[i[1] if len(i[1]) 
         else np.empty((counts[i[0]],0,len([x_col,y_col][i[0]]))) 
         for i in enumerate(res)
         ]

    return (res[0],res[1])



def stack_stockwise_data(inputs: np.ndarray,dim: int) -> np.ndarray:
  '''
  stacks the data in the array along a certain dimension.

  Parameters:
    inputs (array): the input array.
    dim (int): dimension to concatenate along.

  Returns:
    array: The concatenated array.
  '''

  res=inputs[0]

  for i in range(1,len(inputs)):
    res=np.concatenate((res,inputs[i]),axis=dim)

  return res

def get_model_data(inputs: dict[pd.DataFrame],
                   x_col: list[str] | None,
                   y_col: list[str] | None,
                   input_duration: int,
                   pred_duration: int,
                   window_inc: int,
                   ) -> tuple[np.ndarray,np.ndarray]:
  '''
  converts the dataframe into (x,y) data which can be used for the model.

  Parameters:
    inputs (dict(dataframe)): dictionary of dataframes.
    x_col (list(str)): columns of the dataframes to be included in x.
    y_col (list(str)): columns of the dataframes to be included in y.
    input_duration (int): Number of data points of the stock to be passed into the model.
    pred_duration (int): Number of data points of the stock which is to be predicted by the model.
    window_inc (int): Number by which to move the sliding window.  

  Returns:
    tuple(array,array): tuple containing (x,y) data.
  '''

  with ProcessPoolExecutor() as executor:
    res=list(executor.map(conv_df,inputs.values(),
    repeat(x_col),repeat(y_col),repeat(input_duration),
    repeat(pred_duration),repeat(window_inc)))

  x,y=np.array([res[i][0] for i in range(len(res))]),np.array([res[i][1] for i in range(len(res))])
  x,y=stack_stockwise_data(x,2)[:,:,:],stack_stockwise_data(y[:,:,None,:,0],1)
  return np.array(x),np.array(y)

def conv_to_torch_ds(data: list[torch.Tensor],pred_duration: int =1) -> Dataset:

  vals=[]
  for i in range(len(data[0])):
      vals.append([data[j][i] for j in range(len(data))])
  return stockDataset(
    *[torch.cat(vals[i],dim=0) for i in range(len(vals))],
  )

def get_data_segments(ds: Dataset,
                      input_duration: int,
                      data_block_size:int =252,
                      val_data_split:float =0.1
                      ) -> tuple[Dataset,Dataset,Dataset,namedtuple]:
  '''
  segments the data in the torch dataset into validation, train and test.

  Parameters:
    ds (torch.dataset): the torch dataset.
    input_duration (int): number of datapoints to be included as inputs to the model.
    data_block_size (int): the size of the block which is used to segment data between the 3 datasets.
    val_data_split (float): the percentage of data which is to be assigned to validation and test datasets.

  Returns:
    tuple: which contains the three datasets and a namedtuple which records the distribution of data between the three datasets.
  '''

  val_len=int((data_block_size-3*input_duration)*val_data_split)
  n_blocks=int(len(ds)/data_block_size)
  c_index=StoreVal(len(ds)-n_blocks*data_block_size)
  test_data,val_data,train_data,ds_index=[],[],[],[]
  indices=namedtuple("dataIndices",
            'val_start, val_end, train_start, train_end, test_start, test_end')
  
  for _ in range(n_blocks):
    ds_index.append(
      indices(
        c_index.value,
        c_index.add(val_len),
        c_index.add(input_duration),
        c_index.add(data_block_size-3*input_duration-val_len*2),
        c_index.add(input_duration),
        c_index.add(val_len)
      )
    )
  c_index.add(input_duration)

  for i in range(n_blocks):
      val_data.append(ds[ds_index[i][0]:ds_index[i][1]])
      train_data.append(ds[ds_index[i][2]:ds_index[i][3]])
      test_data.append(ds[ds_index[i][4]:ds_index[i][5]])
  return val_data,train_data,test_data,ds_index

def get_data_loaders(
                    *args: torch.tensor,
                     input_duration: int,
                     batch_size: int,
                     pred_duration: int =1,
                     data_block_size: int =252,
                     val_data_split: float =0.1,
                     shuffle: bool =True,
                     num_workers:int=multiprocessing.cpu_count()-1,
                     ) -> tuple[DataLoader,DataLoader,DataLoader,namedtuple]:
  '''
  segments input data into val,train and test and returns the dataloaders.

  Parameters:
    args: the tensors for which torch dataloaders are to be generated.
    input_duration (int): number of datapoints to be included as inputs to the model.
    batch_size (int): size of the batch.
    pred_duration (int): prediction duration of the model.
    data_block_size (int): the size of the block which is used to segment data between the 3 datasets.
    val_data_split (float): the percentage of data which is to be assigned to validation and test datasets.
    shuffle (bool): flag to whether to shuffle data retrived from the dataloader.
  
  Returns:
    tuple: containing dataloaders for validation,train and test along with the distribution of data into said datasets.

  Note:
    this function supports multiple tensors.
  '''
  ds=stockDataset(*args)
  val,train,test,blocks=get_data_segments(ds,input_duration,data_block_size,val_data_split)
  print(blocks)

  stockvalloader=DataLoader(conv_to_torch_ds(val),batch_size=batch_size,shuffle=False,num_workers=num_workers)
  stocktrainloader=DataLoader(conv_to_torch_ds(train),batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
  stocktestloader=DataLoader(conv_to_torch_ds(test),batch_size=batch_size,shuffle=False,num_workers=num_workers)
  return stockvalloader,stocktrainloader,stocktestloader,blocks

