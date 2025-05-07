import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from env_var import load_env_var

PROCESSED_DATA_DIR=load_env_var('PROCESSED_DATA_DIR','path')
F_PATH=os.path.join(PROCESSED_DATA_DIR,datetime.now().strftime("%y%m%d"))
RAW_DATA_DIR=load_env_var('RAW_DATA_DIR','path')

def save_dict_df(dfs:dict[pd.DataFrame]):
    f_loc=os.path.join(F_PATH+f'_{len(list(dfs.keys()))}')
    for i in dfs.keys():
        dfs[i]=dfs[i].rename({'Index':'date','Date':'date'},axis=1)
    dfs['columns']=dfs[next(iter(dfs.keys()))].columns
    np.savez(f_loc,**dfs)
    print(f'data saved to {f_loc}')
    return f_loc

def list_csvs_dir(dir):
    '''
    absolute path of dir from which files are to be listed.
    returns an array of shape (n_files,2) where each element is of the form (path,name).
    '''
    print('reading csvs in dir:',dir)
    paths=np.array([[os.path.join(dir,i),i] for i in os.listdir(dir) if os.path.isfile(os.path.join(dir,i)) and '.csv' in i])
    return paths

def list_raw_csvs_from_dir(subdir):
    '''
    only subdirectory in raw to be specified.
    '''
    return list_csvs_dir(os.path.join(RAW_DATA_DIR,subdir))

def load_from_gix(gix_idx:int=None,prev_pf_idx:int=None,curr_pf_idx:int=None,req_user_inp:bool=True,use_pfs:bool=True):
    '''
    loads data from a gix file, optionally selects required stocks relevant for rebalancing and collates data into a .npz file.
    '''
    gixs=list_raw_csvs_from_dir('gix')
    if gix_idx is None and req_user_inp:
        print(list(enumerate(gixs[:,1])))
        gix_idx=np.array(input('enter gix_idx:')).astype(np.int16)
    print(f'User chose: \n gix:{gixs[gix_idx,1]}')
    df=pd.read_csv(gixs[gix_idx,0],sep=' ',)
    print(df.shape,df.columns)
    print(f'gix NA counts: {df.isna().sum().sum()}/{df.shape[0]*df.shape[1]} = {(df.isna().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%')

    if use_pfs:
        pfs=list_raw_csvs_from_dir('pf')
        if prev_pf_idx is None or curr_pf_idx is None and req_user_inp:
            print(list(enumerate(pfs[:,1])))
            curr_pf_idx,prev_pf_idx=np.array(input('enter current_pf_idx and prev_pf_idx:').split(',')).astype(np.int16)
        print(f'User chose:\n current_pf:{pfs[curr_pf_idx,1]}, previous_pf: {pfs[prev_pf_idx,1]}')
        stocklist=np.concatenate((
            np.unique(
                np.concatenate((
                    pd.read_csv(pfs[curr_pf_idx,0])['Unnamed: 0'],
                    pd.read_csv(pfs[prev_pf_idx,0])['Unnamed: 0']
                    ))
                ),
            ['Index']
            ))
        print('unique columns:',stocklist)
        df=df[stocklist]
        print(f'relevant stock NA counts: {df.isna().sum().sum()}/{df.shape[0]*df.shape[1]} = {(df.isna().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%')
    
    dfs={}
    for i in df.drop(['Index'],axis=1).columns:
        dfs[i]=pd.DataFrame(df[['Index',i]])
    return save_dict_df(dfs)

def load_from_dir(dir,abspath=False,name_extractor=lambda f_name:f_name.split('.')[0]):
    '''
    loads all csvs in a dir,collates data and saves it into a single .npz file.
    '''
    if abspath:
        csvs=list_csvs_dir(dir)
    else:
        csvs=list_raw_csvs_from_dir(os.path.join('csvs',dir))
    df_dict={}
    for i in csvs:
        df_dict[name_extractor(i[1])]=pd.read_csv(i[0]).drop('Unnamed: 0',axis=1)
    print('data for ',len(list(df_dict.keys())),'stocks read')
    print("stocks:\n",df_dict.keys())
    print(df_dict[list(df_dict.keys())[0]])
    return save_dict_df(df_dict)
def load_df_dict_from_npz(f_loc):
    dfs={}
    data=np.load(f_loc,allow_pickle=True)
    cols=data[list(data.keys())[-1]]
    for i in list(data.keys())[:-1]:
        dfs[i]=pd.DataFrame(data[i],columns=cols)
        for j in dfs[i].columns[1:]:
            dfs[i][j]=dfs[i][j].astype(float)
    return dfs