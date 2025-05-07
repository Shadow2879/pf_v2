import os
import dotenv
from typing import Literal
def get_path(loc):
    return os.path.join(os.getcwd(),loc)

def ensure_dir_exists(loc):
    os.makedirs(loc,exist_ok=True)

def gen_var_path(var):
    path=get_path(var) 
    ensure_dir_exists(path)
    return path

def load_env_var(key:str,type:Literal['int','path','str','array'],arr_sep=',') ->int | str:
    dotenv.load_dotenv()
    var=os.environ.get(f'{key}')
    if var is None:
        match type:
            case "int":
                var=1
            case "path":
                var='/'
            case "str":
                var=''
            case default:
                raise NotImplementedError(f'default value for  type "{type}" env vars')
    match type:
        case "int":
            var=int(var)
        case 'path':
            var=gen_var_path(var)
        case 'array':
            var=var.split(arr_sep)
        case 'str':
            pass
        case default:
            raise NotImplementedError(f'loading env vars with type {type} is not yet implemented')
    return var