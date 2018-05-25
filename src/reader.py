"""
    该文件是bookshelf的数据读取辅助文件
"""
from scipy import io
import numpy as np
import os
import re
import pandas as pd
import gc

#所有数据文件名的list
total_file_list = []

#损伤位置list
damage_location_list = ["L00","L1C","L3A","L13"]
#损伤程度list
damage_level_list = ["D00","DB0","DBB","DHT","D05","D10"]
#震动程度list
shake_level_list = ["V02","V05","V08"]
#属性名
columns = ["3BP","3BC","3AP","3AC","3CP","3CC","3DP","3DC","Destroy Data","2BC","2AP","2AC","2CP"
           ,"2CC","2DP","2DC","1BP","1BC","1AP","1AC","1CP","1CC","1DP","1DC","Time history","2BP","Damage Location",
          "Damage Level","Shake Level"]


def read(file_path):
    """
        根据文件名读取文件
        Args:
            file_path: 文件路径
        Return:
            读取的数据文件
    """
    f = io.loadmat(file_path)
    dict_key = "Data_" + file_path.split("/")[-1]
    dict_key = dict_key.split(".")[0]
    return f[dict_key]
  
def file_generator(file_path_list):
    """
        返回文件数据的生成器
        Args:
            file_path_list:   文件路径的list
        Return:
            数据的生成器
    """
    for file_path in file_path_list:
        data = read(file_path)
        data = pd.DataFrame(data)
        damage_location,damage_level,shake_level = file_path.split("/")[-1].\
                        split("_")[:3]
            
        addition = [[damage_location,damage_level,shake_level]] * data.shape[0]
        addition = pd.DataFrame(addition)
        data = pd.concat([data,addition],axis = 1)
        data.columns = columns
        
        yield data    
    
def read_data(root_path = "./",shake_level = None,damage_location = None,damage_level = None
              ,start_row = 0,row_count = None,one_hot = False):
    """
        根据条件获取相应数据集，None表示所有的数据，比如shake_level = None表示
        所有震动程度的数据，如果上述三个都是None则表示获取所有的数据
        注:如果取太多的文件则肯能内存溢出，报Memory Error,SPARK下不会溢出
        Args:
            start_row:  开始的行数
            row_count:  行数的长度
            root_path:  数据文件的总目录，默认为当前目录，
                            只要数据文件在该目录下就能扫描到，但是该目录下文件越多扫描所花时间越长
            shake_level:   震动程度
            damage_location:   损坏位置
            damage_level:   损伤程度
            one_hot：   是否进行one_hot编码
        Return：
            对应参数的数据
    """
    #搜索文件夹下的文件
    global total_file_list
    if len(total_file_list) == 0:
        filepath_list(root_path)
    #参数的字典
    param_dict = {"shake_level": shake_level,"damage_location": damage_location,"damage_level":damage_level}
    #文件路径list
    file_path_list = list(filter(lambda x: filter_file(x,**param_dict),total_file_list))
    #文件名排序
    file_path_list = sort_file_list(file_path_list = file_path_list)
    
    #取数据
    total_data = None
    data_generator = file_generator(file_path_list)
    for data in data_generator:
        total_data = data if total_data is None else pd.concat([total_data,data],axis = 0)
        
        if start_row < total_data.shape[0] and row_count is not None:
            end_row = start_row + row_count
            if end_row < total_data.shape[0] : 
                break;
            continue;
    try:
        total_data.reset_index(inplace = True,drop = True)
        if one_hot : 
            total_data = one_hot_encoding(total_data,["Shake Level","Damage Location","Damage Level"])
        return total_data.loc[start_row:end_row]
    except Exception as e:
        return None
    
def one_hot_encoding(data,col_name):
    """
        One-Hot编码
        Args:
            data:  要编码的数据
            col_name :   要编码的列
    """
    col_dummies = pd.get_dummies(data[col_name],prefix = col_name)
    data.drop(col_name,axis = 1,inplace = True)
    data = pd.concat([data,col_dummies],axis = 1)
    return data

def sort_file_list(file_path_list):
    """
        文件路径list排序
        Args:
            file_path_list:   文件路径的list
    """
    r_list = []
    #获取参数的list
    f = lambda x:"_".join(x.split("/")[-1].split("_")[:3])
    temp_param_list = list(map(f,file_path_list))  
    temp_param_list = list(set(temp_param_list))

    #根据不同参数分别排序
    for param in temp_param_list:
        sub_file_list = list(filter(lambda x : param in x,file_path_list))
        sub_file_list.sort(key = lambda x:  x.split("/")[-1].split("_")[-1].split(".")[0])
        r_list.extend(sub_file_list)
        
    return r_list
    

def filter_file(file_path,**kargs):
    """
         根据文件名判断是否符合参数条件
         Args:
             file_path:   文件路径
             **kargs:
                shake_level:   震动程度
                damage_location:   损坏位置
                damage_level:   损伤程度
        Return:
            符合返回True，不符合返回False
    """
    file_name_list = file_path.split("/")[-1].split("_")
    for args in kargs.values():
        if args is None: continue;
        if args not in file_name_list:
            return False
    return True
         
    
def filepath_list(root_path = "./"):
    """
        返回目录下的子数据文件列表
        Args:
            root_path:  扫描的父目录
        Return：
            文件名列表
    """
    global total_file_list
    for root,dirs,files in os.walk(root_path):
        #筛选以mat结尾的文件
        files = list(filter(lambda x : x.endswith(".mat"),files))
        #如果文件列表不为空
        if len(files) != 0:
            map_func = map(lambda x: root + "/" + x ,files)
            total_file_list += list(map_func)
    return total_file_list