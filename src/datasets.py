import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Data_Formation(Dataset):
    """
    Класс, загружающий данные для обучения для последующей передачи в класс DataLoader. Стандартное требование для
    этого класса - реализация двух "магических" методов __getitem__ для возможности обращения к элементам данных как к
    элементам массива, и __len__ для возможности вычисления размера датасета. Данный класс реализует две возможности
    загрузки данных - загрузка в память только необходимого батча данных для экономии памяти и возможности работать с 
    большими объёмами данных, и загрузка всего набора данных сразу для более быстрого обучения сети, однако, применима
    только для небольших набороб данных.
    
    Параметры:
    ----------
    path_to_data: str
        Сторока, содержащая путь к данным.
    train: bool
        Флаг, указывающий какой надор данных загружать - тренировочный илил тестовый.
    """
    def __init__(self, path_to_data=None, train=True, one_file=False):
        self.data_list = []
        self.one_file = one_file
        if train == True:
            self.path = os.path.join(path_to_data, 'train')
        else:
            self.path = os.path.join(path_to_data, 'test')
        #вложений немного, поэтому использую os.listdir, а не os.walk
        classes_list = os.listdir(self.path)
        self.target_mat = torch.eye(len(classes_list), dtype=torch.float32)
        
        for cl in classes_list:
            temp_path = os.path.join(self.path, cl)
            for file in os.listdir(temp_path):
                self.data_list.append((os.path.join(temp_path, file), int(cl)))     
        
        self.data_len = len(self.data_list)
        
        if self.one_file == True:
            self.data = None
            for data_part in self.data_list:
                file_path, class_label = data_part
                if self.data is None:
                    with open(file_path, 'rb') as f:
                        self.data = np.load(f)
                    self.targets = np.full(shape=self.data.shape[0], fill_value=class_label)                      
                else:
                    with open(file_path, 'rb') as f:
                        self.data = np.concatenate((self.data, np.load(f)), axis=0)
                    self.targets = np.concatenate((self.targets, 
                                                   np.full(shape=(self.data.shape[0] - self.targets.shape[0]), 
                                                   fill_value=class_label)))
            self.data_len = self.data.shape[0]
                    
    def __getitem__(self, index):
        if self.one_file == False:
            file_path, target = self.data_list[index]
            with open(file_path, 'rb') as f:
                data = np.load(f)              
            return torch.from_numpy(data).to(dtype=torch.float32), \
                  self.target_mat[target]
        else:
            return torch.from_numpy(self.data[index]).to(dtype=torch.float32), \
                   self.target_mat[self.targets[index]]
    
    def __len__(self):
        return self.data_len

    

    
    
    