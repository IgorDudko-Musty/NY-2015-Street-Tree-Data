import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datasets import Data_Formation
import matplotlib.pyplot as plt
import os


class Seq_Model(nn.Module):
    """
    Класс создающий архитектуру нейронной сети. Для обучения была выбрана
    простоя полносвязная сеть исходя из структуры имеющихся данных
    
    Параметры:
    ----------
    input_size: int
        Количество нейронов во входном слое
    output_size: int
        Количество нейронов во выходном слое
    hidden_size: int, optional
        Количество нейронов во первом скрытом слое, остальные скрытые слои
        добавляются автоматически с уменьшением количества нойронов вдвое.
    act: str, optional
        Параметр, указывающий какую функцию активации использовать.
    """
    def __init__(self, input_size, output_size, hidden_size=128, act='relu'):
        super().__init__()
        self.activations = nn.ModuleDict({
                                        'relu': nn.ReLU(),
                                        'lrelu': nn.LeakyReLU()
                                        })
        
        self.layers = nn.ModuleList()
        for i in range(5):
            self.layers.add_module("layer_{}".format(i),
                                   nn.Linear(input_size,
                                             hidden_size,
                                             bias=False))
            self.layers.add_module("act_{}".format(i),
                                   self.activations[act])
            self.layers.add_module("BatchNorm_{}".format(i),
                                   nn.BatchNorm1d(hidden_size))
            self.layers.add_module("dropout_{}".format(i),
                                   nn.Dropout(0.25))
            input_size = hidden_size
            hidden_size = int(hidden_size / 2)
            
        self.layers.add_module("output_layer",
                               nn.Linear(input_size, output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

            
class Model_Implementation():
    """
    Класс, отвечающий за непосредственное создание модели, её обучение,
    валидацию, тестирование и прогнозирование.
    
    Параметры:
    ----------
    path_data_model: list
        Список, содержащий пути к данным и к директории, куда необходимо
        сохранить модель
    mode: str
        Флаг, указывающий в какой режиме работает модель
    batch_size: int, optional
        Размер батча.
    output_size: int, optional
        Количество нейронов в выходном слое.
    device: str, optional
        Устройство, на котором будут происходить расчёты.
    """
    def __init__(self, path_data_model=None, mode='train', batch_size=128,
                 output_size=3, device='cuda'):
        self.mode = mode
        self.batch_size = batch_size
        self.output_size = output_size
        self.device = device
        if self.mode == 'train':
            # загрузка данных для обучения
            self.path_to_data = path_data_model[0]
            self.path_to_model = path_data_model[1]
            train_data = Data_Formation(self.path_to_data,
                                        train=True, one_file=True)
            input_size = train_data[0][0].shape[0]
            train_data, val_data = random_split(train_data, [0.7, 0.3])
            self.train_data_len = len(train_data)
            self.val_data_len = len(val_data)
            
            self.train_load = DataLoader(train_data,
                                         batch_size=self.batch_size,
                                         shuffle=True)
            self.val_load = DataLoader(val_data,
                                       batch_size=self.batch_size,
                                       shuffle=True)
            # создание модели, определение функции потерь для задач классификации и задание оптимизатора.
            self.model = Seq_Model(input_size, self.output_size).to(self.device)
            self.loss_func = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            # списки для сохранения значений функций потерь и метрики
            self.train_loss = []
            self.train_acc = []
            self.val_loss = []
            self.val_acc = []
        else:
            # загрузка обученной модели
            self.path_to_model = path_data_model[1]
            temp_file_list = os.listdir(self.path_to_model)
            for file in temp_file_list:
                if os.path.splitext(file)[-1] == '.pt':
                    self.model_state_dict = torch.load(os.path.join(self.path_to_model, file), map_location=self.device)
                    
            input_size = self.model_state_dict[next(iter(self.model_state_dict.keys()))].shape[1]
            self.model = Seq_Model(input_size, output_size).to(self.device)
            self.loss_func = nn.CrossEntropyLoss()
            self.model.load_state_dict(self.model_state_dict)
            # если необходимы тестовые данные, то загружаем их
            if self.mode == 'test':
                self.path_to_data = path_data_model[0]
                self.test_data = Data_Formation(self.path_to_data,
                                                train=False,
                                                one_file=True)
                self.test_data_len = len(self.test_data)
                
                self.test_load = DataLoader(self.test_data,
                                            batch_size=self.batch_size,
                                            shuffle=True)
                # списки для потерь и метрике на тесте
                self.test_loss = []
                self.test_acc = []
                
    
    def start_fit(self, EPOCHS):
        """
        Основной метод, выполняющий обучение модели.
        
        Параметры:
        ----------
        EPOCHS: int
        Количество эпох для тренировки
        """
        if self.mode != 'train':
            return "Train mode is not used"
        
        best_loss = None
        # цикл тренировки и проверки на валидационнах данных
        for epoch in range(EPOCHS):
            self.train()
            self.validation()
            # проверяем, если на новой эпохе значение функции потерь меньше, то сохраняем модель
            if best_loss is None:
                best_loss = self.val_loss[-1]
                
            if best_loss > self.val_loss[-1]:
                best_loss = self.val_loss[-1]
                is_empty = os.listdir(self.path_to_model)
                if len(is_empty) != 0:
                    os.remove(os.path.join(self.path_to_model, *is_empty))
                torch.save(
                          self.model.state_dict(), 
                          os.path.join(self.path_to_model, 
                                       r'xyd_{}_balanced_{}epoch_{:.3f}loss_{:.3f}acc.pt'.format(self.output_size, 
                                                                           epoch, best_loss, self.val_acc[-1]))
                            )
            print("TOTAL: Epoch [{}/{}],  train_loss: {:.4f}, \train_acc: {:.4f}, \
                  val_loss: {:.4f}, val_acc: {:.4f}".format(epoch + 1, EPOCHS, self.train_loss[-1], 
                                                            self.train_acc[-1], self.val_loss[-1], self.val_acc[-1]))
    
    def train(self):
        """
        Метод, непосредствнно обучающий модель - подгоняющий параметры -
        батча на одной эпохе
        """
        if self.mode != 'train':
            return "Train mode is not used"
        # включаем режим обучения и запускаем стандартный цикл обучения
        self.model.train()
        loop_counter = 0
        loss_in_loop = 0
        true_answer = 0
        # для визуализации бегущей полосы и т.д. используем tqdm
        train_loops = tqdm(self.train_load, leave=False)
        for data, targets in train_loops:
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            pred = self.model(data)
            loss = self.loss_func(pred, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # метрику вычисляем куммулятивно - на каждом батче и суммируем с прошлым значаеним
            loop_counter += 1
            loss_in_loop += loss.item()
            mean_loss_train = loss_in_loop / loop_counter
            
            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            train_loops.set_description('TRAIN: mean_loss: {:.4f}, acc: {:.4f}'.format(mean_loss_train, 
                                        true_answer / (loop_counter * self.batch_size)))
        # заносим значения функции потерь и метрики в список на каждой эпохе
        self.train_loss.append(mean_loss_train)  
        self.train_acc.append(true_answer / self.train_data_len)
    
    def validation(self):
        """
        Метод, непосредствнно прогоняющий батчами модель по валидационным данным
        """
        if self.mode != 'train':
            return "Train mode is not used"
        # включаем режим предсказания и запускаем стандартный цикл прогона по валидационным данным
        self.model.eval()
        with torch.no_grad():
            loop_counter = 0
            loss_in_loop = 0
            true_answer = 0
            # для визуализации бегущей полосы и т.д. используем tqdm
            val_loops = tqdm(self.val_load, leave=False)
            for data, targets in val_loops:
                data = data.to(self.device)
                targets = targets.to(self.device)
                # здесь отличие от цикла обучения только в том, что не делаем обратного распространения ошибки и
                # не корректируем значения параметров
                pred = self.model(data)
                loss = self.loss_func(pred, targets)
                # метрику вычисляем куммулятивно - на каждом батче и суммируем с прошлым значаеним
                loop_counter += 1
                loss_in_loop += loss.item()
                mean_loss_val = loss_in_loop / loop_counter
                
                true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                val_loops.set_description('VALIDATION: mean_loss: {:.4f}, acc:{:.4f}'.format(mean_loss_val, 
                                          true_answer / (loop_counter * self.batch_size)))
            # заносим значения функции потерь и метрики в список на каждой эпохе
            self.val_loss.append(mean_loss_val)  
            self.val_acc.append(true_answer / self.val_data_len)
            
    def testing(self):
        """
        Метод, непосредствнно прогоняющий батчами модель по тестовым данным. Метод идентичен методу валидации
        за тем лишь исключением, что он использует тестовые данные.
        """
        if self.mode != 'test':
            return "Test mode is not used"
        
        self.model.eval()
        with torch.no_grad():
            loop_counter = 0
            loss_in_loop = 0
            true_answer = 0
            test_loops = tqdm(self.test_load, leave=True)
            for data, targets in test_loops:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                pred = self.model(data)
                loss = self.loss_func(pred, targets)
                
                loop_counter += 1
                loss_in_loop += loss.item()
                mean_loss_test = loss_in_loop / loop_counter
                
                true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                test_loops.set_description('TESTING: mean_loss: {:.4f}, acc:{:.4f}'.format(mean_loss_test, 
                                           true_answer / (loop_counter * self.batch_size)))
            self.test_loss.append(mean_loss_test)
            self.test_acc.append(true_answer / self.test_data_len)

    def predict(self, data):
        """
        Метод, выполняющий предсказание модели по предоставленным данным. 
        Может осуществлять предсказания как для данных из тестового нобор с 
        указание предсказанной и правильно меток, так и для неизвестных данных
        с указанием метки класса, к которому они, как считает модель, 
        принадлежат.
        
        Параметры:
        ----------
        data: torch.tensor, list
            Данные для классификации. Могут быть представлены как тензором pytorch, так и списком.
        """
        if self.output_size == 3:
            cl_dict = {0: 'Good', 1: 'Fair', 2: 'Poor'}
        else:
            cl_dict = {0: 'Good', 1: 'No Good'}
        # если данные список, преобразуем из в тензор и придаём необходимую форму
        if self.mode == 'predict':
            if isinstance(data, torch.Tensor) == False:
                data = torch.tensor(data)
            if len(data.shape) == 1:
                data = data.unsqueeze(0)
            # выполняем предсказание для данных
            self.model.eval()
            conf = nn.Softmax(dim=1)(self.model(data[0].unsqueeze(0)))
            cl_ind = torch.argmax(nn.Softmax(dim=1)(self.model(data[0].unsqueeze(0))), dim=1).item()
            print("Prediction is {}\n".format(cl_dict[cl_ind]))
            return {'class': cl_dict[cl_ind],
                    'confidence': '{:.4f}'.format(conf[0,cl_ind].item())}

        if self.mode == 'test':
            # выполняем предсказание для данных из тестового набора
            self.model.eval()
            conf = nn.Softmax(dim=1)(self.model(data[0].unsqueeze(0)))
            cl_ind = torch.argmax(nn.Softmax(dim=1)(self.model(data[0].unsqueeze(0))), dim=1).item()
            print("Prediction is {}; True state is {}\n".format(cl_dict[cl_ind], cl_dict[torch.argmax(data[1]).item()]))
            return {'class': cl_dict[cl_ind],
                    'confidence': '{:.4f}'.format(conf[0, cl_ind].item())}

        return "Wrong mode is used"