from abc import abstractmethod
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
pd.DataFrame
class NitDataset:
    def __init__(self, dataset_name, cache_dir):
        self.dataset = load_dataset(dataset_name, cache_dir = cache_dir)
    
    def getDataset(self):
        return self.dataset
    
    def getTrain(self):
        return self.dataset['train']
    
    def getTest(self):
        if 'test' not in self.dataset:
            return None
        return self.dataset['test']
    
    def getValidation(self):
        if 'validation' not in self.dataset:
            return None
        return self.dataset['validation']
    
    def asDataframe(self, split = 'train'):
        if split == 'train':
            return self.getTrain().to_pandas() if self.getTrain() is not None else None
        elif split == 'test':
            return self.getTest().to_pandas() if self.getTest() is not None else None
        elif split == 'validation':
            return self.getValidation().to_pandas() if self.getValidation() is not None else None
        elif split == 'all':
            return self.getTrain().to_pandas() if self.getTrain() is not None else None, self.getTest().to_pandas() if self.getTest() is not None else None, self.getValidation().to_pandas() if self.getValidation() is not None else None
    
    def updateDataset(self, dataset):
        self.dataset['train'] = dataset['train']
    
    def getColumns(self):
        return self.dataset.column_names
    
    def getDataloaders(self, columns, batch_size, test_size=0.2, validation_size=0):
        raise ValueError("Under Development")
        if not (0 <= test_size <= 1):
            raise ValueError("test_size must be between 0 and 1")
        
        if not (0 <= validation_size <= 1):
            raise ValueError("validation_size must be between 0 and 1")

        validation_dataset = self.getValidation()
        test_dataset = self.getTest()
        train_dataset = self.getTrain()

        if train_dataset is None:
            raise ValueError("Train dataset is empty")
        
        all_columns = self.getColumns()
        columns_to_remove = [column for column in all_columns if column not in columns]
        train_dataset = train_dataset.remove_columns(columns_to_remove)
        if validation_dataset:
            validation_dataset = validation_dataset.remove_columns(columns_to_remove)
        if test_dataset:
            test_dataset = test_dataset.remove_columns(columns_to_remove)

        if test_size * validation_size == 0:
            pd_dataset = self.asDataframe("train")[columns]
            return DataLoader(pd_dataset, batch_size=batch_size, shuffle=True)

        if validation_size > 0 and test_size > 0:
            if validation_dataset and test_dataset:
                return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True), 
                        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False), 
                        DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
            elif validation_dataset:
                train_test = train_dataset.train_test_split(test_size=test_size)
                return (DataLoader(train_test['train'], batch_size=batch_size, shuffle=True), 
                        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False), 
                        DataLoader(train_test['test'], batch_size=batch_size, shuffle=False))
            elif test_dataset:
                train_val = train_dataset.train_test_split(test_size=validation_size)
                return (DataLoader(train_val['train'], batch_size=batch_size, shuffle=True), 
                        DataLoader(train_val['test'], batch_size=batch_size, shuffle=False), 
                        DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
            else:
                train_val_test = train_dataset.train_test_split(test_size=test_size, validation_size=validation_size)
                return (DataLoader(train_val_test['train'], batch_size=batch_size, shuffle=True), 
                        DataLoader(train_val_test['validation'], batch_size=batch_size, shuffle=False), 
                        DataLoader(train_val_test['test'], batch_size=batch_size, shuffle=False))
        
                
        
                
                
            
