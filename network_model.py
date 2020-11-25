from tensorboard.plugins.hparams import api as hp
import run_model

class NetworkModel(run_model.RunModel):
    def __init__(self,model_num=0,input_size=0,num_units=None,DO_val = None, metric = None ,train_dataset=None, test_dataset=None, epochs =0,
                 callback=None):
        super().__init__(model_num,input_size,num_units,DO_val,metric,train_dataset,test_dataset,epochs,callback)


    def set_param(self,num_units,DO_val,metric):
        self.HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete(num_units))
        self.HP_DROPOUT = hp.HParam('dropout', hp.Discrete(DO_val))
        self.METRIC_ACCURACY = metric

    def set_model_num(self,model_num):
        self.model_num=model_num

    def get_param(self):
        return self.HP_NUM_UNITS, self.HP_DROPOUT, self.METRIC_ACCURACY

    def set_train_data(self, data):
       self.train_dataset = data

    def get_training_data(self):
        return self.train_dataset

    def set_test_data(self,data):
        self.test_dataset = data

    def get_test_data(self):
        return self.test_dataset

    def set_epochs(self, epochs):
        self.epochs = epochs

    def get_epochs(self):
        return self.epochs

    def set_callback(self, callback):
        self.callback=callback

    def get_callback(self):
        return self.callback


