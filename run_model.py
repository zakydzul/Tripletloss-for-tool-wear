import tensorflow as tf
from model_arc import model
from tensorboard.plugins.hparams import api as hp

class RunModel(model):
    def __init__(self,model_num=0, input_size=0, num_units=None, DO_val=None, metric=None, train_dataset=None, test_dataset=None,
                 epochs=0,callback=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.callback = callback
        self.model_num = model_num
        super().__init__(input_size,num_units,DO_val,metric)

    def model(self, hparams):
        if self.model_num==0:
            return self.model_seq(hparams)
        else:
            return self.model_res_net(hparams)


    def get_model(self, hp_param_no=0, do_param_no=0):
        model = self.model({
              self.HP_NUM_UNITS: self.HP_NUM_UNITS.domain.values[hp_param_no],
              self.HP_DROPOUT: self.HP_DROPOUT.domain.values[do_param_no],
          })
        return model

    def single_run(self, model):
        model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            callbacks=[self.callback],
            verbose=0,
            epochs = self.epochs
        )

    def file_write(self,log_name='logs'):
            with tf.summary.create_file_writer(log_name + '/hparam_tuning').as_default():
                hp.hparams_config(
                    hparams=[self.HP_NUM_UNITS, self.HP_DROPOUT],
                    metrics=[hp.Metric(self.METRIC_ACCURACY, display_name='val_loss')],
                )

    def multiple_run_model(self, hparams):
        model = self.model(hparams)
        history = model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            callbacks=[self.callback],
            verbose=0,
            epochs=self.epochs)
        print('val_loss :{:.7}'.format(str(history.history['val_loss'][-1])))
        return history

    def run(self, run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            val_loss = self.multiple_run_model(hparams)
            tf.summary.scalar(self.METRIC_ACCURACY, val_loss.history['val_loss'][-1], step=1)

    def multiple_run(self,HP_NUM_UNITS,HP_DROPOUT,log_name='logs'):
        self.file_write(log_name)
        session_num = 0
        for num_units in HP_NUM_UNITS.domain.values:
            for dropout_rate in HP_DROPOUT.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                }
                run_name = "run-%d" % session_num +'_'+str(self.model_num)
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                self.run(log_name + '/hparam_tuning/' + run_name, hparams)
                session_num += 1