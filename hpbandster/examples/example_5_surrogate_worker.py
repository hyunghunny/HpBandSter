import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)

from lookup import *
from match import Data2SurrogateMatcher

class SurrogateWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, **kwargs):
        # TODO:read surrogates from lookup directory
        self.lookup = load('data2', data_folder='./lookup/')            
        super().__init__(**kwargs)
                                 

    def compute(self, config, budget, working_directory, *args, **kwargs):
        num_epochs = budget
        num_params = self.lookup.num_hyperparams

        # Find the approximated configuration index here
        matcher = Data2SurrogateMatcher(self.lookup)
        # XXX:hard coding for data2
        param_orders = ["c1_depth", "p1_size", "c2_depth",
        "p2_size", "f1_width", "window_size",
        "learning_rate", "reg_param", "keep_prop_rate"]
        try:
            target_lookup_index = kwargs["lookup_index"]["lookup_index"]
            hpv = self.lookup.get_hyperparam_vectors()[target_lookup_index]
        except:
            hpv, target_lookup_index = matcher.find_nearest(config)

        test_err = self.lookup.get_test_errors(num_epochs)[target_lookup_index]
        elapsed_time = self.lookup.get_elapsed_times(num_epochs)[target_lookup_index]
        hpv_dict = {}
        hpv_list = hpv.tolist()
        for i in range(len(hpv_list)):
            val = hpv_list[i]
            key = param_orders[i]
            if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5:
                hpv_dict[key] = int(val)
            else:
                hpv_dict[key] = float(val)

        #import IPython; IPython.embed()
        return ({
            'loss': test_err, # remember: HpBandSter always minimizes!
            'info': {
                        'test accuracy': 1.0 - test_err,
                        'elapsed time' : elapsed_time,
						'hyperparams' : hpv_dict,
                        'lookup index' : int(target_lookup_index)
                    }
        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=0.316, default_value='1e-2', log=True)
        reg_param = CSH.UniformFloatHyperparameter('reg_param', lower=0.0, upper=1.0, default_value=0.5, log=False)
        keep_prop_rate = CSH.UniformFloatHyperparameter('keep_prop_rate', lower=0.1, upper=1.0, default_value=0.5, log=False)
        cs.add_hyperparameters([learning_rate, reg_param, keep_prop_rate])
        
        c1_depth = CSH.UniformIntegerHyperparameter('c1_depth', lower=1, upper=350, default_value=32, log=False)
        p1_size = CSH.UniformIntegerHyperparameter('p1_size', lower=2, upper=3, default_value=2, log=False)
        c2_depth = CSH.UniformIntegerHyperparameter('c2_depth', lower=1, upper=350, default_value=64, log=False)
        p2_size = CSH.UniformIntegerHyperparameter('p2_size', lower=2, upper=3, default_value=2, log=False)
        window_size = CSH.UniformIntegerHyperparameter('window_size', lower=2, upper=10, default_value=2, log=False)
        f1_width = CSH.UniformIntegerHyperparameter('f1_width', lower=1, upper=1024, default_value=512, log=False)

        cs.add_hyperparameters([c1_depth, p1_size, c2_depth, p2_size, window_size, f1_width])

        return cs        