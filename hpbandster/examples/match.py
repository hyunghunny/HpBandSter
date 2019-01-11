import lookup as surrogate
import numpy as np

class SurrogateMatcher(object):
    def __init__(self, lookup):
        self.lookup = lookup

    def find_nearest(self, config):
        candidates = []
        
        hpvs = self.lookup.get_hyperparam_vectors()
        # XXX:hard coding for data2
        param_orders = ["c1_depth", "p1_size", "c2_depth", 
        "p2_size", "f1_width", "window_size",
        "learning_rate", "reg_param", "keep_prop_rate"] 
        
        for i in range(len(hpvs)):
            match_count = 0
            hpv = hpvs[i]
            # integer value may match exactly
            if abs(hpv[0] - config['c1_depth']) < 50:
                match_count += 1
            if abs(hpv[2] - config['c2_depth']) < 50:
                match_count += 1            
            if abs(hpv[4] - config['f1_width']) < 100:
                match_count += 1
            if hpv[5] == config['window_size']:
                match_count += 1
            if hpv[1] == config['p1_size']:
                match_count += 1    
            if hpv[3] == config['p2_size']:
                match_count += 1 
            
            # float value may match almost similar
            if abs(hpv[6] - config['learning_rate']) < 0.01:
                match_count += 1
            if abs(hpv[7] - config['reg_param']) < 0.2:
                match_count += 1
            if abs(hpv[8] - config['keep_prop_rate']) < 0.2:
                match_count += 1
            
            candidates.append({"index": i, "hyperparam" : hpv, "rating": match_count})
        
        ratings = []
        for c in candidates:
            ratings.append(c["rating"])
        
        max_index = np.argmax(ratings)
        return candidates[max_index]["hyperparam"],


if __name__ == "__main__":
    s = surrogate.load("data2", 
        data_folder='hpbandster/examples/lookup/', config_folder='hpbandster/examples/hp_conf/')
    c = { 
            "c1_depth": 130, 
            "p1_size": 2, 
            "c2_depth" : 220,
            "p2_size": 3, 
            "f1_width": 600, 
            "window_size": 4,
            "learning_rate": 0.051, 
            "reg_param": 0.4, 
            "keep_prop_rate": 0.3
        }
    m = SurrogateMatcher(s)
    hpv, i = m.find_nearest(c)
    errs = s.get_test_errors()
    err = errs[i] 
    print("hpv: {}, error: {}".format(hpv, err))
    pass