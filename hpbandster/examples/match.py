import lookup as surrogate
import numpy as np

class Data2SurrogateMatcher(object):
    def __init__(self, lookup, used_lookup_index=[]):
        self.lookup = lookup
        self.referred_indices = used_lookup_index

    def find_naive(self, config):
        candidates = []
        
        hpvs = self.lookup.get_hyperparam_vectors()

        for i in range(len(hpvs)):
            match_count = 0
            hpv = hpvs[i]
            # integer value may match exactly
            if abs(hpv[0] - config['c1_depth']) < 20:
                match_count += 1
            if abs(hpv[2] - config['c2_depth']) < 20:
                match_count += 1            
            if abs(hpv[4] - config['f1_width']) < 50:
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
        return candidates[max_index]["hyperparam"], max_index

    def get_vector(self, config):
        vec = np.zeros(9)

        vec[0] = float(config['c1_depth'])
        vec[1] = float(config['p1_size'])
        vec[2] = float(config['c2_depth'])
        vec[3] = float(config['p2_size'])
        vec[4] = float(config['f1_width'])
        vec[5] = float(config['window_size'])
        vec[6] = float(config['learning_rate'])
        vec[7] = float(config['reg_param'])
        vec[8] = float(config['keep_prop_rate'])

        return vec

    def normalize_vector(self, vec):
        norm_vec = np.zeros(9)
        norm_vec[0] = float((vec[0] - 1) / (350 - 1))
        norm_vec[1] = float((vec[1] - 2))
        norm_vec[2] = float((vec[2] - 1) / (350 - 1))
        norm_vec[3] = float((vec[3] - 2))
        norm_vec[5] = float((vec[4] - 1) / 1023)
        norm_vec[4] = float((vec[5] - 2) / 8)
        norm_vec[6] = float((vec[6] - 0.0001) / (10**-0.5 - (10**-4)))
        norm_vec[7] = float((vec[7]))
        norm_vec[8] = float((vec[8] - 0.1) / 0.9)

        return norm_vec

    def find_nearest(self, config):

        hpvs = self.lookup.get_hyperparam_vectors()
        num_samples = len(hpvs)
        candidates = np.setdiff1d(np.arange(num_samples), self.referred_indices)
        selected_vec = self.get_vector(config)
        snv = self.normalize_vector(selected_vec)
        min_dist = 1000.0
        closest_index = None
        for c in candidates:
            hpv = hpvs[c]
            cnv = self.normalize_vector(hpv)
            dist = np.linalg.norm(snv-cnv)
            if dist < min_dist:
                min_dist = dist
                closest_index = c

        return hpvs[closest_index], closest_index


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
    m = Data2SurrogateMatcher(s)
    hpv, i = m.find_nearest(c)
    errs = s.get_test_errors()
    err = errs[i] 
    print("hpv: {}, error: {}".format(hpv, err))
    pass