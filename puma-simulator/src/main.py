import sys
import os
import numpy as np

SIMULATOR_PATH="home/puma/simulator"
sys.path.insert (0, SIMULATOR_PATH + '/include/')
sys.path.insert (0, SIMULATOR_PATH + '/src/') 
sys.path.insert (0, SIMULATOR_PATH +'/')

THIS_PATH = os.getcwd()


from data_convert import *
import ima as ima
from instrn_proto import *
import config as cfg
from dyn_block import *


#path = 'coreMvm_test/'
#wt_path = path
#inst_file = path + 'imem1.npy'
#trace_file = path + 'trace.txt'
#dump_file = path + 'memsim.txt'

datamem_off = cfg.datamem_off # each matrix has 6 memory spaces (2 input buff, 2 output buff, 2 mvm)
phy2log_ratio = cfg.phy2log_ratio # ratio of physical to logical xbar
xbar_sizex = cfg.xbar_sizex
xbar_sizey = cfg.xbar_sizey

weight_files =[]

def populate():
    for i in os.listdir(THIS_PATH):
        if i.endswith('.weights'):
            dataset = i.split('-')[0]
            mat_id = i.partition('mvmu')[2][0]
            os.system('mkdir -p ' + dataset + '/weights')
            wt_path = dataset + '/weights/'
            #print(wt_path)
            os.system('cp '+ i + ' ' + dataset + '/weights')
            with open(i) as f:
                line = f.readline()
                arr = np.fromstring(line, dtype=float, sep=' ')
                log_xbar = np.reshape(arr, (xbar_sizex, xbar_sizey))
                phy_xbar = [np.random.randn(xbar_sizex, xbar_sizey) for i in range(cfg.phy2log_ratio)]

    ## NOTE: weights programmed to xbars are stored in terms of their representative floating values
    ## for use in np.dot (to store bits representation, use fixed point version of np.dot)
                for i in range (xbar_sizex):
                    for j in range (xbar_sizey):
                        temp_val = float2fixed(log_xbar[i][j], cfg.int_bits, cfg.frac_bits)
                        assert (len(temp_val) == 16)
                        for k in range (len(phy_xbar)):
                            if (k==0):
                                val = temp_val[-(k+1)*cfg.xbar_bits:]
                            else:
                                val = temp_val[-(k+1)*cfg.xbar_bits:-(k+1)*cfg.xbar_bits+2]
                            # augment sign extension (used in MSB xbar only)
                            if (k == (len(phy_xbar)-1)):
                                val = (cfg.num_bits - cfg.xbar_bits)*val[0] + val[0:]
                            phy_xbar[k][i][j] = fixed2float(val, cfg.int_bits, cfg.frac_bits)
                # save log_xbar and phy_xbar to disc
                np.save (wt_path+'log_xbar'+str(mat_id), log_xbar)
                for k in range (len(phy_xbar)):
                    np.save (wt_path+'mat'+str(mat_id)+'-phy_xbar'+str(k), phy_xbar[k])

def main():
    transformer_encoder()
    transformer_decoder()
    lightcnn()
    lightdnn()