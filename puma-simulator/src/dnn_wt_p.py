import os
import sys
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
include_dir = os.path.join(root_dir, "include")
sys.path.insert(0, include_dir)


import config as cfg

class dnn_wt:

    def prog_dnn_wt(self, instrnpath, core_dut):  

        ## Program DNN weights on the xbars
        for k in range(cfg.num_matrix):
            for l in range(cfg.phy2log_ratio):
                wt_filename = instrnpath + 'weights/mat'+str(k)+'-phy_xbar'+str(l)+'.npy'
                if (os.path.exists(wt_filename)):  # check if weights for the xbar exist
                    print ('wtfile exits: ' + 'matrix ' + str(k) + 'xbar' + str(l))
                    wt_temp = np.load(wt_filename)
                    core_dut.matrix_list[k]['f'][l].program(wt_temp)
                    core_dut.matrix_list[k]['b'][l].program(wt_temp)


