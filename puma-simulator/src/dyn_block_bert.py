import sys, json

# import dependancy files
import numpy as np
import math
import include.config as cfg
#import include.configTest as cfg
import include.constants as param
import constants_digital as digi_param
import src.ima_modules as imod

from data_convert import *

phy2log_ratio = cfg.phy2log_ratio # ratio of physical to logical xbar
# datamem_off is the start of address sapce of datamemory
datamem_off = cfg.datamem_off # each matrix has 6 memory spaces (1 for f/b, 2 for d)
my_xbar_count = 0


def norm():
    # prepare residual

    # min, max

    # div


def QKT_d(self, mask, mat_id, key):
    # reset the xb out memory before starting to accumulate
    self.xb_outMem_list[mat_id][key].reset ()

    sparsity=0
    sparsity_adc=0
    if cfg.sparse_opt:
        xbar_inMem = self.xb_inMem_list[mat_id][key].read_all ()
        non_0_val = 0
        for i in range(cfg.xbar_sizey):
            if xbar_inMem[i] != '0000000000000000':
                non_0_val = non_0_val +1
        sparsity = int((cfg.xbar_size-non_0_val)*100.0/cfg.xbar_size)
        sparsity_adc = sparsity
        if (sparsity%10!=0):
            sparsity = sparsity-(sparsity%10)
        else:
            if (sparsity == 100):
                sparsity = sparsity-10

    ## Loop to cover all bits of inputs
    for k in xrange (int(math.ceil(cfg.input_prec / cfg.dac_res))): #quantization affects the # of streams
    #for k in xrange (1):
        # read the values from the xbar's input register
        out_xb_inMem = self.xb_inMem_list[mat_id][key].read (cfg.dac_res)
        
        #*************************************** HACK *********************************************
        ###### CAUTION: Not replicated exact "functional" circuit behaviour for analog parts
        ###### Use propagate (not propagate_hack) for DAC, Xbar, TIA, SNH, ADC when above is done
        #*************************************** HACK *********************************************

        # convert digital values to analog
        out_dac = self.dacArray_list[mat_id][key].propagate_dummy(out_xb_inMem) #pass through

        # Do for (data_width/xbar_bits) xbars
        num_xb = int(math.ceil(float(cfg.weight_width) / cfg.xbar_bits))  # # of XBs change with quantization
        out_xbar = [[] for x in range(num_xb)]
        out_snh = [[] for x in range(num_xb)]
        for m in range (num_xb):
            # compute dot-product
            out_xbar[m] = self.matrix_list[mat_id][key][m].propagate_dummy(out_dac, sparsity)        
            # do sampling and hold
            out_snh[m] = self.snh_list[mat_id*num_xb+m].propagate_dummy(out_xbar[m])

        # each of the num_xb produce shifted bits of output (weight bits have been distributed)
        for j in xrange (cfg.xbar_sizex): # this 'for' across xbar outs to adc happens via mux
            #out_sna = '0'*cfg.data_width # a zero for first sna
            out_sna = 0.0 # a zero for first sna
            for m in range (num_xb):
                # convert from analog to digital
                adc_id = (mat_id*num_xb + m) % cfg.num_adc
                out_mux1 = self.mux1_list[mat_id].propagate_dummy(out_snh[m][j]) # i is the ith xbar
                out_mux2 = self.mux2_list[mat_id % cfg.num_adc].propagate_dummy(out_mux1)
                out_adc = self.adc_list[adc_id].propagate_dummy(out_mux2, sparsity_adc)

                # shift and add outputs from different wt_bits
                alu_op = 'sna'
                [out_sna, ovf] = self.alu_list[0].propagate_float (out_sna, out_adc, alu_op, \
                        m*cfg.xbar_bits)

            # convert the inter-xbar sna output to fixed hereon
            out_sna = float2fixed(out_sna, cfg.int_bits, cfg.frac_bits)
            # read from xbar's output register
            out_xb_outMem = self.xb_outMem_list[mat_id][key].read (j)
            # shift and add - make a dedicated sna unit -- PENDING
            alu_op = 'sna'
            # modify (len(out_adc) to adc_res) when ADC functionality is implemented
            [out_sna, ovf] = self.alu_list[0].propagate (out_xb_outMem, out_sna, alu_op, k*cfg.dac_res)
            if (cfg.debug and ovf):
                fid.write ('IMA: ' + str(self.ima_id) + ' ALU Overflow Exception ' +\
                        self.de_aluop + ' allowed to run')
            # store back to xbar's output register & restart it
            self.xb_outMem_list[mat_id][key].write (out_sna)
        self.xb_outMem_list[mat_id][key].restart()

    # stride the inputs if applicable
    self.xb_inMem_list[mat_id][key].stride(self.de_val1, self.de_val2)


def nonlinear():
    # min, max

    # lut

def mulV():

def residual():
    # aggregate if after attention

    # residual add


def feedforward():
    # mvm


def conv():


def transformer_encoder():
    ## Traverse through the matrices in a core
    for i in xrange(cfg.num_matrix):
        print ("ima_id: " +str(self.ima_id) + " mat_id: "  +str(i) + " MVM")
        norm()
        QKT_d(i,'f')
        nonlinear(i,'softmax')
        mulV()
        residual()
        norm()
        feedforward()
        residual
        

def transformer_decoder():
    ## Traverse through the matrices in a core
    for i in xrange(cfg.num_matrix):
        print ("ima_id: " +str(self.ima_id) + " mat_id: "  +str(i) + " MVM")
        norm()
        QKT_d(i,'f')
        nonlinear(i,'softmax')
        mulV(mask)
        residual()
        norm()
        feedforward()
        residual

def lightcnn():
    ## Traverse through the matrices in a core
    for i in xrange(cfg.num_matrix):
        print ("ima_id: " +str(self.ima_id) + " mat_id: "  +str(i) + " MVM")
        conv(i,'f')
        nonlinear(i,'relu')

        
def lightdnn():
    ## Traverse through the matrices in a core
    for i in xrange(cfg.num_matrix):
        print ("ima_id: " +str(self.ima_id) + " mat_id: "  +str(i) + " MVM")
        feedforward(i,'f')