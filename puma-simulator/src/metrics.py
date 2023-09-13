# APIs to compute ima power and area stats

import sys

# import dependency files
import config as cfg
import reram.puma-simulator.include.constants as param

# Compute metrics of the ima based on paramaters in config file and dicts in constants file
# Area is computed as the summation of all component area (doesn't consider physical layout)
def compute_area (): #in mm2
    area = 0.0
    if cfg.MVMU_ver == "Analog":
        area += (cfg.num_matrix*11) * cfg.xbar_size * param.dac_area # 1 dac for input of f/b/d xbars, each phy xbar in d-xbar will have a dac_array, hence 8
        area += (cfg.num_matrix*2) * cfg.xbar_size * param.snh_area # snh for f/b xbars
        area += (cfg.num_matrix*2) * param.sna_area # sna for one each f/b xbars
        area += cfg.num_adc * param.adc_area # adc
        area += (cfg.num_matrix*3) * param.xbar_outMem_area # xbar_outMem (1 OR for 8 xbars - 16 bit weights, 2 bit xbars)
        area += (cfg.num_matrix*4) * cfg.phy2log_ratio * param.xbar_area # d-xbar has 2X xbars than f/b
    else:
        area += (cfg.num_matrix*2) * param.xbar_area # d-xbar are not needed in Digital MVMUs only f and b are there
    area += (cfg.num_matrix*3) * param.xbar_inMem_area # xbar_inMem one each for f/b/d xbars
    area += param.instrnMem_area # instrnMem
    area += param.dataMem_area # dataMem
    area += param.alu_area # alu
    area += param.act_area # activation (sigmoid)
    area += param.ccu_area
    #print ('6 IMA area: ' + str (6 * area) + ' mm2')

    ## Comapre with ISSAC area for iso-xbars (computational efficiency - ops/mm2)
    #ima_issac_area = 0.157/12.0 + 8 * 1*param.xbar_inMem_area - 0.0021 #- 0.00077 # 8 * param.xbar_inMem_area is area of
    ## 2KB RAM in an IMA - acc to CACTI 6.0
    #print ('12 IMA area ISSAC: ' + str (12 * ima_issac_area) + ' mm2')
    #print ('Area increase for iso xbars: ' + str((6*area - 12*ima_issac_area) / (12*ima_issac_area)*100) + ' %')
    return area

# Leakage power is computed as sum of leakage powers of all components
def compute_pow_leak ():
    leak_pow = 0.0
    if cfg.MVMU_ver == "Analog":
        leak_pow += (cfg.num_matrix*11) * cfg.xbar_size * param.dac_pow_leak # dac
        leak_pow += (cfg.num_matrix*2) * cfg.xbar_size * param.snh_pow_leak # snh
        leak_pow += cfg.num_adc * param.adc_pow_leak # adc
        leak_pow += (cfg.num_matrix*2) * param.sna_pow_leak # sna
        leak_pow += (cfg.num_matrix*3) * param.xbar_outMem_pow_leak # xbar_outMem
        leak_pow += (cfg.num_matrix*4) * param.xbar_pow_leak # xbar power fr analog
    else:
        leak_pow += (cfg.num_matrix*2) * param.xbar_pow_leak # d-xbar are not needed in Digital MVMUs only f and b are there
    leak_pow += (cfg.num_matrix*3) * param.xbar_inMem_pow_leak # xbar_inMem
    leak_pow += param.instrnMem_pow_leak # instrnMem
    leak_pow += param.dataMem_pow_leak # dataMem
    leak_pow += param.alu_pow_leak # alu
    leak_pow += param.act_pow_leak # activation (sigmoid) areacompute_leak_pow: # in mw
    # print ('IMA leak power: ' + str (leak_pow) + ' mW')
    return leak_pow

# Peak dynamic power (assumes all components are being accessed in each cycle)
def compute_pow_dyn ():
    dyn_pow = 0.0
    if cfg.MVMU_ver == "Analog":
    # dyn_pow += cfg.num_xbar/2 * 1.2 # (adding dyn pow the way issac does for comparison)
        dyn_pow += (cfg.num_matrix*11) * cfg.xbar_size * param.dac_pow_dyn # dac
        dyn_pow += (cfg.num_matrix*2) * cfg.xbar_size * param.snh_pow_dyn # snh
        dyn_pow += cfg.num_adc * param.adc_pow_dyn # adc
        dyn_pow += (cfg.num_matrix*2) * param.sna_pow_dyn # sna
        dyn_pow += (cfg.num_matrix*3) * param.xbar_outMem_pow_dyn # xbar_outMem (1 OR for 8 xbars - 16 bit weights, 2 bit xbars)
        dyn_pow += (cfg.num_matrix*4) * param.xbar_ip_pow_dyn # xbar ip power considred as ip>op power
    else:
        dyn_pow += (cfg.num_matrix*2) * param.xbar_ip_pow_dyn # xbar ip power considred as ip>op power # d-xbar are not needed in Digital MVMUs only f and b are there
    dyn_pow += (cfg.num_matrix*3) * (param.xbar_inMem_pow_dyn_write + param.xbar_inMem_pow_dyn_read/cfg.xbar_size) # xbar_inMem - num_xbar * dac_res bits will be
        #   read from xb_inMem in an interval that equals xbar_access time
    dyn_pow += param.instrnMem_pow_dyn # instrnMem
    dyn_pow += param.dataMem_pow_dyn # dataMem
    dyn_pow += param.alu_pow_dyn # alu
    dyn_pow += param.act_pow_dyn # activation (sigmoid) areacompute_leak_pow: # in mw
    # print ('IMA dyn power: ' + str (dyn_pow) + ' mW')
    return dyn_pow

# Peak power of ima - leak pow + peak_dyn power
def compute_pow_peak ():
    peak_pow = compute_pow_leak() + compute_pow_dyn()
    #print ('6 IMA peak (leak+dyn) power: ' + str (6 * peak_pow) + ' mW')

    ## Compare with ISSAC for iso-xbars (computational effciiency - ops/mm2)
    #ima_issac_pow = 289/12.0 # 12 IMA
    #print ('12 IMA power ISSAC: ' + str (12 * ima_issac_pow) + ' mW')
    #print ('Peak power increase for iso xbars: ' + str((6 *peak_pow - 12*ima_issac_pow) / (12*ima_issac_pow)*100) + ' %')
    return peak_pow

#compute_area()
#compute_pow_leak()
#compute_pow_dyn()
#compute_pow_peak()

# Computes the latency for Analog mvm instruction based on DPE configuration
def xbComputeLatency_Analog (self, mask):
    latency_out_list = []
    fb_found = 0
    d_found = 0
    latency_out_list = []
    for idx, temp in enumerate(mask):
        print("idx", idx)
        if ((temp[0] == '1') or (temp[1] == '1')):
            fb_found += 1
            #break
        if (temp[2] == '1'):
            d_found += 1
            #break

        ## MVM inner product goes through a 3 stage pipeline (each stage consumes 128 cycles - xbar aces latency)
        # Cycle1 - xbar_inMem + DAC + XBar
        # Cycle2 - SnH + ADC
        # Cycle3 - SnA + xbar_outMem
        # The above pipeline is valid for one ADC per physical xbar only !! (Update for other cases, if required)
        num_stage = 3
        #lat_temp = self.matrix_list[0]['f'][0].getIpLatency() # due to xbar access
        lat_temp = 0
        # We assume all ADCs in a matrix has the same resolution
        adc_idx = idx*cfg.num_adc_per_matrix
        lat_temp = self.adc_list[adc_idx].getLatency()
        '''
        print("adc_idx", adc_idx)
        print("lat_temp", lat_temp)
        print("self.adc_list[adc_idx].adc_res", self.adc_list[adc_idx].adc_res)
        for adccccc in self.adc_list:
            print("adccccc.adc_res", adccccc.adc_res)
        print("---")
        '''
        latency_ip = lat_temp * ((cfg.input_prec / cfg.dac_res) + num_stage - 1) * float(int(fb_found>0))*(math.ceil(float(cfg.weight_width)/ \
        cfg.xbar_bits) /math.ceil(float(cfg.data_width)/cfg.xbar_bits)) # last term to account for the effect of quantization on latency
        ## MVM outer product occurs in 4 cycles to take care of all i/o polarities (++, +-, -+, --)
        num_phase = 4
        lat_temp = self.matrix_list[0]['f'][0].getOpLatency()
        #latency_op = lat_temp * num_phase * d_found
        latency_op = lat_temp * num_phase * float(int(d_found>0))
        ## output latency should be the max of ip/op operation
        latency_out = max(latency_ip, latency_op)
        print ("Mask", mask)
        print ("Latency IP", latency_ip)
        print ("Latency OP", latency_op)
        print ("latency_out", latency_out)
        latency_out_list.append(latency_out)
    return max(latency_out_list)