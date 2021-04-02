import pandas as pd
import numpy as np
import sys
import math
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import random
import gc
from joblib import Parallel, delayed
import multiprocessing   


def computeTransportRxnTimes(path,expt_start,expt_end,avg=False, cogtRNANum = 1, ribosomeNum = 1, scaling=1, NR_scaling = {'k1r':718.0,'k2f':1475.0,'k2r_nr':1120.0,'k3_nr':6.0,'k4':209.0}):
    """Calculates transport (how long particular tRNA unbound) and 
    reaction times (how long particular tRNA bound) from simulations
    
    Arguments:
        path {[type]} -- String to output folder
        simtime {[type]} -- Length of time over which to 
        num_rib {[type]} -- [description]
        expt_start {[type]} -- [description]
        expt_end {[type]} -- [description]
    
    Keyword Arguments:
        avg {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    df_outputs = pd.read_csv(path+"outputReactionsList.txt",sep=" ",header=None) #Add batch processing here potentially
    transport_time = list()
    reaction_time = list()
    search_time = list()
    success_incorp = list()
    rxn17_tot = list()
    rxn21_tot = list()
    ribosome_reaction_time = list()
    print("Computing...") 
    NR_tRNA = int(round(8/42*(42-(cogtRNANum-2))))+(ribosomeNum-1) #Non-matching ribosomes make up the first ribosomeNum-1 labels
    NR_SCALINGFACTOR = 3.3 #computeNRLatency(NR_scaling)/(1000/NR_scaling['k1r']) #Scaling factor for how much slower near cognate mismatch reactions are compared to non cognate mismatches
    reactantarray = list()
    #scaling = scaling*(8/40*4.6+32/40*1.4)/1.4 ##Adjust scaling to account for near-cognate ternary complexes
    for expt_num, row in df_outputs.iterrows():
        succincorp_count = 0
        rxn17_count = 0
        rxn21_count = 0
        if(expt_num>=expt_start and expt_num<expt_end):
            try:
                my_cols=["time","rxn","x","y","z","reactantA","productA","productB","productC"]
                df = pd.read_csv(path+row[0],delimiter=" ",header=None, names=my_cols,dtype={'reactantA':str,'productA':str,'productB':str,'productC':str})

                df=df.loc[df['rxn'].isin(["rxn17","rxn18","rxn19","rxn20","rxn21","rxn22","rxn23","rxn24","rxn26"])]

                ##Gets the id of which cognate tRNA succesfully bound to ribosomes (needed in cases where more than one cognate tRNA in voxel)
                df_succ_tRNA_id = int(df.loc[df['rxn'] == 'rxn18']['reactantA'].values[0].split('.')[0])
                df_succ_ribosome_id = int(df.loc[df['rxn'] == 'rxn18']['reactantA'].values[0].split('.')[1])

                df_rib = df[df['rxn'].isin(["rxn17","rxn18","rxn23","rxn24"])]
                df=df[df['reactantA'].apply(str).str.split('.').str[0].apply(int)==df_succ_tRNA_id]

                df=df[['time','rxn', 'reactantA','productA']]
                df_rib = df_rib[['time','rxn','reactantA','productA']]

                ## Calculate elong time from tracking succesful ribosome
                rib_reaction_time_i = list()
                rib_unbound_time_i = list()
                i=-1
                for _,row in df_rib.iterrows():
                    i+=1
                    if(row['rxn']=='rxn23'):
                        if(int(row['reactantA'])<=NR_tRNA):
                           # reactantarray.append(int(row['reactantA']))
                            rib_reaction_time_i.append(NR_SCALINGFACTOR*(float(df_rib.iloc[[i+1]]['time'])-row['time']))
                        else:
                            rib_reaction_time_i.append(1*(float(df_rib.iloc[[i+1]]['time'])-row['time']))
                    if(row['rxn']=='rxn24'):
                        rib_unbound_time_i.append(float(df_rib.iloc[[i+1]]['time'])-row['time'])
                    if(row['rxn']=='rxn17'):
                         if(i==0):
                             rib_unbound_time_i.append(float(row['time']))
                         rib_reaction_time_i.append(float(df_rib.iloc[[i+1]]['time'])-row['time'])
                #print(np.sum(rib_reaction_time_i),'\n',np.sum(rib_unbound_time_i))

                transport_time_i = list()
                reaction_time_i = list()
                i=-1
                single_RxnTime = list() #Created to aggregate transport time between unsuccesful rxns into those btwn only succesful rxns
                single_TransportTime = list() #Created to aggregate transport time between unsuccesful rxns into those btwn only succesful rxns
                for _, row in df.iterrows():
                    i+=1
                    if(row["rxn"]=='rxn20'):
                        break
                    ### If the rxn is the cognate tRNA binding to a ribosome (can be a cognate or non-cognate binding reaction)
                    if((row["rxn"]=='rxn17' or row["rxn"]=='rxn21' or row["rxn"]=='rxn26')): #and succincorp_count<num_rib):
                        if(i>0):
                            ## Compute the time between the binding and when the tRNA was last let free
                            single_TransportTime.append(row['time']-float(df.iloc[[i-1]]['time']))
                        else:
                            single_TransportTime.append(row['time'])
                        if(row["rxn"]=='rxn17'):
                            rxn17_count+=1
                        elif (row["rxn"]=='rxn21'):
                            rxn21_count+=1

                    ### If the rxn is after binding between tRNA-ribosome, include as a reaction_time
                    if((row['rxn']=='rxn22' or row['rxn']=='rxn19')):# and succincorp_count<num_rib): #Ignore rxn18 because add that on separately
                        single_RxnTime.append(row['time']-float(df.iloc[[i-1]]['time']))

                    ### If the reaction is the binding of the cognate tRNA to the matching cognate ribosome 
                    #[note, this does not include the time for succesful incorp, but does include cognate tRNA unbinding time from cognate ribosome]
                    #Thus, we capture reaction time and transport time between cognate tRNA - cognate ribosome initial binding events
                    if(row['rxn']=='rxn17' or row['rxn']=='rxn26'):
                        succincorp_count+=1
                        transport_time_i.append(sum(single_TransportTime))
                        reaction_time_i.append(sum(single_RxnTime))
                        rxn17_tot.append(rxn17_count)
                        rxn21_tot.append(rxn21_count)
                        single_TransportTime = list()
                        single_RxnTime = list()

                #print(reaction_time_i)
                #if(succincorp_count<num_rib and avg==True):
                 #   transport_time_i = transport_time_i[:-1] #drops the last transport time if there wasn't a reaction recorded afterwards (else last transport time would be too short)
                #Need to scale both transport and reaction time: since all reactions are set to happen a scaling factor shorter,
                #the time the cognate tRNA spends in transport also reduces by 10x since ribosomes are available to be bound quicker.
                #i.e., if ribosomes are all bound by othe non-cognates for 10x longer, the cognate tRNA spends 10x longer in transport time.
                #transport_time.append([np.sum(transport_time_i)])
                transport_time.append([np.sum(rib_reaction_time_i)*scaling+np.sum(rib_unbound_time_i) - np.sum(reaction_time_i)*scaling])
                reaction_time.append([np.sum(reaction_time_i)*scaling])
                #search_time.append([np.sum(transport_time_i)+np.sum(reaction_time_i)*scaling])
                search_time.append([np.sum(rib_reaction_time_i)*scaling+np.sum(rib_unbound_time_i)])
                success_incorp.append([np.sum(succincorp_count)])
            except:
                print("missing expt")
                print(expt_num)
    #print(plt.hist(reactantarray,bins=np.arange(50)))
    return transport_time, reaction_time, success_incorp,rxn17_tot,rxn21_tot, search_time

def transportRxnCalc(ptRNA, pCodon, ensmbl_latency_dict, bias=1):
    colors = ['darkblue','#D43F3A']
    phi_list = [0.13,0.22,0.30,0.36,0.39,0.42]
    markers = ['*','^']
    transport_phi = list()
    reaction_phi = list()
    search_phi = list()
    transport_std_phi =list()
    rxn_std_phi =list()
    search_std_phi =list()
    search_list = list()
    
    p_codon_count_hist_weighted_avg=cognateDistrib(ptRNA,pCodon)
    
    transport_vals_list = list()
    reaction_vals_list = list()
    search_vals_list = list()
    transport_var_list = list()
    rxn_var_list = list()
    search_var_list = list()
    
    #for range(1,7)
    for i in range(list(ensmbl_latency_dict.keys())[0],list(ensmbl_latency_dict.keys())[-1]+1):
        transport_vals = ensmbl_latency_dict[i].avg_transportT*1000/1608733*p_codon_count_hist_weighted_avg[i]#/(1-p_codon_count_hist_weighted_avg[0])
        rxn_vals = ensmbl_latency_dict[i].avg_rxnT*1000/1608733*p_codon_count_hist_weighted_avg[i]#/(1-p_codon_count_hist_weighted_avg[0])
        search_vals = ensmbl_latency_dict[i].avg_searchT*1000/1608733*p_codon_count_hist_weighted_avg[i]#/(1-p_codon_count_hist_weighted_avg[0])
        
        ##To scale variance correctly, need to multiply by square of the constant being multiplied to the mean
        transport_var = (ensmbl_latency_dict[i].std_transportT)**2*(1000/1608733*p_codon_count_hist_weighted_avg[i])**2#/(1-p_codon_count_hist_weighted_avg[0])
        rxn_var = (ensmbl_latency_dict[i].std_rxnT)**2*(1000/1608733*p_codon_count_hist_weighted_avg[i])**2#/(1-p_codon_count_hist_weighted_avg[0])
        search_var = (ensmbl_latency_dict[i].std_searchT)**2*(1000/1608733*p_codon_count_hist_weighted_avg[i])**2#/(1-p_codon_count_hist_weighted_avg[0])

        transport_vals_list.append(np.array(transport_vals))
        reaction_vals_list.append(np.array(rxn_vals))
        search_vals_list.append(np.array(search_vals))
        
        transport_var_list.append(np.array(transport_var))
        rxn_var_list.append(np.array(rxn_var))
        search_var_list.append(np.array(search_var))
        
        search_list.append(np.array(ensmbl_latency_dict[i].searchT)*1000/1608733)
        #print('Unweighted search time (', str(i), ' cognate)', np.array(search_vals/p_codon_count_hist_weighted_avg[i]))
    transport_phi.append(np.sum(transport_vals_list))
    reaction_phi.append(np.sum(reaction_vals_list))
    search_phi.append(np.sum(search_vals_list))
    
    transport_std_phi.append(np.sqrt(np.sum(transport_var_list)))
    rxn_std_phi.append(np.sqrt(np.sum(rxn_var_list)))
    search_std_phi.append(np.sqrt(np.sum(search_var_list)))

    #print("Transport time: ", transport_phi, " +/- ", transport_std_phi)
    #print("Reaction time: ", reaction_phi, " +/- ", rxn_std_phi)
    #print("Search time: ", search_phi, " +/- ", search_std_phi)
    
    return search_list,transport_phi, reaction_phi, search_phi, transport_std_phi,rxn_std_phi,search_std_phi


def computeNRLatency(NR_scaling = {'k1r':718,'k2f':1475,'k2r_nr':1120,'k3_nr':6,'k4':209}):
    t1r = 1000/NR_scaling['k1r']
    t2f = 1000/NR_scaling['k2f']
    t2r_nr = 1000/NR_scaling['k2r_nr']
    t3_nr = 1000/NR_scaling['k3_nr']
    t4 = 1000/NR_scaling['k4']

    t1r_exp=np.random.exponential(t1r,size=4000)
    t2f_exp=np.random.exponential(t2f,size=4000)
    t2r_nr_exp=np.random.exponential(t2r_nr,size=4000)
    t3_nr_exp=np.random.exponential(t3_nr,size=4000)
    t4_exp=np.random.exponential(t4,size=4000)

        #Near-cognate calculation
    dwelltime_nr_success = list()
    dwelltime_nr_fail = list()
    success_count = 0
    fail_count = 0

    t2f_exp=np.random.exponential(t2f,size=4000)

    for i in range(10000):
        dwell_t = 0
        state=1
        while state != 0 and state != 3:
            dwell_t1r = np.random.choice(t1r_exp)
            dwell_t2f = np.random.choice(t2f_exp)
            if state==1:
                if dwell_t1r<dwell_t2f: 
                    dwell_t+=np.random.choice(t1r_exp)
                    dwelltime_nr_fail.append(dwell_t)
                    state=0
                    fail_count += 1
                else:
                    dwell_t+=np.random.choice(t2f_exp)
                    state = 2
                    
            if state==2:
                dwell_t2r_nr = np.random.choice(t2r_nr_exp)
                dwell_t3_nr = np.random.choice(t3_nr_exp)
                if dwell_t2r_nr < dwell_t3_nr:
                    dwell_t+= np.random.choice(t2r_nr_exp)
                    state = 1
                else:
                    dwell_t += np.random.choice(t3_nr_exp)
                    state = 3
                    dwelltime_nr_success.append(dwell_t)
                    success_count+=1
                    
    return np.mean(dwelltime_nr_fail)

def cognateDistrib(ptRNA,pCodon, extra = False,extra2=False):

    ptRNA = np.divide(ptRNA,sum(ptRNA))
    pCodon= np.divide(pCodon, sum(pCodon))

    tRNA_tags = ["Ala1B", "Ala2", "Arg2", "Arg3", "Arg4", "Arg5", "Asn", "Asp1", "Cys", "Gln1", "Gln2", \
    "Glu2", "Gly2", "Gly3", "His", "Ile1", "Leu1", "Leu2", "Leu3", "Leu4", "Leu5", "Lys", \
    "Met_m", "Phe", "Pro1", "Pro2", "Pro3", "Sel_Cys", "Ser1", "Ser2", "Ser3", "Ser5", "Thr1", \
    "Thr2", "Thr3", "Thr4", "Trp", "Tyr1pTyr2", "Val1", "Val2ApB"]

    ptRNA_dict = dict(zip(tRNA_tags, ptRNA))
    
    codonLabels = pd.read_excel('codonValues.xlsx',header=None)[5]
    pcodon_dict = dict(zip(codonLabels,pCodon))

    #Note AUA does not have an assigned tRNA
    codon_dict={'GGG': ['Gly2'], 'GGA': ['Gly2'], 'GGU': ['Gly3'], 'GGC': ['Gly3'], \
    'GAG': ['Glu2'], 'GAA': ['Glu2'], 'GAU': ['Asp1'], 'GAC': ['Asp1'], \
    'GUG': ['Val1'], 'GUA': ['Val1'], 'GUU': ['Val1','Val2ApB'], \
    'GUC': ['Val2ApB'], 'GCG': ['Ala1B'], 'GCA': ['Ala1B'], 'GCU': ['Ala1B'], \
    'GCC': ['Ala2'], 'AGG': ['Arg5'], 'AGA': ['Arg4'], 'AGU': ['Ser3'], \
    'AGC': ['Ser3'], 'AAG': ['Lys'], 'AAA': ['Lys'], 'AAU': ['Asn'], \
    'AAC': ['Asn'], 'AUG': ['Met_m'], 'AUA': [], 'AUU': ['Ile1'], \
    'AUC': ['Ile1'], 'ACG': ['Thr2','Thr4'], 'ACA': ['Thr4'], \
    'ACU': ['Thr1','Thr4','Thr3'], 'ACC': ['Thr3','Thr1'], \
    'UGG': ['Trp'], 'UGA': ['Sel_Cys'], 'UGU': ['Cys'], 'UGC': ['Cys'], \
    'UAU': ['Tyr1pTyr2'], 'UAC': ['Tyr1pTyr2'], 'UUG': ['Leu5','Leu4'], \
    'UUA': ['Leu5'], 'UUU': ['Phe'], 'UUC': ['Phe'], 'UCG': ['Ser1','Ser2'], \
    'UCA': ['Ser1'], 'UCU': ['Ser5','Ser1'], 'UCC': ['Ser5'], 'CGG': ['Arg3'], \
    'CGA': ['Arg2'], 'CGU': ['Arg2'], 'CGC': ['Arg2'], 'CAG': ['Gln2'], \
    'CAA': ['Gln1'], 'CAU': ['His'], 'CAC': ['His'], 'CUG': ['Leu1','Leu3'], \
    'CUA': ['Leu3'], 'CUU': ['Leu2'], 'CUC': ['Leu2'], 'CCG': ['Pro1','Pro3'], \
    'CCA': ['Pro3'], 'CCU': ['Pro2','Pro3'], 'CCC': ['Pro2']}
    
    tRNA_dict = {'Ala1B': ['GCU', 'GCA', 'GCG'], 'Ala2': ['GCC'],'Arg2': ['CGU','CGC','CGA'], \
    'Arg3': ['CGG'], 'Arg4':['AGA'], 'Arg5': ['AGG'], 'Asn':['AAC','AAU'], 'Asp1': ['GAC','GAU'], \
    'Cys':['UGC','UGU'], 'Gln1':['CAA'], 'Gln2':['CAG'], 'Glu2': ['GAA', 'GAG'], 'Gly2':['GGA','GGG'], \
    'Gly3':['GGC','GGU'], 'His':['CAC','CAU'], 'Ile1': ['AUC','AUU'],'Leu1':['CUG'],'Leu2':['CUC','CUU'], \
    'Leu3': ['CUA','CUG'], 'Leu4': ['UUG'], 'Leu5': ['UUA', 'UUG'], 'Lys':['AAA', 'AAG'], 'Met_m':['AUG'], \
    'Phe': ['UUC', 'UUU'], 'Pro1': ['CCG'], 'Pro2': ['CCC','CCU'], 'Pro3': ['CCA', 'CCU', 'CCG'], \
    'Sec': ['UGA'], 'Ser1': ['UCA','UCU','UCG'], 'Ser2': ['UCG'], 'Ser3': ['AGC','AGU'], 'Ser5':['UCC', 'UCU'], \
    'Thr1':['ACC', 'ACU'], 'Thr2':['ACG'], 'Thr3':['ACC','ACU'], 'Thr4':['ACA','ACU','ACG'],'Trp':['UGG'], \
    'Tyr1pTyr2':['UAC','UAU'], 'Val1': ['GUA','GUG','GUU'], 'Val2ApB': ['GUC','GUU']}
    if extra:
        return ptRNA_dict, pcodon_dict, codon_dict, codonLabels, pCodon
    if extra2:
        return ptRNA_dict, pcodon_dict, codon_dict, codonLabels, pCodon, tRNA_dict


    cells = 1
    voxels = 10000
    time = 180
    N=42
    tRNA_distrib_arr = list()
    codon_count = {}
    codon_time = {}
    codon_time_avg = {}
    codon_time_weighted_avg={}
    codon_count_hist = {}
    codon_count_hist_weighted_avg = np.zeros(N)
    p_codon_tRNA = {}

    #np.random.seed(0) #made this change

    for key in codon_dict:
        codon_count[key] = []
        codon_time[key] = []
        codon_time_avg[key] = []
        codon_time_weighted_avg[key]=[]
        codon_count_hist[key]=[]
        p_codon_tRNA[key] = []

    # Construct dictionary that assigns probability of all tRNA specific to a certain codon
    # to that codon (p_codon_tRNA)
    for codon in codon_dict:
        p_codon_tRNA_i = 0
        for tRNA in codon_dict[codon]:
            p_codon_tRNA_i += ptRNA_dict[tRNA]
        p_codon_tRNA[codon].append(p_codon_tRNA_i)

    for cell in range(cells):
        # Generate distribution for cognate tRNA count for each codon
        for i in range(voxels):

            #Choose 1 random codon for tranlsation voxel (weighted by codon probabilities), and identify cognate and non cognate ternary complexes
            codon_vox = np.random.choice(codonLabels, 1)
            cognatetRNA = codon_dict[codon_vox[0]]
            noncognatetRNA = [tRNA for tRNA in tRNA_tags if tRNA not in codon_dict[codon_vox[0]]]

            ##Create biased tRNA distribution, if bias exists.
            #biased_ptRNA = ptRNA.copy()
            #for _,tRNA_i in enumerate(cognatetRNA):
            #    biased_ptRNA[tRNA_tags.index(tRNA_i)]=biased_ptRNA[tRNA_tags.index(tRNA_i)]*bias
            #for _,tRNA_i in enumerate(noncognatetRNA):
            #    biased_ptRNA[tRNA_tags.index(tRNA_i)]=biased_ptRNA[tRNA_tags.index(tRNA_i)]/bias
            #biased_ptRNA = biased_ptRNA/sum(biased_ptRNA)

            #Construct translation voxel (weighted by specific tRNA abundances and bias)
            tRNA_vox = list(np.random.choice(tRNA_tags,N,p=ptRNA))


            #Count how many cognate tRNA appeared in the translation unit (for given codon) and record in codon_count
            codon_count_i = 0
            for tRNA in cognatetRNA:
                codon_count_i += tRNA_vox.count(tRNA)
            if tRNA_vox.count(tRNA)==0:
                codon_count[codon_vox[0]].append(1)
            else:
                codon_count[codon_vox[0]].append(codon_count_i)

        for codon in codon_count:
            #Generate histogram of cognate tRNA counts for each codon
            codon_count_hist[codon] = np.histogram(codon_count[codon], bins=np.arange(0,N+1))[0]/sum(np.histogram(codon_count[codon], bins=np.arange(0,N+1))[0])

            #Weight histogram by codon probabilities to generate weighted average histogram for all codon
            codon_count_hist_weighted_avg += codon_count_hist[codon]*pcodon_dict[codon]
        p_codon_count_hist_weighted_avg = codon_count_hist_weighted_avg
        #print(p_codon_count_hist_weighted_avg)


    return p_codon_count_hist_weighted_avg


def nearcognateDistrib(ptRNA,pCodon):

    ptRNA = np.divide(ptRNA,sum(ptRNA))
    pCodon= np.divide(pCodon, sum(pCodon))

    tRNA_tags = ["Ala1B", "Ala2", "Arg2", "Arg3", "Arg4", "Arg5", "Asn", "Asp1", "Cys", "Gln1", "Gln2", \
    "Glu2", "Gly2", "Gly3", "His", "Ile1", "Leu1", "Leu2", "Leu3", "Leu4", "Leu5", "Lys", \
    "Met_m", "Phe", "Pro1", "Pro2", "Pro3", "Sel_Cys", "Ser1", "Ser2", "Ser3", "Ser5", "Thr1", \
    "Thr2", "Thr3", "Thr4", "Trp", "Tyr1pTyr2", "Val1", "Val2ApB"]

    ptRNA_dict = dict(zip(tRNA_tags, ptRNA))
    
    codonLabels = pd.read_excel('codonValues.xlsx',header=None)[5]
    pcodon_dict = dict(zip(codonLabels,pCodon))

    #Note AUA does not have an assigned tRNA
    codon_dict={'GGG': ['Gly2'], 'GGA': ['Gly2'], 'GGU': ['Gly3'], 'GGC': ['Gly3'], \
    'GAG': ['Glu2'], 'GAA': ['Glu2'], 'GAU': ['Asp1'], 'GAC': ['Asp1'], \
    'GUG': ['Val1'], 'GUA': ['Val1'], 'GUU': ['Val1','Val2ApB'], \
    'GUC': ['Val2ApB'], 'GCG': ['Ala1B'], 'GCA': ['Ala1B'], 'GCU': ['Ala1B'], \
    'GCC': ['Ala2'], 'AGG': ['Arg5'], 'AGA': ['Arg4'], 'AGU': ['Ser3'], \
    'AGC': ['Ser3'], 'AAG': ['Lys'], 'AAA': ['Lys'], 'AAU': ['Asn'], \
    'AAC': ['Asn'], 'AUG': ['Met_m'], 'AUA': [], 'AUU': ['Ile1'], \
    'AUC': ['Ile1'], 'ACG': ['Thr2','Thr4'], 'ACA': ['Thr4'], \
    'ACU': ['Thr1','Thr4','Thr3'], 'ACC': ['Thr3','Thr1'], \
    'UGG': ['Trp'], 'UGA': ['Sel_Cys'], 'UGU': ['Cys'], 'UGC': ['Cys'], \
    'UAU': ['Tyr1pTyr2'], 'UAC': ['Tyr1pTyr2'], 'UUG': ['Leu5','Leu4'], \
    'UUA': ['Leu5'], 'UUU': ['Phe'], 'UUC': ['Phe'], 'UCG': ['Ser1','Ser2'], \
    'UCA': ['Ser1'], 'UCU': ['Ser5','Ser1'], 'UCC': ['Ser5'], 'CGG': ['Arg3'], \
    'CGA': ['Arg2'], 'CGU': ['Arg2'], 'CGC': ['Arg2'], 'CAG': ['Gln2'], \
    'CAA': ['Gln1'], 'CAU': ['His'], 'CAC': ['His'], 'CUG': ['Leu1','Leu3'], \
    'CUA': ['Leu3'], 'CUU': ['Leu2'], 'CUC': ['Leu2'], 'CCG': ['Pro1','Pro3'], \
    'CCA': ['Pro3'], 'CCU': ['Pro2','Pro3'], 'CCC': ['Pro2']}

    cells = 1
    voxels = 100000
    time = 180
    tRNA_distrib_arr = list()
    codon_count = {}
    codon_time = {}
    codon_time_avg = {}
    codon_time_weighted_avg={}
    codon_count_hist = {}
    codon_count_hist_weighted_avg = np.zeros(42)
    p_codon_tRNA = {}

    #np.random.seed(0)

    for key in codon_dict:
        codon_count[key] = []
        codon_time[key] = []
        codon_time_avg[key] = []
        codon_time_weighted_avg[key]=[]
        codon_count_hist[key]=[]
        p_codon_tRNA[key] = []

    # Construct dictionary that assigns probability of all tRNA specific to a certain codon
    # to that codon (p_codon_tRNA)
    for codon in codon_dict:
        p_codon_tRNA_i = 0
        for tRNA in codon_dict[codon]:
            p_codon_tRNA_i += ptRNA_dict[tRNA]
        p_codon_tRNA[codon].append(p_codon_tRNA_i)

    for cell in range(cells):
        # Generate distribution for cognate tRNA count for each codon
        for i in range(voxels):

            #Choose 1 random codon for translation voxel (weighted by codon probabilities), and identify cognate and non cognate ternary complexes
            codon_vox = np.random.choice(codonLabels, 4)
            
            cognatetRNA = codon_dict[codon_vox[0]]
            nearcognatetRNA = list()
            for codon in neighbors(codon_vox[0]):
                if(codon in codon_dict):
                    for tRNA in codon_dict[codon]:
                        nearcognatetRNA.append(tRNA)
            #Construct translation voxel (weighted by specific tRNA abundances and bias)
            tRNA_vox = list(np.random.choice(tRNA_tags,42,p=ptRNA))


            #Count how many near-cognate tRNA appeared in the translation unit (for given codon) and record in codon_count
            codon_count_i = 0
            for tRNA in nearcognatetRNA:
                codon_count_i += tRNA_vox.count(tRNA)
            codon_count[codon_vox[0]].append(codon_count_i)

        for codon in codon_count:
            #Generate histogram of cognate tRNA counts for each codon
            codon_count_hist[codon] = np.histogram(codon_count[codon], bins=np.arange(0,43))[0]/sum(np.histogram(codon_count[codon], bins=np.arange(0,43))[0])

            #Weight histogram by codon probabilities to generate weighted average histogram for all codon
            codon_count_hist_weighted_avg += codon_count_hist[codon]*pcodon_dict[codon]
        p_codon_count_hist_weighted_avg = codon_count_hist_weighted_avg
        #print(p_codon_count_hist_weighted_avg)
    return p_codon_count_hist_weighted_avg


class CellLatencies:
    def __init__ (self,TransportRxnTimesarr,bootstrap=True):
        self.transportT = [i for trans_i in TransportRxnTimesarr[0] for i in trans_i]
        self.rxnT = [i for reac_i in TransportRxnTimesarr[1] for i in reac_i]
        self.incorrRxn = TransportRxnTimesarr[4]
        self.searchT = [i for search_i in TransportRxnTimesarr[5] for i in search_i]
       
        self.avg_transportT = np.average(self.transportT)     
        self.avg_rxnT = np.average(self.rxnT)   
        self.avg_searchT = np.average(self.searchT)    

        self.std_transportT = np.std(self.transportT)/np.sqrt(len(self.transportT)-1)
        self.std_searchT = np.std(self.searchT)/np.sqrt(len(self.searchT)-1)
        self.std_rxnT = np.std(self.rxnT)/np.sqrt(len(self.rxnT)-1)




#Input: tRNA and codon probability distributions as well as a dictionary
#of ensemble latency dictionaries
#Returns: Elongation latencies and standard errors for each input growth rate
def computeElongationLatency(ptRNA,pCodon,ensmbl_latency_dict):
    rxndiff=dict()
    transportRxnResults = transportRxnCalc(ptRNA,pCodon,ensmbl_latency_dict)
    rxndiff['30'] = transportRxnResults[1:]
    
    #The added scalar values are the average reaction latencies following succesful reaction
    return([rxndiff[d][2][0]+(1000/1475+1000/1529+1000/209+1000/200+1000/32) for d in rxndiff],[rxndiff[d][5][0] for d in rxndiff])


def computeElongationLatency_multithread(input):
    ptRNA = input[0]
    pCodon = input[1]
    ensmbl_latency_dict = input[2]
    transportRxnResults = transportRxnCalc(ptRNA,pCodon,ensmbl_latency_dict)
    
    rxndiff=dict()
    rxndiff['30'] = transportRxnResults[1:]
    return([rxndiff[d][2][0]+(1000/1475+1000/1529+1000/209+1000/200+1000/32) for d in rxndiff],[rxndiff[d][5][0] for d in rxndiff])

def run_ga_tRNA(tRNA_list,codon_list,elong_list,ensmbl_latency_dict,minRange,maxRange,objective='fast'):
    num_cores = 16

    #### Compute fitness
    if objective == 'fast':
        fitness = (1/np.array(elong_list))/sum((1/np.array(elong_list)))
    elif objective == 'slow':
        fitness = (np.array(elong_list))/sum((np.array(elong_list)))

    #### Number of candidates n removing as well as n mating to create n offspring
    n = 10
    #### Identify the least fit candidates from the population
    cull_indices = np.argpartition(fitness, n)[:n]

    #### Choose parents based on weighting fitness
    parent_indices = np.argpartition(fitness, n)[-n:]
    tRNA_list=np.array(tRNA_list)
    parents = tRNA_list[parent_indices]

    #### Mate k random pairs of 2 without replacement and renormalize
    k=5
    couples = np.random.choice(np.arange(len(parents)), size = (k,2),replace=False)
    recombination_rate = 0.1
    mutation_rate  = 0.05

    recombined_children = list()
    recombined_children_elongt = list()
    for couple_index in couples:
        couple = parents[couple_index]
        recombination_num = int(len(couple[0])*recombination_rate)
        recombination_locs = np.random.choice(len(couple[0]),recombination_num)
        recombination_values_0 = couple[0][recombination_locs] 
        couple[0][recombination_locs] = couple[1][recombination_locs]
        couple[1][recombination_locs] = recombination_values_0
        
        #### Mutate children
        mutation_num = int(len(couple[0])*mutation_rate)
        recombination_locs = np.random.choice(len(couple[0]),mutation_num)
        couple[0][recombination_locs] = np.random.uniform(minRange,maxRange,mutation_num)
        couple[1][recombination_locs] = np.random.uniform(minRange,maxRange,mutation_num)
        
        ### Re-normalize each recombined children
        child_0 = couple[0]/np.sum(couple[0])
        child_1 = couple[1]/np.sum(couple[1])
        
        ### Add children to list
        recombined_children.append(list(child_0))
        recombined_children.append(list(child_1))
        
    #### Compute elong_t of the recombined children, multithreaded
    inputs=[[recombined_children[i],codon_list,ensmbl_latency_dict] for i in np.arange(len(couples)*2)]
    parallel_elong = Parallel(n_jobs=num_cores,backend='loky')(delayed(computeElongationLatency_multithread)(i) for i in inputs)
    for _,items in enumerate(parallel_elong):
        recombined_children_elongt.append(items[0][0])
#    del(a)
#    gc.collect()
    
    #### Have recombined children and their elong_t replaced culled candidates
    tRNA_list[cull_indices] = recombined_children
    elong_list[cull_indices] = recombined_children_elongt
    return fitness, tRNA_list, elong_list

def run_ga_CodonSweep(tRNA_list,codon_list,elong_list,ensmbl_latency_dict,minRange,maxRange,objective='fast', 
    syn_codon_list = [[0,1,2,3],[4,5],[6,7],[8,9,10,11],[12,13,14,15],[16,17,46,47,48,49],[18,19,42,43,44,45],\
    [20,21],[22,23],[24],[25,26,27],[28,29,30,31],[32],[33],[34,35],[36,37],[38,39,54,55,56,57],[40,41],\
    [50,51],[52,53],[58,59,60,61]],\
    pCodon = [2.36, 1.26, 45.55, 34.17, 16.97, 57.86, 19.27, 33.74, 14.98, 22.31, 43.18, 7.67, \
    24.11, 24.87, 39.49, 11.81, 0.03, 0.63, 2.19, 9.31, 17.22, 55.01, 5.61, 29.21, 21.67, 0.52, 15.79, \
    43.86, 4.17, 2.61, 20.64, 26.7, 7.03, 0.19, 2.76, 3.81, 6.72, 16.52, 4.27, 2.73, 7.92, 23.25, 2.51,\
    1.98, 16.33, 11.68, 0.62, 0.67, 43.82, 20.59, 27.28, 7.01, 6.78, 14.21, 60.75, 0.82, 3.86, 4.09, \
    28.82, 5.18, 4.38, 1.09]):
    num_cores = 16

    #### Compute fitness
    if objective == 'fast':
        fitness = (1/np.array(elong_list))/sum((1/np.array(elong_list)))
    elif objective == 'slow':
        fitness = (np.array(elong_list))/sum((np.array(elong_list)))

    #### Number of candidates n removing as well as n mating to create n offspring
    n = 10

    #### Identify the least fit candidates from the population
    cull_indices = np.argpartition(fitness, n)[:n]

    #### Choose parents based on weighting fitness
    import random
    #parent_indices = np.array(random.choices(np.arange(len(p_tRNA_list)), fitness, k=10))
    parent_indices = np.argpartition(fitness, n)[-n:]
    codon_list=np.array(codon_list)
    parents = codon_list[parent_indices]

    #### Mate k random pairs of 2 without replacement and renormalize
    k=5
    couples = np.random.choice(np.arange(len(parents)), size = (k,2),replace=False)
    recombination_rate = 0.1
    mutation_rate  = 0.05

    recombined_children = list()
    recombined_children_elongt = list()
    
    
    for couple_index in couples:
        couple = parents[couple_index]
        recombination_num = int(len(couple[0])*recombination_rate)
        recombination_locs = np.random.choice(len(couple[0]),recombination_num)
        recombination_values_0 = couple[0][recombination_locs] 
        couple[0][recombination_locs] = couple[1][recombination_locs]
        couple[1][recombination_locs] = recombination_values_0
        
        #### Mutate children
        mutation_num = int(len(couple[0])*mutation_rate)
        recombination_locs = np.random.choice(len(couple[0]),mutation_num)
        couple[0][recombination_locs] = np.random.uniform(minRange,maxRange,mutation_num)
        couple[1][recombination_locs] = np.random.uniform(minRange,maxRange,mutation_num)
        
        ### Re-normalize each recombined children
        ##First re-normalize to make sure synonymous codons have, together, the same total frequency (keep the genome the same)
        pCodon = np.array(pCodon)/np.sum(np.array(pCodon))

        syn_codon_freqs = list()
        for syn_codons in syn_codon_list:
            syn_codon_freqs.append(sum(pCodon[syn_codons]))
            couple[0][syn_codons] = sum(pCodon[syn_codons])/sum(couple[0][syn_codons])*couple[0][syn_codons]
            couple[1][syn_codons] = sum(pCodon[syn_codons])/sum(couple[1][syn_codons])*couple[1][syn_codons]
        child_0 = couple[0]/np.sum(couple[0])
        child_1 = couple[1]/np.sum(couple[1])
        
        ### Add children to list
        recombined_children.append(list(child_0))
        recombined_children.append(list(child_1))
        
    #### Compute elong_t of the recombined children, multithreaded
    inputs=[[tRNA_list,recombined_children[i],ensmbl_latency_dict] for i in np.arange(len(couples)*2)]
    parallel_elong = Parallel(n_jobs=num_cores,backend='loky')(delayed(computeElongationLatency_multithread)(i) for i in inputs)
    for _,items in enumerate(parallel_elong):
        recombined_children_elongt.append(items[0][0])
    
    #### Have recombined children and their elong_t replaced culled candidates
    codon_list[cull_indices] = recombined_children
    elong_list[cull_indices] = recombined_children_elongt
    return fitness, codon_list, elong_list

def calc_R2(x,y,y_hat):
    SS_err = np.sum((y-y_hat)**2) # Sum of squared errors
    SS_tot = np.sum((y-np.average(y))**2) #Sum of squares total (proportional to variance; n times larger than variance)
    return 1-SS_err/SS_tot