#libraries
import random
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

#configuration dictionaries
map_split_vars = {
    'nhits': "TSplitEventChunk/TSplitEventChunk.data_.THits",
    'nveto': "TSplitEventChunk/TSplitEventChunk.data_.VHits",
    #'tposhits': "TSplitEventChunk/TSplitEventChunk.data_.TposHits",
    #'vposhits': "TSplitEventChunk/TSplitEventChunk.data_.VposHits",
    #'tpmt': "TSplitEventChunk/TSplitEventChunk.data_.TPMT",
    #'vpmt': "TSplitEventChunk/TSplitEventChunk.data_.VPMT",
    #'tq': "TSplitEventChunk/TSplitEventChunk.data_.TQ",
    #'vq': "TSplitEventChunk/TSplitEventChunk.data_.VQ",
    #'qtot': "TSplitEventChunk/TSplitEventChunk.data_.Qtot",
    'tqpos': "TSplitEventChunk/TSplitEventChunk.data_.TQpos", #total high change tank charge
    #'vqpos': "TSplitEventChunk/TSplitEventChunk.data_.VQpos",
    #'tt': "TSplitEventChunk/TSplitEventChunk.data_.TT",
    #'vt': "TSplitEventChunk/TSplitEventChunk.data_.VT",
    #'tmin': "TSplitEventChunk/TSplitEventChunk.data_.Tmin",
    #tmax': "TSplitEventChunk/TSplitEventChunk.data_.Tmax",
    #'avgttim': "TSplitEventChunk/TSplitEventChunk.data_.AvgTTim",
    #'peakttim': "TSplitEventChunk/TSplitEventChunk.data_.PeakTTim",
    'rmsttim': "TSplitEventChunk/TSplitEventChunk.data_.rmsTTim", # rms time for high change tank hits
    #'avgvtim': "TSplitEventChunk/TSplitEventChunk.data_.AvgVTim",
    #'peakvtim': "TSplitEventChunk/TSplitEventChunk.data_.PeakVTim",
    # 'rmsvtim': "TSplitEventChunk/TSplitEventChunk.data_.rmsVTim"
    }

map_onetrack_vars = {
    'E': "TOneTrackChunk/TOneTrackChunk.data_.E",
    'F': "TOneTrackChunk/TOneTrackChunk.data_.F", 
    'T': "TOneTrackChunk/TOneTrackChunk.data_.T",
    'chunk_id': "TOneTrackChunk/TOneTrackChunk.chunk_id_",
    'fUniqueID': "TOneTrackChunk/TOneTrackChunk.data_.fUniqueID",
    'fBits': "TOneTrackChunk/TOneTrackChunk.data_.fBits",
    'iterations': "TOneTrackChunk/TOneTrackChunk.data_.iterations",
    'trackType': "TOneTrackChunk/TOneTrackChunk.data_.trackType",
    'X': "TOneTrackChunk/TOneTrackChunk.data_.X",
    'Y': "TOneTrackChunk/TOneTrackChunk.data_.Y",
    'Z': "TOneTrackChunk/TOneTrackChunk.data_.Z",
    'UX': "TOneTrackChunk/TOneTrackChunk.data_.UX",
    'UY': "TOneTrackChunk/TOneTrackChunk.data_.UY",
    'UZ': "TOneTrackChunk/TOneTrackChunk.data_.UZ",
    'distToMeanCer': "TOneTrackChunk/TOneTrackChunk.data_.distToMeanCer",
    'fluxScale': "TOneTrackChunk/TOneTrackChunk.data_.fluxScale",
    'relativeSci': "TOneTrackChunk/TOneTrackChunk.data_.relativeSci"}

#data processing functions

##########################################################################################
#INPUT: IPFS DATA DUMP (EVNT#,NFSP,IFSP)
#OUTPUT: LIST WHERE EVERY ENTRY IS 8-VECTOR WITH COUNTS OF PARTICLES 1-3 AND 5-9 FOR THAT EVENT
#REALIZE 1=GAMMA, 2,3,5,6 ARE LEPTONS, 7,8,9 ARE PI'S

def turn_IPFS_dump_to_list(IPFS_dump_file):
    
    #this is OUTPUT. It'll be 8-subentries per entry
    counts_all_events = []

    #we map the relevant particle ID to the entry it'll have in the counts_one_event
    particle_map = {"1":0, "2":1, "3":2, "5":3, "6":4, "7":5, "8":6, "9":7}
    
    #pop open the IPFS dump
    with open(IPFS_dump_file, 'r') as IPFS_dump:
        for line in IPFS_dump:

            #empty 8-vector to store the counts for this evnt in
            counts_one_event = np.zeros(8, dtype=int)

            #take the line, (which will be evnt #, NFSP, all IPFS written out)
            #strip away the spaces at end of lines
            #split takes the "2 4 3" string into [2, 4, 3] list
            #i also cut away the first 2 entries which are evntNumber and NFSP
            particle_list_for_this_event = ((line.strip()).split())[2:]
            
            #add on the relevant particles into the correct spot of the counts_one_event
            for ID in particle_list_for_this_event:
                if ID in particle_map:
                    counts_one_event[particle_map[ID]] += 1
            
            #then add the 8-vector for this event to hte whole list for all events
            counts_all_events.append(counts_one_event)
    
    return counts_all_events

############################################################################################

counts_relevant_particles_all_events = turn_IPFS_dump_to_list(r'C:\Users\Mathias\Desktop\NEUTRINO CODING\event # followed by NFSP followed by IPFS.txt')

#we input the ROOT tree (already opened w uproot) and a dictionary for branches we want to run LDA on, i.e. either map_onetrack_vars or map split vars
#we output dictionaries of vars with 1st subentry. we also output for 2nd subentry but some evnts lack as well as list of what variables open
def extract_branches(tree, dict):
  
    full_branch_dictionary = {}
    first_subentry_branch_dictionary = {}
    second_subentry_branch_dictionary = {}
    list_of_what_opens = []
    
    for branch_name, branch_path in dict.items(): #dict.items() breaks dict into [(key1,value1),...]
        try:
            full_branch_dictionary[branch_name] = tree[branch_path].array()
            first_subentry_branch_dictionary[branch_name] = ak.firsts(full_branch_dictionary[branch_name])  #about 10% of events dont even have first entry for onetrack branches but then .firsts method pads with None and we filter latr
              # Only try muon for these. these are the vars where we care about the muon entry
                #the line below pads up all [] or [#] to [None,None] and [#,None]
                #then we take the 1st entry i.e. the muon info
                #so from line below we get some list a la [None,#,#,None,#,...]
            second_subentry_branch_dictionary[branch_name] = ak.pad_none(full_branch_dictionary[branch_name], 2)[:, 1]
            list_of_what_opens.append(branch_name)
            print(f"OK {branch_name}")
        except Exception as e:
            print(f"NO {branch_name}")
    
    return first_subentry_branch_dictionary, second_subentry_branch_dictionary, list_of_what_opens
#notice that the second_subentry will contain a bunch of Nones. thats fine because we mask out evnt without Onetrack muon entries (as well as el entries) later

#inputs the OneTrackFirsts, OneTrackSconds or SplitEvntFirsts dictionaries (which are padded)
#outputs dictionary but with mask s.t. we only keep the entries that pass the mask
#example. OneTrackFirsts will be {"E":[entry1, entry2, ...],"F":[...],}
#so mask1 will give True for all the entries that both have 0th and 1st entry and discard the rest
#and we push all the 3 dictionaries thru the same masks so same var's survive
def apply_mask(data_dict, mask):
    return {k: v[mask] for k, v in data_dict.items()} 

# INPUT - cleaned-up variable maps. OUTPUT: data matrix to push into LDA
# basically LDA needs needs the entries for var1 in col 1, var2 in col2 etc. 
# thus this matrix inputs the maps of variables
# adds the data for each branch in this list called columns
#also makes a list of all the names. useful b/c now were pushing a bunch of dictionaries together so its good to keep track what data is what variable
#we run np.column_stack which basically transposes the matrix so each row now correpsonds to one event
def build_data_matrix(map_split_vars, onetrack_vars_we_run_LDA_on):
    columns = []
    names = []
    
    for data, name in onetrack_vars_we_run_LDA_on:
        columns.append(ak.to_numpy(data))
        names.append(name)
    
    for var_name in map_split_vars:
        columns.append(ak.to_numpy(map_split_vars[var_name]))
        names.append(var_name)
    
    return np.column_stack(columns), names

#coordinate calculation functions

#function to derive cylindrical coordinates from onetrack branch. either electron or muon hypothesis
def calc_coords(ot_dict):
    r_cyl = np.sqrt(ot_dict['X']**2 + ot_dict['Y']**2)
    phi_cyl = np.arctan2(ot_dict['Y'], ot_dict['X']) #always use arctan2 instead of arctan cuz arctan2 gives -180,180 not just -90,90
    R_spher = np.sqrt(ot_dict['X']**2 + ot_dict['Y']**2 + ot_dict['Z']**2)
    theta_spher = np.arccos(ot_dict['Z'] / R_spher)
    
    ur_cyl = np.sqrt(ot_dict['UX']**2 + ot_dict['UY']**2)
    uphi_cyl = np.arctan2(ot_dict['UY'], ot_dict['UX'])
    UR_spher = np.sqrt(ot_dict['UX']**2 + ot_dict['UY']**2 + ot_dict['UZ']**2)
    u_theta_spher = np.arccos(ot_dict['UZ'] / UR_spher)
    
    return {'r_cyl': r_cyl, 'phi_cyl': phi_cyl, 'R_spher': R_spher, 'theta_spher': theta_spher,
            'ur_cyl': ur_cyl, 'uphi_cyl': uphi_cyl, 'UR_spher': UR_spher, 'u_theta_spher': u_theta_spher}

def norm_calculator(x,y,z):
    R = np.sqrt(x**2 + y**2 + z**2)
    return R

def calc_mu_endpt(ot_X,ot_Y,ot_Z,ot_UX,ot_UY,ot_UZ,dist_cer):
    
    ot_X_second = ak.fill_none(ak.pad_none(ot_X, 2), -99999)[:, 1]
    ot_Y_second = ak.fill_none(ak.pad_none(ot_Y, 2), -99999)[:, 1]
    ot_Z_second = ak.fill_none(ak.pad_none(ot_Z, 2), -99999)[:, 1]

    ot_UX_second = ak.fill_none(ak.pad_none(ot_UX, 2), -99999)[:, 1]
    ot_UY_second = ak.fill_none(ak.pad_none(ot_UY, 2), -99999)[:, 1]
    ot_UZ_second = ak.fill_none(ak.pad_none(ot_UZ, 2), -99999)[:, 1]
    dist_second =  ak.fill_none(ak.pad_none(dist_cer, 2), -99999)[:, 1]

    endpoint_R_mu = np.sqrt((ot_X_second + 2*dist_second*ot_UX_second)**2 + 
                       (ot_Y_second + 2*dist_second*ot_UY_second)**2 + 
                       (ot_Z_second + 2*dist_second*ot_UZ_second)**2)
    return endpoint_R_mu

# masking functions

def empty_mask(array):
    return ak.num(array) > 0

def empty_mask_second(list):
    padded_list = ak.pad_none(list, 2)[:, 1]
    mask = ~ak.is_none(padded_list)
    return mask

# plotting functionss

#function to create histogram plots
def histo_plotter(data_list, bins,title):
    plt.figure(figsize=(4,4))
    plt.hist(data_list, bins=bins)
    plt.title(title)
    plt.show()
    return