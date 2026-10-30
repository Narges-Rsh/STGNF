import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
#from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
import csv
#from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as auc_func
#from sklearn.metrics import roc_curve
#from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def score_dataset(score, metadata, args=None, save_results=True, directory="results/csv_files/"):
    gt_arr, scores_arr = get_dataset_scores(score, metadata, args=args, save_results=save_results, directory=directory)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc_roc, auc_pr, eer, eer_threshold, best_f1, best_f1_threshold = score_auc(scores_np, gt_np)   
    # AUC-PR Calculation 
    #auc_pr=score_auc_pr(scores_np, gt_np)     
    #precision, recall, _ = precision_recall_curve(gt_np, scores_np)
    #auc_pr = auc_func(recall, precision)
    # Calculate EER and EER threshold
    #eer, eer_threshold = score_eer(scores_np, gt_np)

    #return auc, auc_pr, scores_np   
    return auc_roc, scores_np, auc_pr, eer, eer_threshold,best_f1, best_f1_threshold 
    
    
    





def get_dataset_scores(scores, metadata, args=None, save_results=True, directory="results/csv_files/"):  
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    
    if save_results:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if args.dataset == 'UBnormal':
        pose_segs_root = 'data/UBnormal/pose/test'
        clip_list = os.listdir(pose_segs_root)
        clip_list = sorted(
            fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
        per_frame_scores_root = 'data/UBnormal/gt/'
        
    elif args.dataset == 'ABC_poselift':
        per_frame_scores_root = 'data/ABC_poselift/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
          
    elif args.dataset == 'ABC':
        per_frame_scores_root = 'data/ABC/gt_per_camera/C6'      
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))  
    
    elif args.dataset=='ShanghaiTech':
        per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
        
    elif args.dataset=='CHAD':
        per_frame_scores_root = 'data/CHAD/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
        
        
    elif args.dataset=='NWPUC':
        per_frame_scores_root = 'data/NWPUC/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
        
    elif args.dataset=="HuVAD":
        per_frame_scores_root = 'data/HuVAD/Normal_Training/GT/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
        
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")   
        
    clips=[]   
    print("Scoring {} clips".format(len(clip_list)))
    #First:
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)


              #my code for saving the results:
        if save_results:

               
            clip_score_save = np.array(clip_score)
            if np.all(clip_score_save == np.inf):  # Only check for np.inf (positive infinity)
                clip_score_save.fill(1)  # Replace all np.inf with 1

            else:
                clip_score_save[clip_score_save == np.inf] = np.nanmax(clip_score_save[clip_score_save != np.inf])  # Replace with max
                clip_score_save[clip_score_save == -np.inf] = np.nanmin(clip_score_save[clip_score_save != -np.inf])  # Replace with min


            with open(directory+clip.split(".")[0]+".csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(zip(clip_gt, clip_score_save))
            

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    
  
    
    
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    return dataset_gt_arr, dataset_scores_arr


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    
    #Standard AUC-ROC
    auc_roc= roc_auc_score(gt, scores_np)   #Area Under the ROC Curve
    #precision, recall, _ = precision_recall_curve(gt, scores_np)
    #auc_pr = auc_func(recall, precision)
    # auc_pr = average_precision_score(gt, scores_np)
    # fpr, tpr, thresholds = roc_curve(gt, scores_np)  # Get FPR, TPR, thresholds from ROC curve
    # fnr = 1 - tpr  # False Negative Rate = 1 - True Positive Rate

    # # Calculate EER where FPR equals FNR
    # eer_index = np.nanargmin(np.abs(fnr - fpr))  # Find index where FPR = FNR
    # eer_threshold = thresholds[eer_index]  # Threshold at which EER occurs
    # eer = fpr[eer_index]  # Equal Error Rate (where FPR = FNR)
    # return auc, auc_pr, eer, eer_threshold
    
    #
    
    precision, recall, thresholds = precision_recall_curve(gt, scores_np)
    auc_precision_recall = auc(recall, precision)
    
    
    #calculate best F1-score and its threshold
    ## F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-5)
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    
    # The thresholds array from precision_recall_curve is 1 element shorter than precision/recall
    # so we handle the index carefully
    if best_f1_index < len(thresholds):
        best_f1_threshold = thresholds[best_f1_index]
    else:
        best_f1_threshold = thresholds[-1]
    # -------------------------------------------------------v
    
    
    fpr, tpr, threshold = roc_curve(gt, scores_np, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return auc_roc, auc_precision_recall, eer, eer_threshold, best_f1, best_f1_threshold

# AUC-PR Calculation
# def score_auc_pr(scores_np, gt):
#     precision, recall, _ = precision_recall_curve(gt, scores_np)
#     auc_pr = auc_func(recall, precision)  # Area Under the Precision-Recall Curve
#     return auc_pr

# EER Calculation (Equal Error Rate)
# def score_eer(scores_np, gt):
#     fpr, tpr, thresholds = roc_curve(gt, scores_np)  # Get FPR, TPR, thresholds from ROC curve
#     fnr = 1 - tpr  # False Negative Rate = 1 - True Positive Rate

#     # Calculate EER where FPR equals FNR
#     eer_index = np.nanargmin(np.abs(fnr - fpr))  # Find index where FPR = FNR
#     eer_threshold = thresholds[eer_index]  # Threshold at which EER occurs
#     eer = fpr[eer_index]  # Equal Error Rate (where FPR = FNR)
    
#     return eer, eer_threshold
   



def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):
    if args.dataset == 'UBnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        clip_id = type + "_" + clip_id
    else:
        
          #modified
        #scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
        parts = clip.split('.')[0].split('_')
        # if len(parts) >= 3:
        #     scene_id = int(parts[0])
        #     clip_id = int(parts[1])
        #     clip_idx = int(parts[2])
        # else:
        #     raise ValueError(f"Unexpected clip filename format: {clip}")
        if len(parts) == 3:
            scene_id = int(parts[0])
            clip_id = int(parts[1])
            clip_idx = int(parts[2])
        elif len(parts) == 2:
            scene_id = int(parts[0])
            clip_id = int(parts[1])
            clip_idx = 0  # default value
        else:
            raise ValueError(f"Unexpected clip filename format: {clip}")
        
        
        if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
            return None, None
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]  
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)
    if args.dataset != "UBnormal":
        clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amin(clip_ppl_score_arr, axis=0)

    return clip_gt, clip_score
