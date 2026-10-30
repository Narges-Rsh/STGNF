# eval_only.py
import random
import numpy as np
import torch
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from args import init_parser, init_sub_args
from args import create_exp_dirs
from dataset import get_dataset_and_loader
from utils.train_utils import init_model_params
from utils.scoring_utils import score_dataset
import pandas as pd
import argparse
import json
from argparse import Namespace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config file (JSON or YAML)")
    parser.add_argument("--checkpoint", type=str, help="Optional path to model checkpoint")
    args_cfg = parser.parse_args()  
    
    

    with open(args_cfg.cfg, "r") as f:
        config = json.load(f)

    args = Namespace(**config)
    
    if args_cfg.checkpoint:
        args.checkpoint = args_cfg.checkpoint
        
    args.optimizer = args.model_optimizer
    args.lr = args.model_lr
    args.weight_decay = args.model_weight_decay
    
    if args.seed == 999:
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        np.random.seed(0)

    
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    
    args.checkpoint = args.checkpoint  
    pretrained = args.checkpoint

   
    dataset, loader = get_dataset_and_loader(args, trans_list=[], only_test=True)

   
    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    trainer = Trainer(args, model, None, loader["test"])  # No train loader
    trainer.load_checkpoint(pretrained)

    #Evaluate
    normality_scores = trainer.test()
    
    # Debug: check metadata and scores before scoring
    print(" Metadata length:", len(dataset["test"].metadata))
    print(" First few metadata entries:", dataset["test"].metadata[:3])
    print(" Number of normality scores:", len(normality_scores))

    # Score
    auc_roc, scores_np, auc_pr, eer, eer_threshold = score_dataset(normality_scores, dataset["test"].metadata, args=args)

    # Save metadata
    metadata_df = pd.DataFrame(dataset["test"].metadata)
    metadata_df.to_csv("test_metadata.csv", index=False)
    print("Saved test metadata to test_metadata.csv")

    
    print("\n--- Evaluation Results ---")
    print(f"AUC-ROC       : {auc_roc:.4f}")
    print(f"AUC-PR        : {auc_pr:.4f}")
    print(f"EER           : {eer:.4f}")
    print(f"EER Threshold : {eer_threshold:.4f}")
    print(f"# Test Samples: {scores_np.shape[0]}")


if __name__ == '__main__':
    main()





