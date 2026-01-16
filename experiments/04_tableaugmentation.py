import os
import shutil
import time
import gc
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from src.config.config import (
    DATASET_IDS, SEEDS, MAX_N, OVERALL_TIME_LIMIT, PATIENCE,
    TABPFN_CONFIG_FINETUNING
)
from src.utils.data_utils import (
    load_openml_dataset, make_60_20_20_splits, compute_extended_metrics
)


def generate_task_from_dataset(df: pd.DataFrame, label: str, seed: int, 
                               reuse_original_target: bool = True, 
                               n_samples: int = None) -> pd.DataFrame:
    """
    Реализация TableAugmentation логики:
    1. Bootstrap rows (сэмплирование с повтором)
    2. Случайное выбрание признаков (50%-100%)
    """
    rng = np.random.RandomState(seed)
    
    if n_samples is None:
        n_samples = len(df)
    
    subset_indices = rng.choice(len(df), size=n_samples, replace=True)
    df_sub = df.iloc[subset_indices].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != label]
    n_feats = len(feature_cols)
    
    if n_feats > 1:
        frac = rng.uniform(0.5, 1.0)
        n_keep = int(np.ceil(n_feats * frac))
        n_keep = max(1, min(n_keep, n_feats))
        selected_features = rng.choice(feature_cols, size=n_keep, replace=False).tolist()
    else:
        selected_features = feature_cols

    if reuse_original_target:
        target_col = label
        final_features = selected_features
    else:
        potential_cols = selected_features + [label]
        target_col = rng.choice(potential_cols)
        final_features = [c for c in potential_cols if c != target_col]

    df_synth = df_sub[final_features + [target_col]].copy()
    
    return df_synth


def run_tableaugmentation_experiment(sample_size="variable"):
    results = []
    
    if sample_size == "variable":
        output_suffix = "final"
    elif sample_size == "5000":
        output_suffix = "5k_upsample_final"
    else:
        output_suffix = f"{sample_size}_final"
    
    n_samples_fixed = None
    if sample_size != "variable":
        n_samples_fixed = int(sample_size) if sample_size != "5000" else 5000

    for dataset_id in DATASET_IDS:
        print(f"\n{'='*40}")
        print(f"=== OpenML dataset {dataset_id} ===")
        try:
            df, label = load_openml_dataset(dataset_id, max_n=MAX_N)
        except Exception as e:
            print(f"Skip dataset {dataset_id}: {e}")
            continue

        for seed in SEEDS:
            print(f"\n--- Seed={seed} ---")
            train_df, val_df, test_df = make_60_20_20_splits(df, label, seed)

            exp_root = os.path.join("models", "tableaugmentation", f"dataset_{dataset_id}_seed{seed}")
            os.makedirs(exp_root, exist_ok=True)
            current_model_path = os.path.join(exp_root, "current")
            best_model_path = os.path.join(exp_root, "best")

            hyperparameters = {"TABPFNMIX": [TABPFN_CONFIG_FINETUNING]}

            start_time = time.time()
            best_val_loss = float("inf")
            best_iter = -1
            no_improve = 0
            has_best = False
            iter_idx = 0
            val_loss_history = []

            while True:
                elapsed = time.time() - start_time
                remaining = OVERALL_TIME_LIMIT - elapsed
                if remaining <= 60 or no_improve >= PATIENCE:
                    break

                iter_idx += 1
                print(f"\n[Iter {iter_idx}] Task Generation (TableAugmentation)...")

                task_seed = seed * 10000 + iter_idx
                
                try:
                    synthetic_train_df = generate_task_from_dataset(
                        train_df, label=label, seed=task_seed,
                        reuse_original_target=True, n_samples=n_samples_fixed
                    )
                    
                    current_cols = synthetic_train_df.columns
                    val_df_subset = val_df[current_cols]

                except Exception as e:
                    print(f"Generation failed: {e}")
                    no_improve += 1
                    continue

                if os.path.exists(current_model_path):
                    shutil.rmtree(current_model_path)

                try:
                    predictor = TabularPredictor(
                        label=label, path=current_model_path,
                        eval_metric="log_loss", verbosity=0
                    )
                    predictor.fit(
                        train_data=synthetic_train_df,
                        tuning_data=val_df_subset,
                        hyperparameters=hyperparameters,
                        time_limit=remaining,
                    )
                except Exception as e:
                    print(f"Training failed: {e}")
                    no_improve += 1
                    val_loss_history.append(None)
                    continue

                try:
                    metrics_val = compute_extended_metrics(predictor, val_df, label)
                    curr_val_loss = metrics_val["log_loss"]
                except Exception as e:
                    print(f"Metrics calc failed: {e}")
                    no_improve += 1
                    continue

                val_loss_history.append(curr_val_loss)
                print(f"Iter {iter_idx}: Val LogLoss = {curr_val_loss:.5f}")

                if curr_val_loss < (best_val_loss - 1e-6):
                    best_val_loss = curr_val_loss
                    best_iter = iter_idx
                    no_improve = 0
                    has_best = True
                    if os.path.exists(best_model_path):
                        shutil.rmtree(best_model_path)
                    shutil.copytree(current_model_path, best_model_path)
                    print("-> New Best Model!")
                else:
                    no_improve += 1
                
                del predictor
                gc.collect()

            if has_best:
                print(f"\nEvaluating Best Model (Iter {best_iter})...")
                try:
                    best_predictor = TabularPredictor.load(best_model_path)
                    test_metrics = compute_extended_metrics(best_predictor, test_df, label)
                    row = {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "best_iter": best_iter,
                        "best_val_log_loss": best_val_loss,
                        "test_log_loss": test_metrics["log_loss"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_f1_weighted": test_metrics["f1_weighted"],
                        "test_roc_auc": test_metrics["roc_auc"],
                    }
                    results.append(row)
                    print("Result:", row)
                    del best_predictor
                except Exception as e:
                    print(f"Final evaluation failed: {e}")
                gc.collect()

            if os.path.exists(exp_root):
                shutil.rmtree(exp_root)

        pd.DataFrame(results).to_csv(f"results/04_tableaugmentation_{output_suffix}.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    import sys
    sample_size = sys.argv[1] if len(sys.argv) > 1 else "variable"
    print(f"Running TableAugmentation experiment with sample_size={sample_size}")
    run_tableaugmentation_experiment(sample_size=sample_size)
