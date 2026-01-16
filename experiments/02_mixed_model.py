import os
import shutil
import time
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from autogluon.tabular import TabularPredictor

from src.config.config import (
    DATASET_IDS, SEEDS, MAX_N, OVERALL_TIME_LIMIT, PATIENCE,
    TABPFN_CONFIG_5K_FAST
)
from src.utils.data_utils import (
    load_openml_dataset, make_60_20_20_splits, compute_extended_metrics
)
from src.generators.synthetic import MixedModelGenerator


def run_mixed_model_experiment(sample_size="variable"):
    results = []
    
    if sample_size == "variable":
        output_suffix = "final"
    elif sample_size == "5000":
        output_suffix = "5k_final"
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
            train_df_raw, val_df_raw, test_df_raw = make_60_20_20_splits(df, label, seed)

            X_train_raw = train_df_raw.drop(columns=[label])
            feature_names = X_train_raw.columns.tolist()
            
            try:
                preprocessor, feature_names_proc = MixedModelGenerator.get_fitted_preprocessor(X_train_raw)
            except ValueError as e:
                print(f"Preprocessing failed: {e}")
                continue
            
            X_train_proc_np = preprocessor.transform(X_train_raw).astype('float32')
            X_val_proc_np = preprocessor.transform(val_df_raw.drop(columns=[label])).astype('float32')
            X_test_proc_np = preprocessor.transform(test_df_raw.drop(columns=[label])).astype('float32')
            
            X_train_proc = pd.DataFrame(X_train_proc_np, columns=feature_names_proc)
            y_train = train_df_raw[label].reset_index(drop=True)
            
            val_df_proc = pd.DataFrame(X_val_proc_np, columns=feature_names_proc)
            val_df_proc[label] = val_df_raw[label].values
            
            test_df_proc = pd.DataFrame(X_test_proc_np, columns=feature_names_proc)
            test_df_proc[label] = test_df_raw[label].values

            exp_root = os.path.join("models", "mixed_model", f"dataset_{dataset_id}_seed{seed}")
            os.makedirs(exp_root, exist_ok=True)
            current_model_path = os.path.join(exp_root, "current")
            best_model_path = os.path.join(exp_root, "best")

            hyperparameters = {"TABPFNMIX": [TABPFN_CONFIG_5K_FAST]}

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
                print(f"\n[Iter {iter_idx}] Task Generation (Mixed Model)...")

                task_seed = seed * 10000 + iter_idx
                try:
                    X_synthetic, y_synthetic = MixedModelGenerator.generate(
                        X_train_proc, y_train, task_seed, feature_names_proc,
                        n_samples=n_samples_fixed
                    )
                    
                    synthetic_train_df = X_synthetic.copy()
                    synthetic_train_df[label] = y_synthetic
                    
                    if synthetic_train_df.isnull().any().any():
                        raise ValueError("Synthetic data contains NaNs")

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
                        tuning_data=val_df_proc,
                        hyperparameters=hyperparameters,
                        time_limit=remaining,
                    )
                except Exception as e:
                    print(f"Training failed: {e}")
                    no_improve += 1
                    val_loss_history.append(None)
                    continue

                try:
                    metrics_val = compute_extended_metrics(predictor, val_df_proc, label)
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
                    print(f"-> No improvement. Best: {best_val_loss:.5f}")
                
                del predictor
                gc.collect()

            if has_best:
                print(f"\nEvaluating Best Model (Iter {best_iter})...")
                try:
                    best_predictor = TabularPredictor.load(best_model_path)
                    test_metrics = compute_extended_metrics(best_predictor, test_df_proc, label)
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
            else:
                print("No successful model found.")

            if os.path.exists(exp_root):
                shutil.rmtree(exp_root)

        pd.DataFrame(results).to_csv(f"results/03_mixed_model_{output_suffix}.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    import sys
    sample_size = sys.argv[1] if len(sys.argv) > 1 else "variable"
    print(f"Running Mixed Model experiment with sample_size={sample_size}")
    run_mixed_model_experiment(sample_size=sample_size)
