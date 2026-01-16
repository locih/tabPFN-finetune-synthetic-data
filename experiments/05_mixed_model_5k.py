"""
Эксперимент: Mixed Model с фиксированным размером 5000 образцов
Генерирует синтетические данные размером ровно 5000 строк (upsampling/downsampling).
"""
import os
import shutil
import time
import gc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor

from src.config.config import (
    DATASET_IDS, SEEDS, MAX_N, OVERALL_TIME_LIMIT, PATIENCE,
    TABPFN_CONFIG_5K_FAST
)
from src.utils.data_utils import (
    load_openml_dataset, make_60_20_20_splits, compute_extended_metrics
)
from src.generators.synthetic import MixedModelGenerator


def run_mixed_model_5k_experiment():
    """Запускает эксперимент смешанных моделей с фиксированным размером 5000."""
    results = []

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

            exp_root = os.path.join("models", "mixed_model_5k", f"dataset_{dataset_id}_seed{seed}")
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

            while True:
                elapsed = time.time() - start_time
                remaining = OVERALL_TIME_LIMIT - elapsed
                if remaining <= 60 or no_improve >= PATIENCE:
                    break

                iter_idx += 1
                print(f"\n[Iter {iter_idx}] Task Generation (Mixed Model 5k)...")

                task_seed = seed * 10000 + iter_idx
                try:
                    rng = __import__('numpy').random.RandomState(task_seed)
                    
                    bgm_params = MixedModelGenerator.sample_bgm_params(rng)
                    clf = MixedModelGenerator.sample_classifier(rng)
                    
                    try:
                        bgm = __import__('sklearn.mixture').mixture.BayesianGaussianMixture(**bgm_params, random_state=task_seed)
                        bgm.fit(X_train_proc)
                    except Exception:
                        raise
                    
                    clf.fit(X_train_proc, y_train)
                    
                    X_syn_np, _ = bgm.sample(n_samples=5000)
                    X_synthetic = pd.DataFrame(X_syn_np, columns=feature_names_proc)
                    y_synthetic = clf.predict(X_synthetic)
                    
                    if len(__import__('numpy').unique(y_synthetic)) < 2:
                        raise ValueError("Only 1 class generated")

                except Exception as e:
                    print(f"Generation failed: {e}")
                    no_improve += 1
                    continue

                synthetic_train_df = X_synthetic.copy()
                synthetic_train_df[label] = y_synthetic

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
                    continue

                try:
                    metrics_val = compute_extended_metrics(predictor, val_df_proc, label)
                    curr_val_loss = metrics_val["log_loss"]
                except Exception as e:
                    print(f"Metrics calc failed: {e}")
                    no_improve += 1
                    continue

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
                        "synthetic_size": 5000,
                    }
                    results.append(row)
                    del best_predictor
                except Exception as e:
                    print(f"Final evaluation failed: {e}")
                gc.collect()

            if os.path.exists(exp_root):
                shutil.rmtree(exp_root)

        pd.DataFrame(results).to_csv("results/mixed_model_5k_partial.csv", index=False)

    pd.DataFrame(results).to_csv("results/mixed_model_5k_final.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    run_mixed_model_5k_experiment()
