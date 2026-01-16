import time
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

from src.config.config import (
    MAX_N, OVERALL_TIME_LIMIT, PATIENCE, SAMPLE_SIZES_SCALING,
    TABPFN_TIME_LIMIT_SCALING, TABPFN_CONFIG_5K_FAST
)
from src.utils.data_utils import (
    load_openml_dataset, compute_extended_metrics
)
from src.generators.synthetic import MixedModelGenerator, GMMGenerator
from src.generators.sdv_generators import CTGANGenerator, TVAEGenerator, GaussianCopulaGenerator


def run_scaling_experiment(method_name="mixed_model", dataset_id=1464, sample_sizes=None, seeds=None):
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES_SCALING
    if seeds is None:
        seeds = [0, 1, 2]
    results = []
    
    print(f"Loading Dataset {dataset_id}...")
    try:
        df_full, label = load_openml_dataset(dataset_id, max_n=MAX_N)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    hyperparameters = {"TABPFNMIX": [TABPFN_CONFIG_5K_FAST]}

    for seed in seeds:
        print(f"\n{'='*50}\nSeed: {seed}\n{'='*50}")
        
        train_raw, test_raw = train_test_split(
            df_full, test_size=0.3, random_state=seed,
            stratify=df_full[label]
        )
        train_raw, val_raw = train_test_split(
            train_raw, test_size=0.25, random_state=seed,
            stratify=train_raw[label]
        )
        
        X_train_raw = train_raw.drop(columns=[label])
        feature_names = X_train_raw.columns.tolist()
        
        try:
            preprocessor, feature_names_proc = MixedModelGenerator.get_fitted_preprocessor(X_train_raw)
        except ValueError as e:
            print(f"Preprocessing failed: {e}")
            continue
        
        X_train_proc_np = preprocessor.transform(X_train_raw).astype('float32')
        X_val_proc_np = preprocessor.transform(val_raw.drop(columns=[label])).astype('float32')
        X_test_proc_np = preprocessor.transform(test_raw.drop(columns=[label])).astype('float32')
        
        X_train_proc = pd.DataFrame(X_train_proc_np, columns=feature_names_proc)
        y_train = train_raw[label].reset_index(drop=True)
        
        val_df_proc = pd.DataFrame(X_val_proc_np, columns=feature_names_proc)
        val_df_proc[label] = val_raw[label].values
        
        test_df_proc = pd.DataFrame(X_test_proc_np, columns=feature_names_proc)
        test_df_proc[label] = test_raw[label].values

        for n_samples in SAMPLE_SIZES_SCALING:
            print(f"\n--- Size: {n_samples} ---")

            best_val_loss = float("inf")
            no_improve = 0
            has_best = False
            best_predictor = None
            
            start_time_size = time.time()
            iter_idx = 0
            
            while True:
                elapsed = time.time() - start_time_size
                if elapsed > OVERALL_TIME_LIMIT / 10 or no_improve >= PATIENCE:  # Уменьшаем time_limit для размера
                    break
                    
                iter_idx += 1
                task_seed = seed * 100000 + n_samples * 100 + iter_idx
                
                try:
                    if method_name == "mixed_model":
                        X_synth, y_synth = MixedModelGenerator.generate(
                            X_train_proc, y_train, task_seed, feature_names_proc
                        )
                    elif method_name == "gmm":
                        X_synth, y_synth = GMMGenerator.generate(
                            X_train_proc, y_train, task_seed, feature_names_proc
                        )
                    elif method_name == "ctgan":
                        X_synth, y_synth = CTGANGenerator.generate(
                            pd.concat([X_train_proc, y_train.to_frame(label)], axis=1),
                            label, n_samples, task_seed
                        )
                        X_synth = X_synth.drop(columns=[label])
                    elif method_name == "tvae":
                        X_synth, y_synth = TVAEGenerator.generate(
                            pd.concat([X_train_proc, y_train.to_frame(label)], axis=1),
                            label, n_samples, task_seed
                        )
                        X_synth = X_synth.drop(columns=[label])
                    else:
                        raise ValueError(f"Unknown method: {method_name}")
                    
                    if len(np.unique(y_synth)) < 2:
                        raise ValueError("Only 1 class in synthetic data")
                        
                except Exception as e:
                    print(f"  Generation failed: {e}")
                    no_improve += 1
                    continue

                synth_df = X_synth.copy()
                synth_df[label] = y_synth

                try:
                    predictor = TabularPredictor(
                        label=label, path=None,
                        eval_metric="log_loss", verbosity=0
                    )
                    predictor.fit(
                        train_data=synth_df,
                        tuning_data=val_df_proc,
                        hyperparameters=hyperparameters,
                        time_limit=TABPFN_TIME_LIMIT_SCALING
                    )
                except Exception as e:
                    print(f"  Fit fail: {e}")
                    no_improve += 1
                    continue

                try:
                    metrics = compute_extended_metrics(predictor, val_df_proc, label)
                    curr_loss = metrics["log_loss"]
                except Exception:
                    no_improve += 1
                    continue
                
                print(f"  Iter {iter_idx}: Val LL={curr_loss:.4f} (Best: {best_val_loss:.4f})")
                
                if curr_loss < (best_val_loss - 1e-5):
                    best_val_loss = curr_loss
                    no_improve = 0
                    has_best = True
                    best_predictor = predictor
                else:
                    no_improve += 1
                
                gc.collect()

            if has_best and best_predictor:
                try:
                    test_metrics = compute_extended_metrics(best_predictor, test_df_proc, label)
                    
                    row = {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "synthetic_size": n_samples,
                        "iters_run": iter_idx,
                        "best_val_log_loss": best_val_loss,
                        "test_log_loss": test_metrics["log_loss"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_roc_auc": test_metrics["roc_auc"]
                    }
                    results.append(row)
                    print(f"Result Size {n_samples}: Test LL={test_metrics['log_loss']:.4f}")
                    
                    del best_predictor
                except Exception as e:
                    print(f"  Final eval fail: {e}")
                gc.collect()

        pd.DataFrame(results).to_csv(f"results/06_scaling_{method_name}_final.csv", index=False)
    
    if results:
        df_results = pd.DataFrame(results)
        summary = df_results.groupby('synthetic_size')['test_log_loss'].agg(['mean', 'std', 'count'])
        summary.to_csv(f"results/06_scaling_{method_name}_summary.csv")
        print("\n" + "="*60)
        print(f"=== Summary by Synthetic Data Size ({method_name}) ===")
        print("="*60)
        print(summary)


if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "mixed_model"
    print(f"Running scaling experiment for method: {method}")
    run_scaling_experiment(method_name=method)
    print("\nDone.")
