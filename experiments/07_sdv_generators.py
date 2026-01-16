"""
Эксперимент: SDV Генераторы (CTGAN, TVAE, Gaussian Copula)
Использует библиотеку Synthetic Data Vault для генерации синтетических данных.
"""
import os
import shutil
import time
import gc
import pandas as pd
from autogluon.tabular import TabularPredictor

from src.config.config import (
    DATASET_IDS, SEEDS, MAX_N, OVERALL_TIME_LIMIT, PATIENCE,
    TABPFN_CONFIG_FINETUNING, SDV_EPOCHS
)
from src.utils.data_utils import (
    load_openml_dataset, make_60_20_20_splits, compute_extended_metrics
)
from src.generators.sdv_generators import CTGANGenerator, TVAEGenerator, GaussianCopulaGenerator


def run_sdv_experiment(generator_type='tvae'):
    """
    Запускает эксперимент с SDV генератором.
    
    Args:
        generator_type: 'ctgan', 'tvae', или 'copula'
    """
    if generator_type == 'ctgan':
        GeneratorClass = CTGANGenerator
        name = "CTGAN"
    elif generator_type == 'tvae':
        GeneratorClass = TVAEGenerator
        name = "TVAE"
    elif generator_type == 'copula':
        GeneratorClass = GaussianCopulaGenerator
        name = "Gaussian Copula"
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    results = []

    for dataset_id in DATASET_IDS:
        print(f"\n{'='*40}")
        print(f"=== OpenML dataset {dataset_id} ({name}) ===")
        try:
            df, label = load_openml_dataset(dataset_id, max_n=MAX_N)
        except Exception as e:
            print(f"Skip dataset {dataset_id}: {e}")
            continue

        for seed in SEEDS:
            print(f"\n--- Seed={seed} ---")
            train_df_raw, val_df_raw, test_df_raw = make_60_20_20_splits(df, label, seed)

            exp_root = os.path.join("models", f"sdv_{generator_type}", f"dataset_{dataset_id}_seed{seed}")
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
                print(f"\n[Iter {iter_idx}] Task Generation ({name})...")

                task_seed = seed * 10000 + iter_idx
                
                try:
                    if generator_type == 'ctgan':
                        X_syn, y_syn = CTGANGenerator.generate(
                            train_df_raw, label, task_seed, epochs=SDV_EPOCHS
                        )
                    elif generator_type == 'tvae':
                        X_syn, y_syn = TVAEGenerator.generate(
                            train_df_raw, label, task_seed, epochs=SDV_EPOCHS
                        )
                    else:
                        X_syn, y_syn = GaussianCopulaGenerator.generate(
                            train_df_raw, label, task_seed
                        )
                    
                    if not X_syn.empty:
                        X_syn, y_syn = GeneratorClass.ensure_classes_presence(
                            X_syn, y_syn,
                            train_df_raw.drop(columns=[label]),
                            train_df_raw[label]
                        )

                    synthetic_train_df = X_syn.copy()
                    synthetic_train_df[label] = y_syn

                    if synthetic_train_df.empty:
                        print("Generated data empty. Skip.")
                        no_improve += 1
                        continue

                except Exception as e:
                    print(f"Generation failed: {e}")
                    no_improve += 1
                    val_loss_history.append(None)
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
                        tuning_data=val_df_raw,
                        hyperparameters=hyperparameters,
                        time_limit=remaining,
                    )
                except Exception as e:
                    print(f"Training failed: {e}")
                    no_improve += 1
                    val_loss_history.append(None)
                    continue


                try:
                    metrics_val = compute_extended_metrics(predictor, val_df_raw, label)
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
                    test_metrics = compute_extended_metrics(best_predictor, test_df_raw, label)
                    row = {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "best_iter": best_iter,
                        "generator": generator_type,
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

        pd.DataFrame(results).to_csv(f"results/sdv_{generator_type}_partial.csv", index=False)

    pd.DataFrame(results).to_csv(f"results/sdv_{generator_type}_final.csv", index=False)
    print(f"\n✓ Done. Results saved to results/sdv_{generator_type}_final.csv")


if __name__ == "__main__":
    import sys
    generator = sys.argv[1] if len(sys.argv) > 1 else 'tvae'
    run_sdv_experiment(generator)
