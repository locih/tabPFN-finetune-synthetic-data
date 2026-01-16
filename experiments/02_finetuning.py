"""
Эксперимент: Обычный файнтюн (Finetuning)
Файнтюнинг TabPFN на оригинальных данных БЕЗ синтетического обучения.
"""
import os
import shutil
import time
import gc
import pandas as pd
from autogluon.tabular import TabularPredictor

from src.config.config import (
    DATASET_IDS, SEEDS, MAX_N, OVERALL_TIME_LIMIT,
    TABPFN_CONFIG_FINETUNING
)
from src.utils.data_utils import (
    load_openml_dataset, make_60_20_20_splits, compute_extended_metrics
)


def run_finetuning_experiment():
    """Запускает эксперимент файнтюнинга без синтетики."""
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
            train_df, val_df, test_df = make_60_20_20_splits(df, label, seed)

            exp_root = os.path.join("models", "finetuning", f"dataset_{dataset_id}_seed{seed}")
            os.makedirs(exp_root, exist_ok=True)
            model_path = os.path.join(exp_root, "model")

            hyperparameters = {"TABPFNMIX": [TABPFN_CONFIG_FINETUNING]}

            if os.path.exists(model_path):
                shutil.rmtree(model_path)

            print(f"Training TabPFN (finetuning on original data)...")
            start_time = time.time()
            
            try:
                predictor = TabularPredictor(
                    label=label, 
                    path=model_path,
                    eval_metric="log_loss", 
                    verbosity=2
                )
                predictor.fit(
                    train_data=train_df,
                    tuning_data=val_df,
                    hyperparameters=hyperparameters,
                    time_limit=OVERALL_TIME_LIMIT,
                )
                train_time = time.time() - start_time
                
            except Exception as e:
                print(f"Training failed: {e}")
                continue

            print(f"\nEvaluating on validation and test sets...")
            val_metrics = compute_extended_metrics(predictor, val_df, label)
            test_metrics = compute_extended_metrics(predictor, test_df, label)
            
            row = {
                "dataset_id": dataset_id,
                "seed": seed,
                "train_time": train_time,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "n_features": len(train_df.columns) - 1,
                "n_classes": len(predictor.class_labels),
                "val_log_loss": val_metrics["log_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_log_loss": test_metrics["log_loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
            results.append(row)
            
            print(f"\n--- Results for dataset {dataset_id}, seed {seed} ---")
            print(f"Val  - LogLoss: {val_metrics['log_loss']:.5f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"Test - LogLoss: {test_metrics['log_loss']:.5f}, Acc: {test_metrics['accuracy']:.4f}")
            
            del predictor
            gc.collect()
            
            pd.DataFrame(results).to_csv("results/finetuning_results.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/finetuning_results_final.csv", index=False)

    print("\n" + "="*60)
    print("=== Aggregated Results by Dataset ===")
    print("="*60)

    agg_stats = results_df.groupby('dataset_id').agg({
        'test_log_loss': ['mean', 'std'],
        'test_accuracy': ['mean', 'std'],
    }).round(4)

    print(agg_stats)
    agg_stats.to_csv("results/finetuning_aggregated.csv")

    print("\n✓ Done. Results saved to results/finetuning_results_final.csv")


if __name__ == "__main__":
    run_finetuning_experiment()
