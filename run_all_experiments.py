"""
Мастер скрипт для запуска всех экспериментов по очереди.
"""
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_experiments():
    """Запускает все доступные эксперименты."""
    
    experiments = [
        ("Baseline (без синтетики)", "experiments/01_baseline.py", []),
        ("Finetuning (на реальных данных)", "experiments/02_finetuning.py", []),
        ("Mixed Model (переменный размер)", "experiments/02_mixed_model.py", ["variable"]),
        ("Mixed Model (5000 образцов)", "experiments/02_mixed_model.py", ["5000"]),
        ("TableAugmentation (переменный размер)", "experiments/04_tableaugmentation.py", ["variable"]),
        ("TableAugmentation (5000 образцов)", "experiments/04_tableaugmentation.py", ["5000"]),
        ("GMM (переменный размер)", "experiments/03_gmm.py", ["variable"]),
        ("GMM (5000 образцов)", "experiments/03_gmm.py", ["5000"]),
        ("SDV Generators (CTGAN, TVAE, Copula)", "experiments/07_sdv_generators.py", []),
    ]
    
    print("="*70)
    print("  TabPFN Synthetic Data Experiments - Master Runner")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_start = time.time()
    
    for i, (name, script_path, args) in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(experiments)}] {name}")
        print(f"Script: {script_path}")
        if args:
            print(f"Args: {' '.join(args)}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        exp_start = time.time()
        
        try:
            import subprocess
            cmd = ["python", script_path] + args
            result = subprocess.run(cmd, check=True)
            if result.returncode != 0:
                print(f"⚠️ Experiment failed with return code {result.returncode}")
                continue
        
        except Exception as e:
            print(f"❌ Error in experiment: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        elapsed = time.time() - exp_start
        print(f"\n✅ Experiment completed in {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print("\nResults summary:")
    print("-" * 70)
    
    results_files = [
        "results/baseline_results_final.csv",
        "results/mixed_model_final.csv",
        "results/gmm_final.csv",
    ]
    
    import pandas as pd
    for result_file in results_files:
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            avg_test_loss = df['test_log_loss'].mean()
            std_test_loss = df['test_log_loss'].std()
            print(f"\n{os.path.basename(result_file)}:")
            print(f"  Average test log_loss: {avg_test_loss:.5f} ± {std_test_loss:.5f}")
            print(f"  Number of results: {len(df)}")
        else:
            print(f"\n{result_file}: NOT FOUND (experiment may have failed)")


if __name__ == "__main__":
    run_experiments()
