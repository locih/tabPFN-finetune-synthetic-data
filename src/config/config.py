DATASET_IDS = [31, 37, 46, 1464, 40701]
SEEDS = [0, 1, 2, 3, 4]
MAX_N = 1000

OVERALL_TIME_LIMIT = 4 * 3600
PATIENCE = 10
TEST_FRACTION = 0.2
VAL_FRACTION = 0.25

TABPFN_CLASSIFIER_PATH = "autogluon/tabpfn-mix-1.0-classifier"
TABPFN_REGRESSOR_PATH = "autogluon/tabpfn-mix-1.0-regressor"

TABPFN_CONFIG_BASELINE = {
    "model_path_classifier": TABPFN_CLASSIFIER_PATH,
    "model_path_regressor": TABPFN_REGRESSOR_PATH,
    "n_ensembles": 1,
    "disable_fm_preprocessing": True,
}

TABPFN_CONFIG_FINETUNING = {
    "model_path_classifier": TABPFN_CLASSIFIER_PATH,
    "model_path_regressor": TABPFN_REGRESSOR_PATH,
    "n_ensembles": 1,
    "max_epochs": 30,
    "learning_rate": 1e-7,
    "batch_size": 4,
    "weight_decay": 0.01,
    "disable_fm_preprocessing": True,
}

TABPFN_CONFIG_5K_FAST = {
    "model_path_classifier": TABPFN_CLASSIFIER_PATH,
    "model_path_regressor": TABPFN_REGRESSOR_PATH,
    "n_ensembles": 1,
    "max_epochs": 10,
    "learning_rate": 1e-1,
    "batch_size": 16,
    "weight_decay": 0.01,
    "disable_fm_preprocessing": True,
}

SDV_EPOCHS = 300

SAMPLE_SIZES_SCALING = [100, 500, 1000, 2500, 5000, 10000]
TABPFN_TIME_LIMIT_SCALING = 60

DAG_MAX_SAMPLES = 1000
DAG_MAX_FEATURES = 50
