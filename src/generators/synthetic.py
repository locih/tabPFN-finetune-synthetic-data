"""
Генераторы синтетических данных для обучения моделей.
Включает различные стратегии: смешанные модели, GMM, каузальные графы, SDV.
"""
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


class MixedModelGenerator:
    """
    Генератор синтетических данных на основе BGM (Bayesian Gaussian Mixture) 
    и случайного классификатора (Teacher Model).
    """
    
    @staticmethod
    def sample_bgm_params(rng):
        """Генерирует случайные параметры для BGM."""
        def log_uniform(a, b):
            return 10 ** rng.uniform(np.log10(a), np.log10(b))

        return dict(
            n_components=rng.randint(1, 31),
            covariance_type=rng.choice(["full", "tied", "diag", "spherical"]),
            tol=log_uniform(1e-5, 1e-1),
            reg_covar=log_uniform(1e-7, 1e-4),
            max_iter=rng.randint(100, 1001),
            n_init=rng.randint(1, 11),
            init_params="kmeans" if rng.rand() > 0.5 else "random",
            weight_concentration_prior_type=rng.choice(["dirichlet_process", "dirichlet_distribution"]),
            mean_precision_prior=rng.uniform(0.1, 10.0),
            warm_start=bool(rng.randint(0, 2)),
            verbose=0,
        )

    @staticmethod
    def sample_classifier(rng):
        """Генерирует случайный классификатор (Teacher Model)."""
        model_name = rng.choice(["RandomForest", "DecisionTree", "MLP", "SVC", "HistGradientBoosting"])

        def log_int(a, b):
            return int(round(10 ** rng.uniform(np.log10(a), np.log10(b))))

        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=log_int(10, 500),
                criterion=rng.choice(["gini", "log_loss", "entropy"]),
                max_depth=log_int(10, 100),
                min_samples_split=rng.randint(2, 21),
                min_samples_leaf=rng.randint(1, 11),
                max_leaf_nodes=rng.randint(10, 101),
                bootstrap=bool(rng.randint(0, 2)),
                n_jobs=-1
            )

        elif model_name == "DecisionTree":
            return DecisionTreeClassifier(
                criterion=rng.choice(["gini", "entropy", "log_loss"]),
                splitter=rng.choice(["best", "random"]),
                max_depth=log_int(5, 100),
                min_samples_split=rng.randint(2, 21),
                min_samples_leaf=rng.randint(1, 11),
                max_features=rng.choice([0.1, 0.25, 0.5, 0.75, 1.0, "sqrt", "log2", None])
            )

        elif model_name == "MLP":
            return MLPClassifier(
                hidden_layer_sizes=(rng.randint(1, 101),),
                activation=rng.choice(["relu", "logistic", "tanh"]),
                solver=rng.choice(["adam", "sgd", "lbfgs"]),
                alpha=rng.uniform(0.0001, 0.1),
                batch_size=rng.choice([32, 64, 128, "auto"]),
                learning_rate=rng.choice(["constant", "invscaling", "adaptive"]),
                learning_rate_init=rng.uniform(0.0001, 0.01),
                max_iter=rng.randint(100, 1001),
                momentum=rng.uniform(0.5, 0.95),
                nesterovs_momentum=bool(rng.randint(0, 2)),
                early_stopping=bool(rng.randint(0, 2))
            )

        elif model_name == "SVC":
            def log_float(a, b):
                return 10 ** rng.uniform(np.log10(a), np.log10(b))
            
            return SVC(
                kernel=rng.choice(["linear", "rbf", "poly", "sigmoid"]),
                C=log_float(1e-6, 1e6),
                degree=rng.randint(1, 6),
                gamma=rng.choice(["scale", "auto"]),
                coef0=rng.uniform(-1, 1),
                shrinking=bool(rng.randint(0, 2)),
                probability=True,
                tol=10 ** rng.uniform(-5, -2),
                class_weight=rng.choice([None, "balanced"]),
                max_iter=rng.randint(100, 1001),
                break_ties=bool(rng.randint(0, 2))
            )

        else:
            return HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=rng.uniform(0.01, 1.0),
                max_iter=rng.randint(50, 1001),
                max_leaf_nodes=rng.randint(5, 101),
                max_depth=rng.randint(3, 16),
                min_samples_leaf=rng.randint(5, 101),
                l2_regularization=rng.uniform(0.0, 1.0),
                max_bins=rng.randint(10, 256)
            )

    @staticmethod
    def get_fitted_preprocessor(X: pd.DataFrame):
        """Создает и обучает препроцессор для числовых и категориальных признаков."""
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        transformers = []
        
        if num_cols:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipe, num_cols))
        
        if cat_cols:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('cat', cat_pipe, cat_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            verbose_feature_names_out=False,
            sparse_threshold=0
        )
        
        preprocessor.fit(X)
        
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            feature_names = num_cols + cat_cols

        return preprocessor, feature_names

    @staticmethod
    def generate(X_train: pd.DataFrame, y_train: pd.Series, seed: int, feature_names: list, n_samples: int = None) -> tuple:
        """
        Генерирует синтетические данные на основе BGM и Teacher модели.
        
        Args:
            X_train: Признаки обучающего набора (DataFrame)
            y_train: Метки обучающего набора (Series)
            seed: Seed для воспроизводимости
            feature_names: Список имен признаков
            n_samples: Количество образцов для генерации (по умолчанию = размер X_train)
            
        Returns:
            X_synthetic, y_synthetic: Синтетические данные
        """
        if n_samples is None:
            n_samples = len(X_train)
            
        rng = np.random.RandomState(seed)
        
        bgm_params = MixedModelGenerator.sample_bgm_params(rng)
        clf = MixedModelGenerator.sample_classifier(rng)
        
        try:
            bgm = BayesianGaussianMixture(**bgm_params, random_state=seed)
            bgm.fit(X_train)
        except Exception:
            idx = rng.choice(len(X_train), n_samples, replace=True)
            return X_train.iloc[idx].reset_index(drop=True), y_train.iloc[idx].values

        try:
            clf.fit(X_train, y_train)
        except Exception:
            idx = rng.choice(len(X_train), n_samples, replace=True)
            return X_train.iloc[idx].reset_index(drop=True), y_train.iloc[idx].values

        X_synthetic_np, _ = bgm.sample(n_samples=n_samples)
        X_synthetic_df = pd.DataFrame(X_synthetic_np, columns=feature_names)
        
        y_synthetic = clf.predict(X_synthetic_df)
        
        if len(np.unique(y_synthetic)) < 2:
            raise ValueError("Generated only 1 class")

        return X_synthetic_df, y_synthetic


class GMMGenerator:
    """
    Генератор синтетических данных на основе Gaussian Mixture Model.
    Эмулирует TabPFN Prior для обучения моделей.
    """
    
    @staticmethod
    def generate(X_train: pd.DataFrame, y_train: pd.Series, seed: int, feature_names: list, n_samples: int = None) -> tuple:
        """
        Генерирует синтетические данные используя GMM.
        
        Args:
            X_train: Признаки обучающего набора (DataFrame)
            y_train: Метки обучающего набора (Series)
            seed: Seed для воспроизводимости
            feature_names: Список имен признаков
            n_samples: Количество образцов для генерации (по умолчанию = размер X_train)
            
        Returns:
            X_synthetic, y_synthetic: Синтетические данные
        """
        if n_samples is None:
            n_samples = len(X_train)
            
        np.random.seed(seed)
        random.seed(seed)
        
        full_data = X_train.copy()
        target_name = y_train.name if y_train.name else "target"
        full_data[target_name] = y_train.values
        
        n_components = random.randint(2, min(12, len(X_train) // 10 + 2))
        
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=seed,
                reg_covar=1e-5
            )
            gmm.fit(full_data)
            X_syn_np, _ = gmm.sample(n_samples)
            synthetic_df = pd.DataFrame(X_syn_np, columns=full_data.columns)
        except Exception as e:
            print(f"  [GMM Error] {e}. Fallback to bootstrap.")
            synthetic_df = full_data.sample(n=n_samples, replace=True).reset_index(drop=True)

        for col in synthetic_df.columns:
            min_v, max_v = full_data[col].min(), full_data[col].max()
            synthetic_df[col] = synthetic_df[col].clip(min_v, max_v).round().astype(int)

        y_synthetic = synthetic_df[target_name]
        X_synthetic = synthetic_df.drop(columns=[target_name])[feature_names]
        
        return X_synthetic, y_synthetic
