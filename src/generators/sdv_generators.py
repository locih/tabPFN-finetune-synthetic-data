import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
    SDV_VERSION = 1
except ImportError:
    try:
        from sdv.tabular import CTGAN, TVAE, GaussianCopula
        SDV_VERSION = 0
    except ImportError:
        SDV_VERSION = None


class SDVGeneratorBase:
    """Базовый класс для SDV генераторов."""
    
    @staticmethod
    def fix_dtypes(df):
        """Конвертация category -> object для SDV."""
        df_clean = df.copy()
        for col in df_clean.columns:
            if pd.api.types.is_categorical_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].astype('object')
        return df_clean

    @staticmethod
    def ensure_classes_presence(X_syn, y_syn, X_real, y_real):
        """
        Если генератор выдал только 1 класс (mode collapse),
        подмешиваем примеры недостающих классов из реальных данных.
        """
        syn_df = X_syn.copy()
        label_col = y_syn.name if hasattr(y_syn, 'name') else "target"
        syn_df[label_col] = y_syn.values
        
        real_df = X_real.copy()
        real_df[label_col] = y_real.values
        
        unique_real = y_real.unique()
        unique_syn = y_syn.unique()
        
        if len(unique_syn) < len(unique_real):
            missing_classes = set(unique_real) - set(unique_syn)
            
            for cls in missing_classes:
                real_samples = real_df[real_df[label_col] == cls]
                if len(real_samples) > 0:
                    n_inject = min(3, len(real_samples))
                    sample_to_inject = real_samples.sample(n=n_inject, replace=False)
                    syn_df.iloc[:n_inject] = sample_to_inject.values

        return syn_df.drop(columns=[label_col], errors='ignore'), syn_df[label_col]


class CTGANGenerator(SDVGeneratorBase):
    """CTGAN генератор (GAN-based)."""
    
    @staticmethod
    def generate(train_df: pd.DataFrame, label: str, seed: int, epochs: int = 300) -> tuple:
        """
        Генерирует синтетические данные используя CTGAN.
        
        Args:
            train_df: Тренировочные данные
            label: Имя целевой колонки
            seed: Random seed
            epochs: Количество эпох обучения
            
        Returns:
            X_synthetic, y_synthetic: Синтетические данные
        """
        if SDV_VERSION is None:
            raise ImportError("SDV not installed. Run: pip install sdv")
        
        np.random.seed(seed)
        df_ready = CTGANGenerator.fix_dtypes(train_df)
        
        try:
            if SDV_VERSION == 1:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df_ready)
                model = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
                model.fit(df_ready)
                synthetic_df = model.sample(num_rows=len(df_ready))
            else:
                model = __import__('sdv.tabular').tabular.CTGAN(epochs=epochs, verbose=False)
                model.fit(df_ready)
                synthetic_df = model.sample(num_rows=len(df_ready))
            
            return synthetic_df.drop(columns=[label], errors='ignore'), synthetic_df[label]
        
        except Exception as e:
            print(f"  [CTGAN Error] {e}. Fallback to bootstrap.")
            s = train_df.sample(frac=1, replace=True).reset_index(drop=True)
            return s.drop(columns=[label]), s[label]


class TVAEGenerator(SDVGeneratorBase):
    """TVAE генератор (VAE-based)."""
    
    @staticmethod
    def generate(train_df: pd.DataFrame, label: str, seed: int, epochs: int = 300) -> tuple:
        """Генерирует синтетические данные используя TVAE."""
        if SDV_VERSION is None:
            raise ImportError("SDV not installed. Run: pip install sdv")
        
        np.random.seed(seed)
        df_ready = TVAEGenerator.fix_dtypes(train_df)
        
        try:
            if SDV_VERSION == 1:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df_ready)
                model = TVAESynthesizer(metadata, epochs=epochs, verbose=False)
                model.fit(df_ready)
                synthetic_df = model.sample(num_rows=len(df_ready))
            else:
                model = __import__('sdv.tabular').tabular.TVAE(epochs=epochs, verbose=False)
                model.fit(df_ready)
                synthetic_df = model.sample(num_rows=len(df_ready))
            
            return synthetic_df.drop(columns=[label], errors='ignore'), synthetic_df[label]
        
        except Exception as e:
            print(f"  [TVAE Error] {e}. Fallback to bootstrap.")
            s = train_df.sample(frac=1, replace=True).reset_index(drop=True)
            return s.drop(columns=[label]), s[label]


class GaussianCopulaGenerator(SDVGeneratorBase):
    """Gaussian Copula генератор (статистический)."""
    
    @staticmethod
    def generate(train_df: pd.DataFrame, label: str, seed: int) -> tuple:
        """Генерирует синтетические данные используя Gaussian Copula."""
        if SDV_VERSION is None:
            raise ImportError("SDV not installed. Run: pip install sdv")
        
        np.random.seed(seed)
        df_ready = GaussianCopulaGenerator.fix_dtypes(train_df)
        
        try:
            if SDV_VERSION == 1:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(df_ready)
                model = GaussianCopulaSynthesizer(metadata)
                model.fit(df_ready)
                synthetic_df = model.sample(num_rows=len(df_ready))
            else:
                model = __import__('sdv.tabular').tabular.GaussianCopula()
                model.fit(df_ready)
                synthetic_df = model.sample(num_rows=len(df_ready))
            
            return synthetic_df.drop(columns=[label], errors='ignore'), synthetic_df[label]
        
        except Exception as e:
            print(f"  [Copula Error] {e}. Fallback to bootstrap.")
            s = train_df.sample(frac=1, replace=True).reset_index(drop=True)
            return s.drop(columns=[label]), s[label]
