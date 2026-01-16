"""
Вспомогательные функции для работы с данными и метриками.
"""
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from autogluon.tabular import TabularPredictor


def load_openml_dataset(dataset_id: int, max_n: int = 1000):
    """
    Загружает датасет из OpenML.
    Если датасет больше max_n строк, берет случайную стратифицированную выборку.
    
    Args:
        dataset_id: ID датасета на OpenML
        max_n: Максимальное количество строк
        
    Returns:
        df: DataFrame с данными
        label: Имя целевой колонки
    """
    ds = openml.datasets.get_dataset(dataset_id)
    label = ds.default_target_attribute

    X, y, categorical_indicator, attribute_names = ds.get_data(
        dataset_format="dataframe",
        target=label,
    )

    df = X.copy()
    df[label] = y

    if len(df) > max_n:
        y_vals = df[label].values
        idx = np.arange(len(df))

        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=max_n,
            random_state=0,
        )
        sub_idx, _ = next(sss.split(idx, y_vals))
        df = df.iloc[sub_idx].reset_index(drop=True)

    return df, label


def make_60_20_20_splits(df: pd.DataFrame, label: str, seed: int):
    """
    Разбивает датасет на тренировочный (60%), валидационный (20%) и тестовый (20%) сеты.
    Использует стратифицированное разбиение для сохранения распределения классов.
    
    Args:
        df: Исходный DataFrame
        label: Имя целевой колонки
        seed: Seed для воспроизводимости
        
    Returns:
        train_df, val_df, test_df: Три DataFrame'а с данными
    """
    y = df[label].values
    idx = np.arange(len(df))

    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=0.6,
        random_state=seed,
    )
    train_idx, temp_idx = next(sss1.split(idx, y))

    y_temp = y[temp_idx]
    idx_temp = idx[temp_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=0.5,
        random_state=seed + 1000,
    )
    val_rel_idx, test_rel_idx = next(sss2.split(idx_temp, y_temp))

    val_idx = idx_temp[val_rel_idx]
    test_idx = idx_temp[test_rel_idx]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def compute_log_loss(predictor: TabularPredictor, df: pd.DataFrame, label: str) -> float:
    """
    Вычисляет Log Loss для классификатора TabularPredictor.
    Корректно обрабатывает бинарную и мультиклассовую классификацию.
    
    Args:
        predictor: Обученная модель TabularPredictor
        df: DataFrame с тестовыми данными
        label: Имя целевой колонки
        
    Returns:
        log_loss: Значение log loss
    """
    proba = predictor.predict_proba(df, as_pandas=True)
    y_true = df[label].values
    classes = list(predictor.class_labels)

    if isinstance(proba, pd.DataFrame):
        proba = proba[classes]
        return log_loss(y_true, proba.values, labels=classes)

    elif isinstance(proba, pd.Series):
        assert len(classes) == 2, "Series predict_proba только для бинарной классификации"
        p1 = proba.values
        p0 = 1 - p1
        probs = np.vstack([p0, p1]).T
        return log_loss(y_true, probs, labels=classes)

    else:
        return log_loss(y_true, proba, labels=classes)


def compute_extended_metrics(predictor: TabularPredictor, df: pd.DataFrame, label: str) -> dict:
    """
    Вычисляет расширенный набор метрик для классификации:
    Log Loss, Accuracy, F1-weighted, ROC-AUC.
    
    Args:
        predictor: Обученная модель TabularPredictor
        df: DataFrame с данными
        label: Имя целевой колонки
        
    Returns:
        dict: Словарь с метриками
    """
    y_true = df[label].values
    classes = list(predictor.class_labels)
    y_pred = predictor.predict(df).values
    proba = predictor.predict_proba(df, as_pandas=False)
    
    if len(classes) == 2 and proba.ndim == 1:
        p1 = proba
        p0 = 1 - p1
        proba_full = np.vstack([p0, p1]).T
        ll = log_loss(y_true, proba_full, labels=classes)
    else:
        ll = log_loss(y_true, proba, labels=classes)
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        if len(classes) == 2:
            pos_probs = proba[:, 1] if proba.ndim == 2 else proba
            auc = roc_auc_score(y_true, pos_probs)
        else:
            lb = LabelBinarizer()
            lb.fit(classes)
            y_true_bin = lb.transform(y_true)
            if y_true_bin.shape[1] == proba.shape[1]:
                auc = roc_auc_score(y_true_bin, proba, multi_class='ovr', average='weighted')
            else:
                auc = np.nan
    except ValueError:
        auc = np.nan

    return {"log_loss": ll, "accuracy": acc, "f1_weighted": f1, "roc_auc": auc}
