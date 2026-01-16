# Описание экспериментов

## Экспериментальный процесс

Все эксперименты следуют единой схеме:

1. **Загрузка датасета** из OpenML
2. **Разбиение данных** на тренировку (60%), валидацию (20%) и тест (20%)
3. **Цикл оптимизации:**
   - Генерация синтетических данных
   - Обучение TabPFN на синтетике
   - Оценка на валидации
   - Если улучшение - сохраняем модель
   - Если no_improve >= PATIENCE - выход
4. **Финальная оценка** лучшей модели на тесте

## Эксперимент 1: Baseline (01_baseline.py)

**Описание:** TabPFN без синтетического обучения.

**Процесс:**
- Прямое обучение TabPFN на оригинальных тренировочных данных

**Результат:**
- `results/baseline_results.csv` - Полные результаты
- `results/baseline_aggregated.csv` - Статистика по датасетам

---

## Эксперимент 2: Mixed Model (02_mixed_model.py)

**Описание:** Синтетические данные через смешанные модели (BGM + Teacher).

**Команды:**
```bash
python experiments/02_mixed_model.py variable

python experiments/02_mixed_model.py 5000

```

**Результат:**
- `results/03_mixed_model_final.csv` - Для variable
- `results/03_mixed_model_5k_final.csv` - Для 5000

---

## Эксперимент 3: GMM (03_gmm.py)

**Описание:** Gaussian Mixture Model для синтеза данных.


**Команды:**
```bash
python experiments/03_gmm.py variable
python experiments/03_gmm.py 5000
```

**Результат:**
- `results/08_gmm_final.csv` - Для variable
- `results/08_gmm_5k_final.csv` - Для 5000

---

## Эксперимент 4: TableAugmentation (04_tableaugmentation.py)

**Описание:** Синтетические данные через случайный выбор признаков и целей.

**Команды:**
```bash
python experiments/04_tableaugmentation.py variable
python experiments/04_tableaugmentation.py 5000
```

**Результат:**
- `results/04_tableaugmentation_final.csv` - Для variable
- `results/04_tableaugmentation_5k_upsample_final.csv` - Для 5000

---

## Эксперимент 6: Scaling Study (06_scaling_experiment.py)

**Описание:** Исследует как производительность зависит от размера синтетической выборки.

**Процесс:**
1. Тестирует один датасет (Blood Transfusion 1464) 
2. Фиксирует метод синтеза данных
3. Варьирует размер синтетической выборки: [100, 500, 1k, 2.5k, 5k, 10k]
4. Для каждого размера - цикл оптимизации с разными конфигами генератора

**Поддерживаемые методы:**
- `mixed_model` - BGM + Teacher (по умолчанию)
- `gmm` - Gaussian Mixture Model
- `ctgan` - SDV CTGAN
- `tvae` - SDV TVAE

**Параметры:**
- `dataset_id`: 1464 (Blood Transfusion)
- `sample_sizes`: [100, 500, 1000, 2500, 5000, 10000]
- `seeds`: [0, 1, 2] для повторяемости

**Команда:**
```bash
python experiments/06_scaling_experiment.py mixed_model

python experiments/06_scaling_experiment.py gmm

python experiments/06_scaling_experiment.py tvae

python experiments/06_scaling_experiment.py ctgan
```

**Результат:**
- `results/06_scaling_{method_name}_final.csv` - Результаты по размерам
  - `synthetic_size`: Количество сгенерированных образцов
  - `test_log_loss`, `test_accuracy`: Метрики
  - `iters_run`: Количество итераций оптимизации

