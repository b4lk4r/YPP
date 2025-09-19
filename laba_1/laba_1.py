from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --- зависимоти ---
try:
    import pandas as pd
    import numpy as np
except Exception as _e:
    print(
        "[ОШИБКА] Требуются пакеты pandas и numpy. Установите: pip install pandas numpy",
        file=sys.stderr,
    )
    raise

# ==========================
#   "СЛОЙ" ОШИБОК (доменные)
# ==========================

class UserInputError(Exception):
    """Ошибки пользовательского ввода (пути, аргументы и т.п.)."""


class DataError(Exception):
    """Ошибки данных/структуры CSV."""


# ==========================
#   "СЛОЙ" ЛОГИКИ (без I/O)
# ==========================

REGION_CANDIDATES: Tuple[str, ...] = (
    "region",
    "регион",
    "субъект рф",
    "субъект",
    "region_name",
    "territory",
)

def _normalize(name: str) -> str:
    return str(name).strip().lower().replace("ё", "е")


@dataclass
class StatsResult:
    region_df: pd.DataFrame
    series_name: str
    maximum: float
    minimum: float
    median: float
    mean: float
    percentiles_df: pd.DataFrame


class DataAnalyzer:
    """Чистая логика анализа. Не выполняет I/O."""

    def __init__(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise DataError("Пустой DataFrame: в файле нет данных.")
        self._df = df

    def detect_region_column(self) -> str:
        columns_norm = {col: _normalize(col) for col in self._df.columns}
        for col, norm in columns_norm.items():
            if norm in REGION_CANDIDATES:
                return col
        # эвристика: строковая/категориальная колонка с наименьшим числом уникальных
        object_like = [
            c
            for c in self._df.columns
            if pd.api.types.is_object_dtype(self._df[c])
            or pd.api.types.is_categorical_dtype(self._df[c])
            or pd.api.types.is_string_dtype(self._df[c])
        ]
        if object_like:
            return min(object_like, key=lambda c: self._df[c].nunique(dropna=True))
        raise DataError(
            "Не удалось определить колонку региона. "
            "Добавьте явную колонку 'region' / 'Регион' или укажите --region-col."
        )

    def filter_by_region(self, region_name: str, region_col: Optional[str] = None) -> pd.DataFrame:
        if not isinstance(region_name, str) or not region_name.strip():
            raise DataError("Название региона должно быть непустой строкой.")
        region_col = region_col or self.detect_region_column()
        target = _normalize(region_name)
        norm_col = self._df[region_col].astype(str).map(_normalize)
        mask_exact = norm_col.eq(target)
        region_df = self._df.loc[mask_exact].copy()
        if region_df.empty:
            mask_partial = norm_col.str.contains(target, na=False)
            region_df = self._df.loc[mask_partial].copy()
        if region_df.empty:
            raise DataError(f"Регион '{region_name}' не найден (колонка региона: '{region_col}').")
        return region_df

    def resolve_numeric_series_by_col_id(self, df: pd.DataFrame, col_id: int) -> pd.Series:
        if not isinstance(col_id, int):
            raise DataError("ID колонки должен быть целым числом.")
        columns: List[str] = list(df.columns)
        idx = col_id
        if not (0 <= idx < len(columns)):
            idx = col_id - 1  # пробуем 1-based
        if not (0 <= idx < len(columns)):
            raise DataError(
                f"Некорректный ID колонки: {col_id}. "
                f"Допустимо: 0..{len(columns)-1} (или 1..{len(columns)})."
            )
        series = pd.to_numeric(df.iloc[:, idx], errors="coerce")
        name = str(columns[idx])
        if series.notna().sum() == 0:
            raise DataError(f"В выбранной колонке '{name}' нет численных данных.")
        return series.rename(name)

    def compute_stats(self, series: pd.Series) -> Dict[str, float]:
        values = series.dropna().astype(float).values
        return {
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
        }

    def compute_percentiles(self, series: pd.Series, step: int = 5, inclusive_100: bool = True) -> pd.DataFrame:
        if step <= 0 or step > 100:
            raise DataError("Шаг перцентилей должен быть в диапазоне 1..100.")
        points = list(range(0, 100, step))
        if inclusive_100 and (not points or points[-1] != 100):
            points.append(100)
        values = series.dropna().astype(float).values
        # Совместимость с разными версиями NumPy: в старых версиях нет параметра `method`
        try:
            perc_values = np.percentile(values, points, method="linear")
        except TypeError:
            # fallback для NumPy < 1.22
            perc_values = np.percentile(values, points, interpolation="linear")
        return pd.DataFrame({"percentile": points, "value": perc_values})

    def analyze(self, region_name: str, col_id: int, region_col: Optional[str] = None) -> StatsResult:
        region_df = self.filter_by_region(region_name, region_col)
        series = self.resolve_numeric_series_by_col_id(region_df, col_id)
        stats = self.compute_stats(series)
        perc_df = self.compute_percentiles(series, step=5, inclusive_100=True)
        return StatsResult(
            region_df=region_df,
            series_name=series.name,
            maximum=stats["max"],
            minimum=stats["min"],
            median=stats["median"],
            mean=stats["mean"],
            percentiles_df=perc_df,
        )


# ==========================
#   "СЛОЙ" ВВОД/ВЫВОД (CLI)
# ==========================

def load_csv(path: str) -> pd.DataFrame:
    if not isinstance(path, str) or not path.strip():
        raise UserInputError("Путь к файлу должен быть непустой строкой.")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise UserInputError(f"Файл не найден: {path}") from e
    except pd.errors.EmptyDataError as e:
        raise DataError(f"Файл пустой или некорректный CSV: {path}") from e
    except pd.errors.ParserError as e:
        raise DataError(f"Ошибка чтения CSV: {e}") from e
    except Exception as e:
        raise DataError(f"Не удалось прочитать CSV: {e}") from e
    if df.empty:
        raise DataError("В файле нет данных (0 строк).")
    return df


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    if max_rows is not None:
        pd.set_option("display.max_rows", max_rows)
    print(df.to_string(index=False))


def print_stats(series_name: str, maximum: float, minimum: float, median: float, mean: float) -> None:
    print_header(f"Статистики по колонке: {series_name}")
    print(f"Максимум: {maximum}")
    print(f"Минимум:  {minimum}")
    print(f"Медиана:  {median}")
    print(f"Среднее:  {mean}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ЛР1: Анализ демографических данных по региону и колонке (один файл)."
    )
    parser.add_argument("--file", help="YPP/russian_demography.csv")
    parser.add_argument("--region", help="Название региона (точное или часть названия)")
    parser.add_argument("--col-id", type=int, help="ID колонки (0-based или 1-based)")
    parser.add_argument("--region-col", help="(опционально) Явное имя колонки региона")
    return parser.parse_args()


def prompt_if_needed(ns: argparse.Namespace) -> argparse.Namespace:
    try:
        if not ns.file:
            ns.file = input("Укажите путь к CSV-файлу: ").strip()
        if not ns.region:
            ns.region = input("Укажите название региона: ").strip()
        if ns.col_id is None:
            ns.col_id = int(input("Укажите ID колонки для метрик (целое): ").strip())
        return ns
    except ValueError as e:
        raise UserInputError("ID колонки должен быть целым числом.") from e


def main() -> int:
    try:
        ns = parse_args()
        ns = prompt_if_needed(ns)

        df = load_csv(ns.file)
        analyzer = DataAnalyzer(df)
        result = analyzer.analyze(region_name=ns.region, col_id=ns.col_id, region_col=ns.region_col)

        print_header("Таблица значений по выбранному региону (все колонки)")
        print_table(result.region_df)

        print_stats(
            series_name=result.series_name,
            maximum=result.maximum,
            minimum=result.minimum,
            median=result.median,
            mean=result.mean,
        )

        print_header("Таблица перцентилей (0..100, шаг 5)")
        print_table(result.percentiles_df)

        return 0

    except (UserInputError, DataError) as e:
        print("\n[ОШИБКА] " + str(e), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[ОШИБКА] Операция прервана пользователем.", file=sys.stderr)
        return 130
    except Exception as e:
        print("\n[ОШИБКА] Непредвиденная ошибка: " + str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())