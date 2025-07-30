"""RSMC Tokyo best-track parser.

Вытаскивает данные из текстового файла формата RSMC (NW Pacific),
преобразует в CSV и генерирует Markdown-описание столбцов.

Запуск:
    python parse_bst.py bst_all.txt --csv out.csv --md schema.md
"""


import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HEADER_INDICATOR = "66666"

DEFAULT_INPUT = Path(__file__).with_name("bst_all.txt")


def _infer_year(two_digit_year: int) -> int:
    """Convert 2-digit year to 4-digit. Heuristic: < 50 ⇒ 2000-, else 1900-."""
    return 2000 + two_digit_year if two_digit_year < 50 else 1900 + two_digit_year


def _parse_header(tokens: list[str]) -> dict:
    """Parse header line tokens into a dictionary."""
    # tokens[0] == '66666'
    header = {
        "intl_id": tokens[1],  # BBBB
        "n_data_lines_declared": int(tokens[2]),  # CCC
        "cyclone_id": tokens[3],  # DDDD
        "last_line_flag": tokens[4],  # F
        "final_analysis_lag_hr": tokens[5],  # G
        "revision_date": tokens[-1],  # I (YYYYMMDD)
    }
    # Name is tokens[6:-1]
    name_tokens = tokens[6:-1]
    header["storm_name"] = " ".join(name_tokens) if name_tokens else None
    return header


def _safe_int(val: str | None) -> int | None:
    """Convert *val* to int when possible, otherwise return ``None``.

    The additional ``isinstance`` guard helps static type checkers understand
    that ``val`` cannot be ``None`` when passed to ``int``.
    """
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_data_line(tokens: list[str]) -> dict:
    """Parse a data line (analysis line) into dict of values. Tokens already split."""
    from typing import Union

    data: dict[str, Union[int, str, float, datetime, None]] = {}

    # Mandatory fields (we assume at least these 6 present)
    t_time = tokens[0]
    yy = int(t_time[:2])
    year = _infer_year(yy)
    dt = datetime(year, int(t_time[2:4]), int(t_time[4:6]), int(t_time[6:8]))
    data["analysis_time"] = dt
    data["indicator"] = tokens[1]
    data["grade"] = _safe_int(tokens[2])

    lat_raw = _safe_int(tokens[3]) if len(tokens) > 3 else None
    data["lat_deg"] = lat_raw / 10 if lat_raw is not None else None

    lon_raw = _safe_int(tokens[4]) if len(tokens) > 4 else None
    data["lon_deg"] = lon_raw / 10 if lon_raw is not None else None
    data["central_pressure_hpa"] = _safe_int(tokens[5])

    # Optional fields following central pressure
    idx = 6
    if len(tokens) > idx:
        data["max_wind_kt"] = _safe_int(tokens[idx])
        idx += 1
    else:
        return data

    if len(tokens) > idx:
        # token with direction + radius
        h_iiii = tokens[idx]
        idx += 1
        if len(h_iiii) >= 5:
            data["r50kt_dir"] = _safe_int(h_iiii[0])
            data["r50kt_long_nm"] = _safe_int(h_iiii[1:])
        else:
            data["r50kt_dir"] = None
            data["r50kt_long_nm"] = _safe_int(h_iiii)
    if len(tokens) > idx:
        data["r50kt_short_nm"] = _safe_int(tokens[idx])
        idx += 1

    if len(tokens) > idx:
        k_llll = tokens[idx]
        idx += 1
        if len(k_llll) >= 5:
            data["r30kt_dir"] = _safe_int(k_llll[0])
            data["r30kt_long_nm"] = _safe_int(k_llll[1:])
        else:
            data["r30kt_dir"] = None
            data["r30kt_long_nm"] = _safe_int(k_llll)

    if len(tokens) > idx:
        data["r30kt_short_nm"] = _safe_int(tokens[idx])
        idx += 1

    if len(tokens) > idx:
        data["landfall_indicator"] = tokens[idx]

    return data


def parse_bst_file(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    current_header: dict | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue  # skip blank lines

            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] == HEADER_INDICATOR and len(tokens) >= 6:
                current_header = _parse_header(tokens)
            else:
                # data line
                if current_header is None:
                    # Skip if somehow data lines before header (should not happen)
                    continue
                data = _parse_data_line(tokens)
                # Merge header metadata into data row
                merged = {**current_header, **data}
                rows.append(merged)
    df = pd.DataFrame(rows)
    # Optional: sort by analysis time
    df.sort_values(["analysis_time", "intl_id"], inplace=True)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse RSMC best-track text file → CSV + Markdown schema")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Путь к bst_all.txt")
    parser.add_argument("--csv", dest="csv_out", help="Выходной CSV (по умолчанию <input>.csv)")
    parser.add_argument("--md", dest="md_out", help="Выходной Markdown (по умолчанию <input>_description.md)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    csv_out = Path(args.csv_out) if args.csv_out else input_path.with_suffix(".csv")
    md_out = Path(args.md_out) if args.md_out else input_path.with_name(f"{input_path.stem}_description.md")

    logging.info("Parsing %s", input_path)
    df = parse_bst_file(input_path)

    # Save CSV
    df.to_csv(csv_out, index=False)
    logging.info("Saved CSV → %s (rows %d)", csv_out, len(df))

    # Generate markdown description
    md = generate_markdown_description(df)
    md_out.write_text(md, encoding="utf-8")
    logging.info("Saved description → %s", md_out)


def generate_markdown_description(df: pd.DataFrame) -> str:
    column_desc = {
        "intl_id": (
            "Международный идентификатор (BB​​BB) — две последние цифры года + порядковый номер шторма",
            '"5101" — первая буря 1951 г.',
        ),
        "n_data_lines_declared": (
            "Количество строк данных (CCC), заявленное в заголовке",
            '"10" означает, что ниже должно быть 10 строк наблюдений',
        ),
        "cyclone_id": ("Серийный идентификатор тропического циклона (DDDD)", '"0014" — 14-я система сезона'),
        "storm_name": ("Имя шторма по РСМЦ Токио", "LUPIT"),
        "revision_date": ("Дата последней ревизии best-track (YYYYMMDD)", "20211104"),
        "last_line_flag": ("Флаг последней строки (0 — распад, 1 — выход из зоны ответственности)", "0"),
        "final_analysis_lag_hr": ("Смещение (ч) между последними данными и финальным анализом", "6"),
        "analysis_time": ("Время анализа (UTC)", "2021-08-13 12:00"),
        "indicator": ('Всегда "002" — служебный код формата', "002"),
        "grade": ("Категория системы: 2=TD, 3=TS, 4=STS, 5=TY, 6=ETC (L), 9=≥TS", "6"),
        "lat_deg": ("Широта центра, °N", "44.7"),
        "lon_deg": ("Долгота центра, °E", "167.5"),
        "central_pressure_hpa": ("Центральное давление, гПа", "992"),
        "max_wind_kt": ("Максимальная средняя скорость ветра, узлы", "0 – данные отсутствуют"),
        "r50kt_dir": ("Направление наибольшего радиуса ветров ≥50 kt (0–9, 0=нет)", "0"),
        "r50kt_long_nm": ("Наибольший радиус ветров ≥50 kt, морские мили", "80"),
        "r50kt_short_nm": ("Наименьший радиус ветров ≥50 kt, морские мили", "40"),
        "r30kt_dir": ("Направление наибольшего радиуса ветров ≥30 kt (0–9, 0=нет)", "3"),
        "r30kt_long_nm": ("Наибольший радиус ветров ≥30 kt, морские мили", "180"),
        "r30kt_short_nm": ("Наименьший радиус ветров ≥30 kt, морские мили", "130"),
        "landfall_indicator": ('Символ "#" если в течение часа после отметки шторм прошёл/вышел на сушу Японии', "#"),
    }

    md_lines = [
        "# Архив RSMC Best-Track (NW Pacific) — описание полей",
        "",
        f"Всего записей: {len(df):,}",
        "",
        "## Схема таблицы",
        "",
    ]
    md_lines.append("| Поле | Что означает | Пример |")
    md_lines.append("|------|--------------|--------|")
    for col in df.columns:
        info = column_desc.get(col, ("", ""))
        md_lines.append(f"| **{col}** | {info[0]} | {info[1]} |")

    md_lines.extend(
        [
            "",
            "## Примечания",
            "",
            "1. Даты приведены в UTC.",
            "2. Широта и долгота рассчитаны как `значение / 10` и положительны для северной широты и восточной долготы.",
            "3. Год в двухзначном формате интерпретируется по правилу: числа <50 относятся к XXI веку (2000+), остальные — к XX (1900+).",
            "4. Не все строки содержат сведения о радиусах 30/50-узловых ветров, поэтому соответствующие поля могут быть пустыми (NaN).",
        ]
    )

    return "\n".join(md_lines)


if __name__ == "__main__":
    main()
