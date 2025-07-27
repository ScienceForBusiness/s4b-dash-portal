#!/usr/bin/env python3
import os
import pandas as pd

CITY_MAX_FLOORS = {"msk_united": 84}


def precompute_floor_elasticity(data_path: str, output_path: str, city_key: str):

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Ошибка загрузки данных сделок: {e}")
        return


    df['area'] = df['area'].round(0)
    df['house_id'] = df['house_id_old']

    pivot = df.pivot_table(
        index=['house_id', 'rooms_number', 'area'],
        columns='floor',
        values='price_disc',
        aggfunc='mean'
    )

    floor_min = 1
    floor_max = CITY_MAX_FLOORS.get(city_key, int(df['floor'].quantile(0.95)))
    floor_pairs = [(i, i + 1) for i in range(floor_min, floor_max)]

    records = []
    for f0, f1 in floor_pairs:
        if f0 in pivot.columns and f1 in pivot.columns:
            sub = pivot[[f0, f1]].dropna(how='any')
            ratios = sub[f1] / sub[f0]
            mean_ratios = ratios.groupby(['house_id', 'rooms_number']).mean()
            for (house_id, _), eps in mean_ratios.items():
                if eps > 0:
                    records.append({
                        'from_floor': f0,
                        'to_floor': f1,
                        'house_id': house_id,
                        'eps': eps
                    })

    df_el = pd.DataFrame(records)
    if not df_el.empty:
        df_el = df_el.groupby(
            ['from_floor', 'to_floor', 'house_id'],
            as_index=False
        )['eps'].mean()

    house_ids = df['house_id'].unique()
    full = pd.DataFrame([
        {'from_floor': f0, 'to_floor': f1, 'house_id': hid}
        for (f0, f1) in floor_pairs
        for hid in house_ids
    ])

    results = full.merge(
        df_el,
        on=['from_floor', 'to_floor', 'house_id'],
        how='left'
    ).rename(columns={'eps': 'elasticity'})

    if results.empty:
        print("Нет данных для вычисления эластичности по этажам.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_parquet(output_path)
    print(f"Данные эластичности по этажам сохранены в {output_path}")

if __name__ == "__main__":
    cities = ["msk_united", "ekb"]
    for city in cities:
        data_path = os.path.join("data", "regions", city, "market_deals", f"{city}_geo_preprocessed_market_deals.parquet")
        output_path = os.path.join("data", "regions", city, "cache", "floor_elasticity.parquet")
        print(f"Обработка города: {city}")
        precompute_floor_elasticity(data_path, output_path, city)
