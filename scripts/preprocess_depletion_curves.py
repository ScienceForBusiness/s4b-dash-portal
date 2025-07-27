import os
import pandas as pd
import numpy as np

def precompute_depletion_curves(apartment_data, house_data, convert_to_days=True, align_time_zero=True):
    curves = []
    quantiles = []
    apartment_data["contract_date"] = pd.to_datetime(apartment_data["contract_date"], errors="coerce")
    sales_by_house = apartment_data.groupby("house_id_old")
    for _, house in house_data.iterrows():
        house_id = house["house_id"]
        ndeals = house.get("ndeals", None)
        if ndeals is None or ndeals <= 0:
            continue
        try:
            house_sales = sales_by_house.get_group(house_id).copy()
        except KeyError:
            house_sales = pd.DataFrame()
        if house_sales.empty:
            if convert_to_days:
                df_curve = pd.DataFrame({"time": [0], "pct": [100], "house_id": [house_id]})
            else:
                continue
            # quantile for empty
            q_time = np.nan
        else:
            house_sales = house_sales.dropna(subset=["contract_date"]).sort_values("contract_date")
            if house_sales.empty:
                q_time = np.nan
                continue
            if convert_to_days:
                if align_time_zero:
                    base_time = house_sales["contract_date"].min()
                    house_sales["time"] = (house_sales["contract_date"] - base_time).dt.days
                else:
                    global_ref = apartment_data["contract_date"].min()
                    house_sales["time"] = (house_sales["contract_date"] - global_ref).dt.days
            else:
                house_sales["time"] = house_sales["contract_date"]
            grouped = house_sales.groupby("time").size().reset_index(name="sales_count").sort_values("time")
            start_time = 0 if (convert_to_days and align_time_zero) else grouped["time"].min()
            times = []
            percentages = []
            if not grouped.empty and grouped["time"].iloc[0] != start_time:
                times.append(start_time)
                percentages.append(100)
            cumulative = 0
            for _, row in grouped.iterrows():
                t = row["time"]
                cumulative += row["sales_count"]
                pct = 100 * (ndeals - cumulative) / ndeals
                times.append(t)
                percentages.append(pct)
            df_curve = pd.DataFrame({"time": times, "pct": percentages, "house_id": [house_id] * len(times)})
            # Compute 5% quantile time
            if not house_sales.empty:
                base = house_sales['contract_date'].min()
                q_date = house_sales['contract_date'].quantile(0.05)
                q_time = (q_date - base).days
            else:
                q_time = np.nan
        curves.append(df_curve)
        quantiles.append({'house_id': house_id, 'q05_time': q_time})
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    quantiles_df = pd.DataFrame(quantiles)
    return {'curves': curves_df, 'quantiles': quantiles_df}

if __name__ == "__main__":
    cities = ["msk_united", "ekb"]
    for city in cities:
        market_deals_dir = os.path.join("data", "regions", city, "market_deals")
        houses_info_dir = os.path.join("data", "regions", city, "houses_info")
        output_dir = os.path.join("data", "regions", city, "cache")
        os.makedirs(output_dir, exist_ok=True)
        apartment_path = os.path.join(market_deals_dir, f"{city}_geo_preprocessed_market_deals.parquet")
        house_path = os.path.join(houses_info_dir, f"{city}_houses.parquet")
        try:
            apartment_data = pd.read_parquet(apartment_path)
            house_data = pd.read_parquet(house_path)
        except Exception as e:
            print(f"Ошибка загрузки данных для города {city}: {e}")
            continue
        result = precompute_depletion_curves(apartment_data, house_data, convert_to_days=True, align_time_zero=True)
        curves = result['curves']
        quantiles = result['quantiles']
        output_path = os.path.join(output_dir, "depletion_curves.parquet")
        curves.to_parquet(output_path)
        print(f"Кривые выбытия сохранены для города {city} в: {output_path}")
        quantile_path = os.path.join(output_dir, 'depletion_quantiles.parquet')
        quantiles.to_parquet(quantile_path)
        print(f"Квантильные времена выбытия сохранены для города {city} в: {quantile_path}")