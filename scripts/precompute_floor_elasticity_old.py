import os
import pandas as pd
from tqdm import  tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=DeprecationWarning
)
FLOOR_ELASTICITY = [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25)
]
def floor_elasticity_hra(x: pd.DataFrame, upper_floor: pd.DataFrame) -> float:
    """
    Функция оценивает коэффициент эластичности цены от этажа при фиксированных house_id, rooms_number, area
    для конкретного перехода между этажами
    :param x: датафрейм для нижнего этажа
    :param upper_floor: датафрейм для верхнего этажа
    :return: коэффициент эластичности цены от этажа при фиксированных house_id, rooms_number, area
    """
    lower_area = x['area'].iloc[0]
    upper_floor = upper_floor[(upper_floor['area'] >= lower_area - 1)
                              & (upper_floor['area'] <= lower_area + 1)]
    if len(upper_floor) == 0:
        return -1
    return upper_floor['price_disc'].mean() / x['price_disc'].mean()

def floor_elasticity_hr(x: pd.DataFrame, elem: tuple) -> float:
    """
    Функция возвращает коэффициент эластичности цены от этажа при фиксированных house_id и rooms_number
    :param x: сделки с фиксированными параметрами house_id и rooms_number
    :param elem: конкретный переход между этажами
    :return: коэффициент эластичности цены от этажа при фиксированных параметрах house_id и rooms_number
    """
    down_floor = x[x['floor'] == elem[0]]
    upper_floor = x[x['floor'] == elem[1]]

    if len(down_floor) == 0 or len(upper_floor) == 0:
        return -1

    res = pd.DataFrame(
        down_floor.groupby(by="area")
                  .apply(lambda x_arg: floor_elasticity_hra(x_arg, upper_floor))
    )
    res = res[res[0] >= 0]
    return res[0].mean()

def house_room_filter(x: pd.DataFrame, elem: tuple) -> bool:
    """
    Функция нужна для фильтрации номера корпуса и количества комнат для данного перехода при оценке эластичности
    по номеру этажа
    :param x: сделки, сгруппированные по house_id и rooms_number
    :param elem: конкретный переход между этажами
    :return:
    """
    return {elem[0], elem[1]}.issubset(set(x['floor']))


def get_floor_elasticity(deals: pd.DataFrame, discounting_mode: str = 'trend'):

    df = deals.copy()
    df['area'] = df['area'].round(0)

    df['house_id'] = df['house_id_old']

    dtest = [df.groupby(by=['house_id', 'rooms_number'])
               .apply(lambda x: house_room_filter(x, FLOOR_ELASTICITY[i]))
             for i in tqdm(range(len(FLOOR_ELASTICITY)))]
    dtest = [pd.DataFrame(dtest[i]).reset_index() for i in range(len(dtest))]
    dtest = [dtest[i][dtest[i][0]] for i in range(len(dtest))]
    dict_house = {FLOOR_ELASTICITY[i]: [(dtest[i]['house_id'].iloc[j], dtest[i]['rooms_number'].iloc[j])
                                        for j in range(len(dtest[i]))]
                  for i in range(len(FLOOR_ELASTICITY))}


    list_results = []
    for i in tqdm(range(len(FLOOR_ELASTICITY))):
        df_temp = df[df[['house_id', 'rooms_number']].apply(tuple, axis=1).isin(dict_house[FLOOR_ELASTICITY[i]])]
        temp_series = df_temp.groupby(by=["house_id", "rooms_number"]).apply(lambda x: floor_elasticity_hr(x, FLOOR_ELASTICITY[i]))
        temp_result = temp_series.reset_index()
        temp_result = temp_result.rename(columns={0: "eps"})
        temp_result = temp_result[~temp_result["eps"].isna()]
        temp_result = temp_result[temp_result["eps"] > 0]
        temp_result["from_floor"] = FLOOR_ELASTICITY[i][0]
        temp_result["to_floor"] = FLOOR_ELASTICITY[i][1]
        list_results.append(temp_result[["from_floor", "to_floor", "house_id", "eps"]])


    df_el = pd.concat(list_results, ignore_index=True)
    df_el = df_el.groupby(['from_floor', 'to_floor', 'house_id'], as_index=False)['eps'].mean()
    house_ids = df['house_id'].unique()
    full = pd.DataFrame([
        {"from_floor": f0, "to_floor": f1, "house_id": hid}
        for (f0, f1) in FLOOR_ELASTICITY for hid in house_ids
    ])
    return full.merge(df_el, on=["from_floor", "to_floor", "house_id"], how="left")
def precompute_floor_elasticity(data_path, output_path):

    try:
        deals = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Ошибка загрузки данных сделок: {e}")
        return

    floor_df = get_floor_elasticity(deals)
    results = floor_df.rename(columns={"eps": "elasticity"})

    if results.empty:
        print("Нет данных для вычисления эластичности по этажам.")
        return

    df_out = results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_parquet(output_path)
    print(f"Предвычисленные данные эластичности по этажам сохранены в {output_path}")


if __name__ == "__main__":
    cities = ["msk_united", "ekb"]
    for city in cities:
        data_path = os.path.join("data", "regions", city, "market_deals", f"{city}_geo_preprocessed_market_deals.parquet")
        output_path = os.path.join("data", "regions", city, "cache", "floor_elasticity.parquet")
        print(f"Обработка города: {city}")
        precompute_floor_elasticity(data_path, output_path)