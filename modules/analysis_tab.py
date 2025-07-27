import numpy as np
import pandas as pd
import streamlit as st
from utils.charts import (
    build_depletion_chart,
    build_elasticity_chart,
    build_histogram,
    build_scatter,
    build_floor_elasticity_chart
)
from components.widgets import safe_multiselect
from utils.translations import rus_columns

def render_analysis_tab(global_filtered_data: pd.DataFrame, house_data: pd.DataFrame, group_configs: dict):
    st.markdown("<div class='section-header'>Анализ данных</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        input[type="number"] {
            max-width: 80px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    city_key = st.session_state.get("city_key", "msk_united")

    if "chart_counter" not in st.session_state:
        st.session_state.chart_counter = 0
    if "graph_configs" not in st.session_state:
        st.session_state.graph_configs = []
    if not st.session_state.graph_configs:
        st.session_state.chart_counter += 1
        default_id = st.session_state.chart_counter
        default_config = {
            "id": default_id,
            "name": f"График {default_id}",
            "chart_type": "Кривая эластичности (площадь)",
            "selected_groups": ["Глобальный"] + (list(group_configs.keys()) if group_configs else []),
            "histogram": {
                "column": "mean_price",
                "log_transform": False,
                "remove_outliers": False,
                "lower_q": 0.05,
                "upper_q": 0.95,
                "normalize": False,
            },
            "scatter": {
                "x": "mean_price",
                "y": "mean_selling_time",
                "normalize": False,
            },
            "depletion": {"show_individual": False},
            "elasticity": {
                "split": 1,
            },
        }
        st.session_state.graph_configs.append(default_config)

    for config in st.session_state.graph_configs:
        with st.container():
            colA, colB = st.columns(2)
            with colA:
                config["name"] = st.text_input(
                    "Название графика", value=config["name"], key=f"chart_name_{config['id']}"
                )
            with colB:
                chart_options = ["Гистограмма", "Скатерплот", "Кривая выбытия", "Кривая эластичности (площадь)", "Кривая эластичности (этаж)"]

                selected_chart_type = st.selectbox(
                    "Тип графика",
                    options=chart_options,
                    index=chart_options.index(config["chart_type"]),
                    key=f"chart_type_{config['id']}"
                )
                if selected_chart_type != config["chart_type"]:
                    config["chart_type"] = selected_chart_type
                    st.rerun()

            available_groups = ["Глобальный"] + (list(group_configs.keys()) if group_configs else [])
            config["selected_groups"] = safe_multiselect(
                "Группы для анализа",
                options=available_groups,
                default=config.get("selected_groups", []),
                key=f"chart_groups_{config['id']}"
            )

            with st.expander("Настройки графика", expanded=False):
                if config["chart_type"] == "Гистограмма":
                    st.markdown("**Настройки гистограммы**", unsafe_allow_html=True)
                    numeric_cols = set()
                    for g in config["selected_groups"]:
                        df_g = global_filtered_data if g == "Глобальный" else group_configs[g]["filtered_data"]
                        numeric_cols.update(df_g.select_dtypes(include=["number"]).columns.tolist())
                    numeric_cols = sorted(list(numeric_cols), key=lambda x: (any(c.isdigit() for c in x), x.lower()))
                    default_col = config["histogram"].get("column", None)
                    if default_col not in numeric_cols:
                        default_col = "mean_price" if "mean_price" in numeric_cols else (numeric_cols[0] if numeric_cols else "")
                    hist_label_map = {rus_columns.get(col, col): col for col in numeric_cols}
                    default_label = rus_columns.get(default_col, default_col)
                    selected_label = st.selectbox(
                        "Признак для гистограммы",
                        options=list(hist_label_map.keys()),
                        index=(list(hist_label_map.keys()).index(default_label) if default_label in hist_label_map else 0),
                        key=f"hist_column_{config['id']}"
                    )
                    config["histogram"]["column"] = hist_label_map[selected_label]
                    config["histogram"]["log_transform"] = st.checkbox(
                        "Логарифмировать",
                        value=config["histogram"].get("log_transform", False),
                        key=f"hist_log_{config['id']}"
                    )
                    config["histogram"]["remove_outliers"] = st.checkbox(
                        "Удалить выбросы",
                        value=config["histogram"].get("remove_outliers", False),
                        key=f"hist_outliers_{config['id']}"
                    )
                    if config["histogram"]["remove_outliers"]:
                        config["histogram"]["lower_q"] = st.slider(
                            "Нижний квантиль",
                            min_value=0.0,
                            max_value=0.3,
                            value=config["histogram"].get("lower_q", 0.05),
                            step=0.01,
                            key=f"hist_lq_{config['id']}"
                        )
                        config["histogram"]["upper_q"] = st.slider(
                            "Верхний квантиль",
                            min_value=0.7,
                            max_value=1.0,
                            value=config["histogram"].get("upper_q", 0.95),
                            step=0.01,
                            key=f"hist_uq_{config['id']}"
                        )
                    config["histogram"]["normalize"] = st.checkbox(
                        "Нормировать гистограмму",
                        value=config["histogram"].get("normalize", False),
                        key=f"hist_normalize_{config['id']}"
                    )

                elif config["chart_type"] == "Скатерплот":
                    st.markdown("**Настройки скатерплота**", unsafe_allow_html=True)
                    numeric_cols = set()
                    for g in config["selected_groups"]:
                        df_g = global_filtered_data if g == "Глобальный" else group_configs[g]["filtered_data"]
                        numeric_cols.update(df_g.select_dtypes(include=["number"]).columns.tolist())
                    numeric_cols = sorted(list(numeric_cols), key=lambda x: (any(c.isdigit() for c in x), x.lower()))
                    scatter_label_map = {rus_columns.get(col, col): col for col in numeric_cols}
                    default_x_key = config["scatter"].get("x", None)
                    default_x_label = rus_columns.get(default_x_key, default_x_key)
                    selected_x_label = st.selectbox(
                        "Ось X",
                        options=list(scatter_label_map.keys()),
                        index=(list(scatter_label_map.keys()).index(default_x_label) if default_x_label in scatter_label_map else 0),
                        key=f"scatter_x_{config['id']}"
                    )
                    config["scatter"]["x"] = scatter_label_map[selected_x_label]
                    default_y_key = config["scatter"].get("y", None)
                    default_y_label = rus_columns.get(default_y_key, default_y_key)
                    selected_y_label = st.selectbox(
                        "Ось Y",
                        options=list(scatter_label_map.keys()),
                        index=(list(scatter_label_map.keys()).index(default_y_label) if default_y_label in scatter_label_map else 0),
                        key=f"scatter_y_{config['id']}"
                    )
                    config["scatter"]["y"] = scatter_label_map[selected_y_label]
                    config["scatter"]["normalize"] = st.checkbox(
                        "Нормировать координаты",
                        value=config["scatter"].get("normalize", False),
                        key=f"scatter_normalize_{config['id']}"
                    )

                elif config["chart_type"] == "Кривая выбытия":
                    st.markdown("**Настройки кривой выбытия**", unsafe_allow_html=True)
                    config["depletion"]["show_individual"] = st.checkbox(
                        "Показать индивидуальные кривые",
                        value=config["depletion"].get("show_individual", False),
                        key=f"depletion_individual_{config['id']}"
                    )
                    config["depletion"]["show_quantile_lines"] = st.checkbox(
                        "Показать 5%-квантиль",
                        value=config["depletion"].get("show_quantile_lines", False),
                        key=f"depletion_quantile_{config['id']}"
                    )

                elif config["chart_type"] == "Кривая эластичности (площадь)":
                    st.markdown("**Настройки кривой эластичности**", unsafe_allow_html=True)
                    split_options = list(range(1, 6))
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        default_split = int(config["elasticity"].get("split", 1))
                        config["elasticity"]["split"] = st.selectbox(
                            "Шаг сегментации (кв.м.)",
                            options=split_options,
                            index=split_options.index(default_split),
                            key=f"elasticity_split_{config['id']}"
                        )

                    precomputed_path =f"data/regions/{st.session_state.get('city_key','msk_united')}/cache/elasticity_curves.parquet"
                    try:
                        pre_df = pd.read_parquet(precomputed_path)
                        pre_df = pre_df[pre_df["split_parameter"] == config["elasticity"]["split"]]
                        seg_min = int(pre_df["area_seg"].min())
                        seg_max = int(pre_df["area_seg"].max())
                    except:
                        seg_min, seg_max = 0, 0
                    default_min = config["elasticity"].get("min_seg", seg_min)
                    default_max = config["elasticity"].get("max_seg", seg_max)
                    default_min = max(default_min, seg_min)
                    default_max = min(default_max, seg_max)
                    with col2:
                        config["elasticity"]["min_seg"] = st.number_input(
                            "Мин. площадь",
                            min_value=seg_min,
                            max_value=seg_max,
                            value=default_min,
                            step=1,
                            key=f"elasticity_min_seg_{config['id']}"
                        )
                    with col3:
                        config["elasticity"]["max_seg"] = st.number_input(
                            "Макс. площадь",
                            min_value=seg_min,
                            max_value=seg_max,
                            value=default_max,
                            step=1,
                            key=f"elasticity_max_seg_{config['id']}"
                        )

            unique_key = f"chart_{config['id']}_{config['chart_type']}"
            if config["chart_type"] == "Гистограмма":
                hist_col = config["histogram"]["column"]
                log_transform = config["histogram"]["log_transform"]
                remove_outliers = config["histogram"]["remove_outliers"]
                lq = config["histogram"]["lower_q"]
                uq = config["histogram"]["upper_q"]
                normalize = config["histogram"]["normalize"]
                chart_data = pd.DataFrame()
                for g in config["selected_groups"]:
                    df_g = (global_filtered_data if g == "Глобальный" else group_configs[g]["filtered_data"]).copy()
                    if hist_col in df_g.columns:
                        col_data = df_g[hist_col].dropna()
                        if remove_outliers and not col_data.empty:
                            lb = col_data.quantile(lq)
                            ub = col_data.quantile(uq)
                            df_g = df_g[(df_g[hist_col] >= lb) & (df_g[hist_col] <= ub)]
                        if log_transform:
                            df_g = df_g[df_g[hist_col] > 0].copy()
                            df_g[hist_col] = np.log(df_g[hist_col])
                        df_plot = df_g[[hist_col]].copy()
                        df_plot["group"] = g
                        chart_data = pd.concat([chart_data, df_plot], ignore_index=True)
                if chart_data.empty:
                    st.info("Нет данных для построения гистограммы.")
                else:
                    color_list = [("#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"]) for g in config["selected_groups"]]
                    fig_hist = build_histogram(chart_data, hist_col, config["selected_groups"], color_list, normalize, height=400)
                    fig_hist.update_layout(
                        xaxis_title=rus_columns.get(hist_col, hist_col),
                        yaxis_title=rus_columns.get("Количество" if not normalize else "Доля",
                                                    "Доля" if normalize else "Количество")
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, key=unique_key)

            elif config["chart_type"] == "Скатерплот":
                x_col = config["scatter"]["x"]
                y_col = config["scatter"]["y"]
                normalize_scatter = config["scatter"]["normalize"]
                chart_data = pd.DataFrame()
                for g in config["selected_groups"]:
                    df_g = (global_filtered_data if g == "Глобальный" else group_configs[g]["filtered_data"]).copy()
                    if x_col in df_g.columns and y_col in df_g.columns:
                        df_plot = df_g[[x_col, y_col, "project", "developer"]].copy()
                        if normalize_scatter:
                            if df_plot[x_col].nunique() > 1:
                                df_plot[x_col] = df_plot[x_col].rank(pct=True)
                            if df_plot[y_col].nunique() > 1:
                                df_plot[y_col] = df_plot[y_col].rank(pct=True)
                        df_plot["group"] = g
                        chart_data = pd.concat([chart_data, df_plot], ignore_index=True)
                if chart_data.empty:
                    st.info("Нет данных для построения скатерплота.")
                else:
                    color_list = [("#FF0000" if g == "Глобальный" else group_configs[g]["vis"]["color"]) for g in config["selected_groups"]]
                    fig_scatter = build_scatter(chart_data, x_col, y_col, config["selected_groups"], color_list, height=400)
                    fig_scatter.update_layout(
                        xaxis_title=rus_columns.get(x_col, x_col),
                        yaxis_title=rus_columns.get(y_col, y_col)
                    )

                    for trace in fig_scatter.data:
                        grp = trace.name
                        df_grp = chart_data[chart_data["group"] == grp]
                        trace.customdata = df_grp[["project", "developer"]].values
                        trace.hovertemplate = (
                            "Проект: %{customdata[0]}<br>"
                            "Застройщик: %{customdata[1]}<br>"
                            + trace.hovertemplate
                        )
                    st.plotly_chart(fig_scatter, use_container_width=True, key=unique_key)

            elif config["chart_type"] == "Кривая выбытия":
                depletion_curve_path = f"data/regions/{city_key}/cache/depletion_curves.parquet"
                fig_dep = build_depletion_chart(
                    depletion_curve_path,
                    config["selected_groups"],
                    global_filtered_data,
                    group_configs,
                    show_individual=config["depletion"].get("show_individual", False),
                    show_quantile_lines=config["depletion"].get("show_quantile_lines", False),
                    limit_to_3y=config["depletion"].get("limit_to_3y", True),
                    height=400
                )
                if fig_dep is None:
                    st.info("Нет данных для построения кривой выбытия.")
                else:
                    st.plotly_chart(fig_dep, use_container_width=True, key=unique_key)
                    if config["depletion"].get("show_quantile_lines", False):
                        from utils.utils import load_depletion_curves
                        dc = load_depletion_curves(depletion_curve_path)
                        for grp in config["selected_groups"]:
                            ids = (
                                house_data["house_id"].unique()
                                if grp == "Глобальный"
                                else group_configs[grp]["filtered_data"]["house_id"].unique()
                            )
                            times = []
                            for hid in ids:
                                dfh = dc[dc["house_id"] == hid]
                                df_cut = dfh[dfh["pct"] <= 95]
                                if not df_cut.empty:
                                    times.append(df_cut["time"].min())
                            if times:
                                mean_q = np.mean(times)
                                st.write(f"{grp}: {mean_q:.1f}")
                            else:
                                st.write(f"{grp}: нет данных")

            elif config["chart_type"] == "Кривая эластичности (площадь)":
                fig_el = build_elasticity_chart(
                    selected_groups=config["selected_groups"],
                    global_filtered_data=global_filtered_data,
                    group_configs=group_configs,
                    split_parameter=config["elasticity"]["split"],
                    min_seg=config["elasticity"]["min_seg"],
                    max_seg=config["elasticity"]["max_seg"]
                )
                if fig_el is None:
                    st.info("Нет данных для построения кривой эластичности.")
                else:
                    st.plotly_chart(fig_el, use_container_width=True, key=unique_key)

            elif config["chart_type"] == "Кривая эластичности (этаж)":
                fig_floor = build_floor_elasticity_chart(
                    selected_groups=config["selected_groups"],
                    global_filtered_data=global_filtered_data,
                    group_configs=group_configs
                )
                if fig_floor is None:
                    st.info("Нет данных для построения графика эластичности по этажам.")
                else:
                    st.plotly_chart(fig_floor, use_container_width=True, key=unique_key)
                    
            if st.button(f"Удалить график «{config['name']}»", key=f"delete_chart_{config['id']}_btn"):
                st.session_state.graph_configs = [cfg for cfg in st.session_state.graph_configs if cfg["id"] != config["id"]]
                st.rerun()
            st.markdown("<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
    if st.button("Добавить график", key="add_chart_bottom"):
        st.session_state.chart_counter += 1
        new_id = st.session_state.chart_counter
        new_config = {
            "id": new_id,
            "name": f"График {new_id}",
            "chart_type": "Гистограмма",
            "selected_groups": ["Глобальный"] + (list(group_configs.keys()) if group_configs else []),
            "histogram": {
                "column": "mean_price",
                "log_transform": False,
                "remove_outliers": False,
                "lower_q": 0.05,
                "upper_q": 0.95,
                "normalize": False,
            },
            "scatter": {
                "x": "mean_price",
                "y": "mean_selling_time",
                "normalize": False,
            },
            "depletion": {"show_individual": False},
            "elasticity": {
                "split": 1,
            },
        }
        st.session_state.graph_configs.append(new_config)
        st.rerun()
