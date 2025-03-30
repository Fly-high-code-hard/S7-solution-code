import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
import xgboost as xgb
import gdown
import os
from pathlib import Path

def download_model():
    model_url = "https://drive.google.com/file/d/17RktWgsYJBdMOg3BF4h6akQf71NGmR0P/view?usp=sharing"  # Замените на реальный ID
    model_path = Path("data/xgb_model_1.pkl")
    
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(model_url, str(model_path), quiet=False)
            print(f"Model downloaded to {model_path}")
        except Exception as e:
            print(f"Download failed: {str(e)}")
            raise
    return model_path

st.title("Сервис прогнозирования задержки рейса")

def set_bg_hack_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://sun9-31.userapi.com/s/v1/ig2/xS9B3xc6_v_2qg1OE0MdJtDZT0lHWQhFPB0swQsUQFouumagzOVJrpACu_wXC3dcjf54ixggrS5tufLoNbSYyGdJ.jpg?quality=95&as=32x21,48x31,72x47,108x70,160x104,240x156,360x235,480x313,540x352,640x417,720x469,1080x704,1280x834,1440x938,2560x1668&from=bu&u=FfXYYWMclyxe4wowmn9Gy9qDcFvTeAFrY7sAY-_taYk&cs=2560x1668");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <style>
        .stSidebar {{
            background: url("https://sun9-31.userapi.com/impg/jWObO7AihaBcWJOWa0sKlFCc-zoMX2qGkuOpuA/Zyys7aEHTwY.jpg?size=2560x1707&quality=95&sign=5005ed75ac5bce69396fe7dc0d3d281a&type=album");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_hack_url()


# Функция для вывода условий компенсации
def get_compensation_info(delay_minutes):
    if delay_minutes == 2:
        return "Рейс будет вовремя"
    elif delay_minutes == 3:
        return "0-2 часа\nКомпенсация не полагается"
    elif delay_minutes == 1:
        return "Более 2 часов:\nНапитки, возможность 2 телефонных звонков или 2 сообщения по электронной почте"
    elif delay_minutes == 0:
        return "Более 4 часов:\nБесплатное горячее питание\nЕсли задержка продолжается, то кормить пассажиров авиакомпания обязана каждые 6 часов в дневное время и каждые 8 часов в ночное"
    elif delay_minutes == 4:
        return "Более 6 часов:\nРазмещение в гостинице + трансфер туда и обратно"

# --------------------------------------------------------

def load_parquet_data(file_path):
    return pd.read_parquet(file_path)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def predict_delay(model, features):
    return model.predict(features)

download_model()

flights_data_path = "ml_data/user_data.parquet"
model_data_path = "ml_data/model_data.parquet"
model_path = "ml_data/xgb_model_1.pkl"

flights = load_parquet_data(flights_data_path)
model_df = load_parquet_data(model_data_path)
model = load_model(model_path)

# --------------------------------------------------------

# Инициализация переменной состояния для индекса данных
if 'start_index' not in st.session_state:
    st.session_state.start_index = 0

# Функция для отображения данных
def show_flights(start_index):
    return flights.iloc[start_index:start_index + 10]

# Главная вкладка
main, flights_data = st.tabs(["Главная", "База рейсов"])

with flights_data:
    st.subheader("База рейсов S7")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button('Предыдущие 10'):
            if st.session_state.start_index > 0:
                st.session_state.start_index -= 10

    with col2:
        if st.button('Следующие 10'):
            if st.session_state.start_index + 10 < len(flights):
                st.session_state.start_index += 10

    st.dataframe(show_flights(st.session_state.start_index))

with main:
    st.subheader("Введите информацию о рейсе")
    search_flight_input = st.text_input("Поиск рейса")
    search_results = flights[flights["flight_number"].astype(str).str.contains(search_flight_input, case=False, na=False)] if search_flight_input else pd.DataFrame()
    
    if not search_results.empty:
        st.write("Найденные рейсы:")
        st.dataframe(search_results)
        selected_flight = st.selectbox("Выберите рейс:", search_results["flight_number"])
    else:
        selected_flight = st.selectbox("Выбрать уже существующий рейс:", flights["flight_number"])
    
    if st.button('Выбрать'):
        st.write(flights[flights["flight_number"] == selected_flight])

    if st.button('Выполнить прогноз задержки рейса'):
        matching_row = model_df.iloc[[selected_flight]]
        prog_del = model.predict(matching_row)
        st.write(get_compensation_info(prog_del))
           