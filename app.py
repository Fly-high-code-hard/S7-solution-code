import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
import xgboost as xgb

st.title("Сервис прогнозирования задержки рейса")

def set_bg_hack_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://sun9-31.userapi.com/s/v1/ig2/xS9B3xc6_v_2qg1OE0MdJtDZT0lHWQhFPB0swQsUQFouumagzOVJrpACu_wXC3dcjf54ixggrS5tufLoNbSYyGdJ.jpg?quality=95&as=32x21,48x31,72x47,108x70,160x104,240x156,360x235,480x313,540x352,640x417,720x469,1080x704,1280x834,1440x938,2560x1668&from=bu&u=FfXYYWMclyxe4wowmn9Gy9qDcFvTeAFrY7sAY-_taYk&cs=2560x1668");
            background-size: cover;
        }}
        
        /* Основная подложка цвета #bdd602 */
        .main .block-container {{
            background-color: #bdd602;
            padding: 2rem;
            border-radius: 10px;
            opacity: 0.9;
        }}
        
        /* Стили для кнопок - теперь светло-зеленые (#bdd602) */
        .stButton>button {{
            background-color: #bdd602 !important;
            color: #333333 !important;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: bold;
        }}
        
        /* Эффект при наведении на кнопку */
        .stButton>button:hover {{
            background-color: #a8c000 !important;
            color: #333333 !important;
        }}
        
        
        /* Подложка для вкладок */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 0.5rem;
            opacity: 0.9;
        }}
        
        /* Цвет текста для лучшей читаемости */
        h1, h2, h3, h4, h5, h6, p, div, span, label {{
            color: #333333 !important;
        }}
        
        /* Специальная подложка для результатов прогноза */
        .prediction-result {{
            background-color: #bdd602;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            opacity: 0.9;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_hack_url()


# Функция для вывода условий компенсации
def get_compensation_info(delay_minutes):
    if delay_minutes == 0:
        return "Рейс отменен"
    elif delay_minutes == 3:
        return "Рейс будет вовремя"
    elif delay_minutes == 4:
        return "0-2 часа\nКомпенсация не полагается"
    elif delay_minutes == 2:
        return "Более 2 часов:\nНапитки, возможность 2 телефонных звонков или 2 сообщения по электронной почте"
    elif delay_minutes == 1:
        return "Более 4 часов:\nБесплатное горячее питание\nЕсли задержка продолжается, то кормить пассажиров авиакомпания обязана каждые 6 часов в дневное время и каждые 8 часов в ночное"
    elif delay_minutes == 5:
        return "Более 6 часов:\nРазмещение в гостинице + трансфер туда и обратно"

# --------------------------------------------------------

def load_parquet_data(file_path):
    return pd.read_parquet(file_path)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def predict_delay(model, features):
    return model.predict(features)

flights_data_path = "data/data_user.parquet"
model_data_path = "data/model_data.parquet"
model_path = "data/xgb_model_1.pkl"

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
    
    # Панель с кнопками для переключения страниц
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button('Предыдущие 10'):
            if st.session_state.start_index > 0:
                st.session_state.start_index -= 10

    with col2:
        if st.button('Следующие 10'):
            if st.session_state.start_index + 10 < len(flights):
                st.session_state.start_index += 10

    # Отображение данных с переключением
    st.dataframe(show_flights(st.session_state.start_index))

with main:
    st.subheader("Введите информацию о рейсе")
    
    search_flight_input = st.text_input("", key="search_flight", placeholder="Поиск рейса")
    
    # Выпадающий список с подложкой
    search_results = flights[flights["flight_number"].astype(str).str.contains(search_flight_input, case=False, na=False)] if search_flight_input else pd.DataFrame()
    
    # Добавляем стили для текста
    st.markdown("""
        <style>
        .highlight-select {
            font-size: 18px !important;
            font-weight: bold !important;
            display: block;
            margin-bottom: -5px; /* Уменьшаем отступ перед selectbox */
        }
        </style>
        """, unsafe_allow_html=True)

    if not search_results.empty:
        st.markdown('<div class="highlight-select">Найденные рейсы:</div>', unsafe_allow_html=True)
        st.dataframe(search_results)
        st.markdown('<div class="highlight-select">Выберите рейс</div>', unsafe_allow_html=True)
        selected_flight = st.selectbox("", search_results["flight_number"])
    else:
        st.markdown('<div class="highlight-select">Выбрать уже существующий рейс</div>', unsafe_allow_html=True)
        selected_flight = st.selectbox("", flights["flight_number"])
    
    # Кнопка "Выбрать" с новым стилем
    if st.button('Выбрать'):
        st.write(flights[flights["flight_number"] == selected_flight])

    # Кнопка "Выполнить прогноз задержки рейса" с новым стилем
    if st.button('Выполнить прогноз задержки рейса'):
        matching_row = model_df[model_df["index"] == selected_flight].drop('index', axis=1)
        prog_del = model.predict(matching_row)
        # Результат с подложкой
        st.markdown(
            f"""
            <div class="prediction-result">
                <h4>Результат прогноза:</h4>
                <p>{get_compensation_info(prog_del)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )