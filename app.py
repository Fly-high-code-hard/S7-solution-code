import streamlit as st
import pandas as pd
import numpy as np
import random

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


st.sidebar.header("Вход в систему")
username = st.sidebar.text_input('Логин')
password = st.sidebar.text_input('Пароль', type  = 'password')
if st.sidebar.button("Войти"):
    st.sidebar.success(f"Добро пожаловать, {username}!")


#база данных о всех существующих рейсах s7, сейчас данные искуственно созданные
flights = pd.DataFrame({
    "Рейс": ['S7 1000', 'S7 1002'],
    "Город отправления": ['Novosibirsk', 'Moscow'],
    "Город прибытия": ['Moscow', 'Saint-Petersburg'],
    "Время вылета": ['10:00', '12:00'],
    "Воздушное судно": ['Airbus 320', 'Boeing 300'],})
flights.index = range(1,len(flights)+1)

main, flights_data= st.tabs(["Главная","База рейсов"])

with flights_data:
    st.subheader("База рейсов S7")
    st.dataframe(flights)

with main:
    st.subheader("Введите информацию о рейсе")
    selected_flight = st.selectbox("Выбрать уже существующий рейс:", flights["Рейс"])
    if st.button('Выбрать'):
        st.write(flights[flights["Рейс"] == selected_flight])

    st.write("Или внесите данные о рейсе вручную")
    flight_number = st.text_input(f"Номер рейса")
    departure_city = st.text_input("Город вылета")
    arrival_city = st.text_input("Город прилета")
    departure_time = st.text_input("Время вылета")
    aircraft = st.text_input("Воздушное судно")
    if st.button('Подтвердить ввод'):
        data_input = pd.DataFrame({"Рейс": [flight_number], "Город отправления": [departure_city],
                                   "Город прибытия": [arrival_city], "Время вылета": [departure_time], "Воздушное судно": [aircraft]})
        st.write(data_input)

    if st.button('Выполнить прогноз задержки рейса'):
        st.write('собственно прогноз')
        #здесь должна быть ML модель, которая выполняет прогноз на основе переменных
        # полученных выше (flight_number, departure_city, arrival_city, departure_time, aircraft) и др., какие будут вам нужны в модели