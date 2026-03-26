import streamlit as st 
import yfinance as yf
import pandas as pd

st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and ***volume*** of GOOGLE!
         
""")

tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(start='2010-01-01', end='2020-01-01')
tickerDf = tickerData.history(period='1y')

st.write("""## Closing Price""")
st.line_chart(tickerDf.Close)

st.write("""## Volume Price""")
st.line_chart(tickerDf.Volume)

st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and ***volume*** of Apple!
         
""")

tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(start='2010-01-01', end='2020-01-01')
tickerDf = tickerData.history(period='1y')

st.write("""## Closing Price""")
st.line_chart(tickerDf.Close)

st.write("""## Volume Price""")
st.line_chart(tickerDf.Volume)