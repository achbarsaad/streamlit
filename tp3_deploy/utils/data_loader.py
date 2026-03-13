import pandas as pd
import streamlit as st
from pathlib import Path




# PAR
ROOT = Path(__file__).parent.parent
@st.cache_data
def load_clean_data():
    return pd.read_csv(ROOT / "data" / "house_prices_clean.csv")
@st.cache_data
def load_raw_data():
    return pd.read_csv(ROOT / "data" / "train.csv")

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
