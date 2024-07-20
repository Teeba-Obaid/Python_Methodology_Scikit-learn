import numpy as np
import plotly.express as px
import pandas as pd


def draw_spider_radar(mydict):
    df = pd.DataFrame(dict(
        r=mydict.values(),
        theta=mydict.keys()))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.show()