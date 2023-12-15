import plotly.express as px
import pandas as pd
import timeformat

def createGantt(data):

    d = pd.DataFrame()

    for process in data:

        s = timeformat.convertTime(int(process[5]))
        f = timeformat.convertTime(int(process[6]))

        frame = pd.DataFrame([dict(Servers=process[4], Start=f"2009-01-01 {s}", Finish=f"2009-01-01 {f}", Resource=f"{process[0]}", Text=f"{process[0]}", Completion_pct=int(process[8]), priority=process[3])])
        d = pd.concat([d, frame], axis=0)
    
    fig = px.timeline(d, x_start="Start", x_end="Finish", y="Servers", color="priority", text="Text", title="Gantt Chart")
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(insidetextanchor="middle")

    fig.show(renderer="browser")