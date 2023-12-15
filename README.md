# Event Simulator

### About 

Event Simulator for simulation of M/M/C | M/G/C | G/G/C systems with priority. This simulator can be used to predict the number of arrivals and the service time each person may take in a given system provided the characteristics of the interarrival and service distributions selected. The results are displayed using a dynamic table built using flask and a gantt chart built using plotly.

### Input Parameters

Following are the possible inputs that may be required

1) Interarrival (IA) Distribution
2) Service Time Distribution
3) IA Mean / IA Variance / IA Standard Deviation / (min, max for uniform dist)
4) Service Time Mean / Variance / Standard Deviation / (min, max for uniform dist)

### Working

1) Arrivals are calculated using the cumulative probability formula for the selected distribution
2) Service Time is generated using random number generation formula for the selected distribution
3) Priorities for each arrival are calculated using linear congruential generator (LCG)

### Pre-Requisites:

- Python. Visit https://www.python.org/downloads/ to download Python
- Pip. To install pip, type the following commands in command prompt
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```
- Dependencies include: Flask | Plotly | Pandas | Numpy
- Install using the following command
```
pip install <insert_name>
```

## Images

### Table
![Table](https://github.com/MuhammadHabibKhan/event-simulator/blob/main/table-image.png)

### Gantt Chart
![chart](https://github.com/MuhammadHabibKhan/event-simulator/blob/main/gantt-chart-image.png)

## Possible Improvements

1) GUI for inputing characteristics of system
2) Handling and discarding un-intended inputs for the software 
