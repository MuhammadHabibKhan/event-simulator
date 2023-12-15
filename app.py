import mmc_priority as sim
import gantt

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')

def display_table():
    return render_template('table.html', data=sim.data)

if __name__ == '__main__':

    sim.generateTable()
    sim.generatePriority()
    sim.displayArrays()

    sim.simulate(2)
    sim.displaySimulation()

    gantt.createGantt(sim.data)

    app.run(host="127.0.0.9", port=8080, debug=True)
