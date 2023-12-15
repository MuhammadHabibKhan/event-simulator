import math
import gantt
import random
import numpy as np

from tabulate import tabulate

from scipy.stats import norm
from scipy.stats import gamma


def LCG(SEED, A, B, M, number):
    # global seed, a, b, m
    pseudoRandomNumbers = np.zeros(number)
    x = SEED
    for i in range(number):
        x = (A * x + B) % M
        pseudoRandomNumbers[i] = float(x) / M
    return pseudoRandomNumbers


def randomBetweenRange(arr, total, lower, upper):
    array = np.copy(arr)
    for i in range(total):
        y = round(((upper - lower) * array[i] + lower),0) #uniform variate
        array[i] = int(y)
    return array


def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n+1):
        result *= i
    return result


def generateNormal(mean, stdDev):
  u1 = random.random()
  u2 = random.random()

  z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
  x = z * stdDev + mean

  return x


def generateUniform(max, min):
    x = min + (max - min) * random.random()
    return x


def generateGamma(mean, variance):
    shape = math.pow(mean, 2) / variance
    scale = variance / mean
    d = shape - 1 / 3
    c = 1 / math.sqrt(9 * d)
    while True:
        U = random.random()
        V = 1 + c * (random.random() - 0.5)
        V = V * V * V
        Z = U - c
        Z = Z / math.sqrt(V)
        X = d * Z
        while X <= -1 or math.log(U) >= (X * (1 - X) + d * math.log(X)):
            U = random.random()
            V = 1 + c * (random.random() - 0.5)
            V = V * V * V
            Z = U - c
            Z = Z / math.sqrt(V)
            X = d * Z
        V1 = random.random()
        V2 = random.random()
        W = math.exp((X - 1) / d)
        if X > 0:
            if V1 <= W:
                return scale * X
        else:
            if V1 <= 1 - W:
                return scale * X
        if V2 <= math.exp(-0.5 * X):
            return scale * X


def poissonPDF(l, x):
    return (math.exp(-l) * math.pow(l, x)) / factorial(x)


def uniformPDF(x, a, b):
    if a <= x <= b:
        return 1 / (b - a)
    else:
        return 0.0
    

def normalPDF(x, mean, std_dev):
    n = norm.pdf(x, loc=mean, scale=std_dev)
    return n


def gammaPDF(x, mean, std_dev):
    shape_parameter = (mean / std_dev) ** 2
    scale_parameter = mean / shape_parameter

    return gamma.pdf(x, a=shape_parameter, scale=scale_parameter)

arrivalMean = 0
arrivalSD = 0
a = 0
b = 0

def cumulativeProbabilityCalc(dist):

    global arrivalMean, arrivalSD

    cp = 0
    i = 0
    probability = 0
    temp = []

    if dist == "poisson":
        arrivalMean = float(input("Enter Arrival Mean: "))

    elif dist == "uniform":
        arrivalMean = float(input("Enter a: "))
        arrivalSD = float(input("Enter b: "))

    elif dist == "normal":
        arrivalMean = float(input("Enter Arrival Mean: "))
        arrivalSD = float(input("Enter Arrival SD: "))

    elif dist == "gamma":
        arrivalMean = float(input("Enter Arrival Mean: "))
        arrivalSD = float(input("Enter Arrival SD: "))

    while cp < 0.9999:

        if dist == "poisson":
            probability = poissonPDF(arrivalMean, i)

        elif dist == "uniform":
            probability = uniformPDF(i, arrivalMean, arrivalSD)

        elif dist == "normal":
            probability = normalPDF(i, arrivalMean, arrivalSD)

        elif dist == "gamma":
            probability = gammaPDF(i, arrivalMean, arrivalSD)
            
        cp += probability
        # print(cp)
        temp.append((cp))
        i += 1

    return temp


def generateExponential(serviceMean):
    # global serviceMean
    lambda_ = 1.0 / serviceMean
    U = random.random()  # Generate a random number between 0 and 1
    X = -math.log(U) / lambda_
    return X


serviceMean = 0
serviceVar = 0
serviceSD = 0

def generateServiceArray(total, dist):

    global serviceMean, serviceVar, serviceSD
    temp = []

    if (dist == "exponential"):
        serviceMean = float(input("Enter Service Mean: "))
    
    elif (dist == "gamma"):
        serviceMean = float(input("Enter Service Mean: "))
        serviceVar = float(input("Enter Service Variance: "))

    elif (dist == "normal"):
        serviceMean = float(input("Enter Service Mean: "))
        serviceSD = float(input("Enter Service Standard Deviation: "))

    elif (dist == "uniform"):
        max = float(input("Enter Max: "))
        min = float(input("Enter Min: "))

    for _ in range(total):

        if (dist == "exponential"):
            random_no = generateExponential(serviceMean)
    
        elif (dist == "gamma"):
            random_no = generateGamma(serviceMean, serviceVar)

        elif (dist == "normal"):
            random_no = generateNormal(serviceMean, serviceSD)

        elif (dist == "uniform"):
            random_no = generateUniform(max, min)

        random_no = abs(math.ceil(random_no))
        temp.append(random_no)

    return temp


def printArray(arr):
    print(" ".join(map(str, arr)))


randNum = 0.0
originalSize = 0

cp_arr = []
cp_lookup = []
serviceTime_arr = []
avg_time_b_arrival = []
interArrival_arr = []
arrivalArray = []
priorityArray = []


def generateTable():
    random.seed()
    global cp_arr, cp_lookup, avg_time_b_arrival, interArrival_arr, serviceTime_arr

    arrivalDist = input("Enter Arrival Dist: ")
                        
    cp_arr = cumulativeProbabilityCalc(arrivalDist)
    cp_arr.pop()

    cp_lookup.append(0)
    avg_time_b_arrival.append(0)

    for i in range(len(cp_arr) - 1):
        cp_lookup.append((cp_arr[i]))
        avg_time_b_arrival.append(i + 1)

    for _ in range(len(cp_arr)):
        randNum = random.random()  # Generate a random number between 0 and 1
        for j in range(len(cp_arr)):
            if randNum > cp_lookup[j] and randNum < cp_arr[j]:
                interArrival_arr.append(int(avg_time_b_arrival[j]))
                break
    
    serviceDist = input("Enter Service Dist: ")
    serviceTime_arr = generateServiceArray(len(interArrival_arr), serviceDist)
    
    arrivalArray.append(0)

    for x in range(len(interArrival_arr)-1):
        arrivalArray.append(arrivalArray[x] + interArrival_arr[x+1])
        


def generatePriority():
    global priorityArray, originalSize
    total = len(interArrival_arr)
    seed = 10112166
    A = 55
    B = 9
    M = 1994
    answer = LCG(seed, A, B, M, total)
    priorityArray = randomBetweenRange(answer, total, 1, 3)
    originalSize = len(priorityArray)



class Process:
    def __init__(self):
        self.ID = ""
        self.arrival = ""
        self.service = ""
        self.priority = ""
        self.remainingService = ""
        self.endTime = ""
        self.start = ""
        self.server = ""
        self.completion = 0


class Server:
    def __init__(self):
        self.serverName = ""
        self.status = ""
        self.currentProcess = ""
        self.start = ""
        self.end = "INF"
        self.repeat = False


sJobs = []
servers = []
processes = []
simulation = []
readyQueue = []
sameClockServer = []
completedProcess = []

itr = 0
clock = 0
shortestJob = 0
currentServer = ""
lowestPriorityServer = ""
currentRemainingService = 0

def update():
    global servers, clock, completedProcess, simulation, readyQueue, itr

    for server in servers:
        if (server.status == "busy"):

            if server.currentProcess.remainingService != 0 and server.repeat == False:

                server.currentProcess.remainingService -= 1
                print("| PID:", server.currentProcess.ID, "| Clock:", clock, "|", "Server: ", server.serverName, " | Service: ", server.currentProcess.service, " | Remaining: ", server.currentProcess.remainingService)
                
                if server.currentProcess.remainingService == 0:

                    server.currentProcess.endTime = clock
                    server.currentProcess.server = server.serverName
                    server.status = "idle"
                    server.end = clock+1

                    print()
                    print("--------------------------------------------------------------------------------------------------------")
                    print("PID:", server.currentProcess.ID, "finished at time", clock+1, " | Server: ", server.serverName)
                    print("--------------------------------------------------------------------------------------------------------")

                    completedProcess.append(server.currentProcess)

                server.currentProcess.completion = int((100-((server.currentProcess.remainingService / server.currentProcess.service)*100)))
                
                simulation.append([server.currentProcess.ID, server.currentProcess.arrival, server.currentProcess.service, server.currentProcess.priority, server.serverName, server.start, server.end , server.currentProcess.remainingService, server.currentProcess.completion])
                server.end = "INF"
                

def simulate(totalServers):
    global processes, sJobs, readyQueue, completedProcess, clock, shortestJob, servers, currentServer, currentRemainingService, lowestPriorityServer, itr, originalSize, simulation, sameClockServer

    for i in range(totalServers):
        s = Server()
        s.serverName = f"Server {i + 1}"
        s.status = "idle"
        s.flag = False
        servers.append(s)

    for x in range (len(arrivalArray)):
        i = Process()
        i.ID = x
        i.arrival = arrivalArray[x]
        i.priority = priorityArray[x]
        i.service = serviceTime_arr[x]
        i.remainingService = serviceTime_arr[x]
        processes.append(i)

    print("Original Size: ", originalSize)

    while not ( (len(processes) == 0) and (len(completedProcess) == originalSize) ):

        print("Completed Process Count: ", len(completedProcess))

        if len(processes) != 0:

            sJobs.clear()
            shortestJob = processes[0]

            for process in processes:

                if shortestJob.arrival > process.arrival:
                    sJobs.clear()
                    sJobs.append(process)
                    shortestJob = process
                
                if shortestJob.arrival == process.arrival:
                    sJobs.append(process)

            if shortestJob.arrival <= clock:
                readyQueue.extend(sJobs)

                for job in sJobs:
                    processes.remove(job)
            

        if (len(readyQueue)) != 0:

            itr = 0

            while (itr < totalServers):

                if (len(readyQueue) == 0):
                    break
                
                readyQueue.sort(key=lambda x: (x.priority, x.arrival, x.remainingService), reverse=False) # again sorting readyQueue as appending preempted tasks
                
                print("-----------------------")
                print("Ready Queue: ", end="| ")

                for process in readyQueue:

                    print(process.ID, end=" | ")

                print()
                print("-----------------------")

                currentServer = ""

                for server in servers:

                    if server.status == "idle":

                        for p in completedProcess:
                            if p.endTime == clock:
                                sameClockServer.append(p.server)

                        if (completedProcess and (server.serverName in sameClockServer)):
                            # print("HEHE")
                            continue

                        currentServer = server
                        currentServer.status = "busy"
                        currentServer.currentProcess = readyQueue[0]
                        currentServer.start = clock

                        readyQueue.remove(readyQueue[0])
                        break

                if currentServer == "":

                    lowestPriorityServer = max(servers, key=lambda x: x.currentProcess.priority)
                    print(lowestPriorityServer.serverName)

                    if (lowestPriorityServer.currentProcess.remainingService != 0):

                        if (lowestPriorityServer.currentProcess.priority > readyQueue[0].priority):

                            print(lowestPriorityServer.currentProcess.priority)
                            print(readyQueue[0].priority)

                            preemptedTask = lowestPriorityServer.currentProcess
                            readyQueue.append(preemptedTask)

                            print("Preempted Task: ", preemptedTask.ID)

                            # before preemption and server change, set lowest priority server end time
                            lowestPriorityServer.end = clock

                            lowestPriorityServer.currentProcess.completion = int((100-((lowestPriorityServer.currentProcess.remainingService / lowestPriorityServer.currentProcess.service)*100)))

                            simulation.append([lowestPriorityServer.currentProcess.ID, lowestPriorityServer.currentProcess.arrival, lowestPriorityServer.currentProcess.service, lowestPriorityServer.currentProcess.priority, lowestPriorityServer.serverName, lowestPriorityServer.start, lowestPriorityServer.end , lowestPriorityServer.currentProcess.remainingService, lowestPriorityServer.currentProcess.completion])

                            currentServer = lowestPriorityServer
                            currentServer.currentProcess = readyQueue[0]
                            
                            currentServer.start = clock
                            currentServer.end = "INF"

                            readyQueue.remove(readyQueue[0])
                            # currentServer.start = clock

                    else:
                        print("break")
                        print("Clock at Preemption: ", clock)

                        update()
                        clock += 1
                        break  # because lowest priority server has higher priority than highest priority in readyQueue

                print("--------------")
                print("Clock: ", clock)
                print("--------------")
                update()

                itr += 1
                
                if (currentServer != ""):
                   currentServer.repeat = True

                # clock += 1
                # continue

            sameClockServer.clear()

            for server in servers:
                server.repeat = False

        else:
            print("------------------")
            print("Clock outside: ", clock)
            print("------------------")
            
            update()
            # clock += 1
            # continue
        
        # update()
        clock += 1


def displayArrays():

    data = np.column_stack((cp_lookup, cp_arr, avg_time_b_arrival, interArrival_arr, arrivalArray, serviceTime_arr, priorityArray))
    headers = ["CPL", "CP", "Avg Time Between Arrival", "InterArrival", "Arrival Time", "Service Time", "Priority"]

    table = tabulate(data, headers, tablefmt="grid")
    print(table)
    
data = []

def displaySimulation():
    global data

    for x in range(len(simulation)):

        #comment the condition to see each step in table
        if (simulation[x][6] == "INF" or simulation[x][6] == "inf"):
            continue

        data.append(np.array([simulation[x][0], simulation[x][1], simulation[x][2], int(simulation[x][3]), simulation[x][4], simulation[x][5], simulation[x][6], simulation[x][7], simulation[x][8]]))

    headers = ["PID", "Arrival", "Service", "Priority", "Server", "Start", "End", "Remaining Service", "Completion %"]

    table = tabulate(np.row_stack(data), headers, tablefmt="grid")
    print(table)


if __name__ == "__main__":

    generateTable()
    generatePriority()
    displayArrays()
    
    simulate(2)
    displaySimulation()

    # T.createTable(data)
    gantt.createGantt(data)
