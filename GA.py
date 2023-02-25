import math,random
from collections import defaultdict

# parameters
GENERATION = 1000
POPULATION_SIZE = 30
ELITE_RATE = 0.3
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY  = 0.01

UNCHANGED_GENS = 0
mutationTimes = 0
bestValue = None
best = []
currentGeneration = 0
currentBest = None
population = []
values = [None for _ in range(POPULATION_SIZE)]
fitnessValues = [None for _ in range(POPULATION_SIZE)]
roulette = [None for _ in range(POPULATION_SIZE)]



# data process
points = {1: (38.24, 20.42), 2: (39.57, 26.15), 3: (40.56, 25.32), 4: (36.26, 23.12), 5: (33.48, 10.54),
      6: (37.56, 12.19), 7: (38.42, 13.11), 8: (37.52, 20.44), 9: (41.23, 9.1), 10:(41.17, 13.05),
      11: (36.08, -5.21),12: (38.47, 15.13), 13: (38.15, 15.35), 14:(37.51, 15.17), 15: (35.49, 14.32),
      16: (39.36, 19.56),17: (38.09, 24.36), 18: (36.09, 23.0), 19: (40.44, 13.57), 20: (40.33, 14.15),
      21: (40.37, 14.23),22: (37.57, 22.56)}
distance_matrix = defaultdict(dict)
def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R

for ka, va in points.items():
    for kb, vb in points.items():
        distance_matrix[ka][kb] = 0.0 if kb == ka else distance(va, vb)


# GA implenment
def randomIndivial():
    a = list(points.keys())
    random.shuffle(a)
    return a

def evaluate(indivial):
    sum = distance_matrix[indivial[0]][indivial[len(indivial) - 1]]
    for i in range(len(indivial)):
        sum += distance_matrix[indivial[i]][indivial[i-1]]
    return sum

def getCurrentBest():
    bestP = 0
    currentBestValue = values[0]
    for i in range(len(population)):
        if values[i] < currentBestValue:
            currentBestValue = values[i]
            bestP = i
    return {
    "bestPosition" : bestP
    , "bestValue"    : currentBestValue
  }


def setBestValue():
    global UNCHANGED_GENS
    global bestValue,currentBest,best,population
    for i in range(len(population)):
        values[i] = evaluate(population[i])
    currentBest = getCurrentBest()
    if bestValue == None or bestValue > currentBest["bestValue"]:
        best = population[currentBest["bestPosition"]]
        bestValue = currentBest["bestValue"]
        UNCHANGED_GENS = 0
    else:
        UNCHANGED_GENS += 1

def setRoulette():
    global fitnessValues,roulette
    for i in range(len(values)):
        fitnessValues[i] = 1.0/values[i]
    sum = 0
    for i in range(len(fitnessValues)):
        sum += fitnessValues[i]
    for i in range(len(roulette)):
        roulette[i] = fitnessValues[i]/sum
    for i in range(len(roulette)):
        roulette[i] += roulette[i-1]

def randomNumber(boundary):
  return int(random.random() * boundary)

def doMutate(seq):
    global mutationTimes
    mutationTimes += 1
    m = randomNumber(len(seq) - 2)
    n = randomNumber(len(seq))
    while m>=n:
        m = randomNumber(len(seq) - 2)
        n = randomNumber(len(seq))
    j = (n-m+1)>>1
    i = 0
    while i<j:
        seq[m+i],seq[n-i] = seq[n-i],seq[m+i]
        i += 1
    return seq

def pushMutate(seq):
    global mutationTimes
    mutationTimes += 1
    m = randomNumber(len(seq)>>1)
    n = randomNumber(len(seq))
    while m>=n:
        m = randomNumber(len(seq)>>1)
        n = randomNumber(len(seq))
    s1 = seq[0:m]
    s2 = seq[m:n]
    s3 = seq[n:len(seq)]
    return s2+s1+s3


def GANextGeneration():
    global currentGeneration
    currentGeneration += 1
    selection()
    crossover()
    mutation()
    setBestValue()

def wheelOut(rand):
  for i in range(len(roulette)):
    if rand <= roulette[i]:
        return i

def deleteByValue(l : list,value):
    pos = l.index(value)
    l.pop(pos)
    return l


def getChild(fun, x, y):
    global distance_matrix,population
    solution = []
    px = population[x].copy()
    py = population[y].copy()
    dx = None
    dy = None
    c = px[randomNumber(len(px))]
    solution.append(c)
    while len(px) > 1:
        if fun == 'next':
            dx = px[(px.index(c)+1)%len(px)]
            dy = py[(py.index(c)+1)%len(py)]
        else:
            dx = px[(len(px)+px.index(c)-1)%len(px)]
            dy = py[(len(py)+py.index(c)-1)%len(py)]
        px = deleteByValue(px,c)
        py = deleteByValue(py,c)
        c = dx if distance_matrix[c][dx] < distance_matrix[c][dy] else dy
        solution.append(c)
    return solution

def doCrossover(x, y):
    global population
    child1 = getChild('next', x, y)
    child2 = getChild('previous', x, y)
    population[x] = child1
    population[y] = child2

def crossover():
    queue = []
    for i in range(POPULATION_SIZE):
        if random.random() < CROSSOVER_PROBABILITY:
            queue.append(i)
    random.shuffle(queue)
    i = 0
    j = len(queue)-1
    while i<j:
        doCrossover(queue[i], queue[i+1])
        i+=2

def selection():
    global population,best
    parents = []
    initnum = 4
    parents.append(population[currentBest["bestPosition"]])
    parents.append(doMutate(best.copy()))
    parents.append(pushMutate(best.copy()))
    parents.append(best.copy())
    setRoulette()
    for i in range(initnum, POPULATION_SIZE):
        parents.append(population[wheelOut(random.random())])
    population = parents


def mutation():
    for i in range(POPULATION_SIZE):
        if random.random() < MUTATION_PROBABILITY:
            if random.random() > 0.5:
                population[i] = pushMutate(population[i])
            else:
                population[i] = doMutate(population[i])
            i-=1


if __name__ == '__main__':
    # # grid search
    # GENERATION_list = [100+i*300 for i in range(5)]
    # POPULATION_SIZE_list = [10+i*20 for i in range(5)]
    # ELITE_RATE_list = [0.1+i*0.075 for i in range(5)]
    # CROSSOVER_PROBABILITY_list = [0.5+i*0.08 for i in range(5)]
    # MUTATION_PROBABILITY_list  = [0.005+i*0.005 for i in range(5)]

    # bestE = None
    # bestParameters = None
    # count = 0
    # for i in range(5):
    #     for j in range(5):
    #         for k in range(5):
    #             for m in range(5):
    #                 for n in range(5):
    #                     GENERATION = GENERATION_list[i]
    #                     POPULATION_SIZE=POPULATION_SIZE_list[j]
    #                     ELITE_RATE =ELITE_RATE_list[k]
    #                     CROSSOVER_PROBABILITY =CROSSOVER_PROBABILITY_list[m]
    #                     MUTATION_PROBABILITY  = MUTATION_PROBABILITY_list[n]
    #                     UNCHANGED_GENS = 0
    #                     mutationTimes = 0
    #                     bestValue = None
    #                     best = []
    #                     currentGeneration = 0
    #                     currentBest = None
    #                     population = []
    #                     values = [None for _ in range(POPULATION_SIZE)]
    #                     fitnessValues = [None for _ in range(POPULATION_SIZE)]
    #                     roulette = [None for _ in range(POPULATION_SIZE)]
                        
    #                     for _ in range(POPULATION_SIZE):
    #                         population.append(randomIndivial())
    #                     setBestValue()
    #                     for _ in range(GENERATION):
    #                         GANextGeneration()

    #                     if bestE == None or bestE > bestValue:
    #                         bestE = bestValue
    #                         parameters = {
    #                             "GENERATION":GENERATION_list[i],
    #                             "POPULATION_SIZE":POPULATION_SIZE_list[j],
    #                             "ELITE_RATE":ELITE_RATE_list[k],
    #                             "CROSSOVER_PROBABILITY":CROSSOVER_PROBABILITY_list[m],
    #                             "MUTATION_PROBABILITY":MUTATION_PROBABILITY_list[n]                                
    #                         }
    #                         bestParameters = parameters.copy()
    #                     count+=1
    #                     print("\r{}".format(str(count/(5*5*5*5*5))))
    # print("\n Result:")
    # print("bestRoute:\n%i mile route:" % bestE)
    # print("bestParameter: "+str(bestParameters))


    # run 30
    # GENERATION = 100
    # POPULATION_SIZE = 70
    # ELITE_RATE = 0.1
    # CROSSOVER_PROBABILITY = 0.74
    # MUTATION_PROBABILITY  = 0.01
    # result = []
    # for _  in range(30):
    #     UNCHANGED_GENS = 0
    #     mutationTimes = 0
    #     bestValue = None
    #     best = []
    #     currentGeneration = 0
    #     currentBest = None
    #     population = []
    #     values = [None for _ in range(POPULATION_SIZE)]
    #     fitnessValues = [None for _ in range(POPULATION_SIZE)]
    #     roulette = [None for _ in range(POPULATION_SIZE)]
    #     for _ in range(POPULATION_SIZE):
    #         population.append(randomIndivial())
    #     setBestValue()
    #     for _ in range(GENERATION):
    #         GANextGeneration()
        
    #     result.append(int(bestValue))
    # average = sum(result)/len(result)
    # print("average:"+str(average))
    
    # total = 0.0
    # stddev = None
    # for v in result:
    #     total += (v - average)**2
    #     stddev = math.sqrt(total/len(result))

    # print("stddev:"+str(stddev))
    # print("result:"+str(result))



    for _ in range(POPULATION_SIZE):
        population.append(randomIndivial())
    setBestValue()
    for _ in range(GENERATION):
        GANextGeneration()
        print("the " + str(currentGeneration) + "th generation with "
                    + str(mutationTimes) + " times of mutation. best value: "
                    + str(bestValue))

    print("\n Result:")
    print("\n%i mile route:" % bestValue)
    while best[0] != 1:
            best = best[1:] + best[:1]  # rotate 1 to start
    print(" ➞  ".join([str(i) for i in best]))
    