import math,time,random,sys
from collections import defaultdict

def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)

def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


class Annealer(object):

    """Abstract class for simulated annealing. 
        Should Overwrite two function: change_state() and calculate_energy()
    """

    # defaults parameters
    Tmax = 25000.0
    Tmin = 2.5
    steps = 50000
    updates = 100

    best_state = None
    best_energy = None
    start_time = None

    def __init__(self, initial_state=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        else:
            raise ValueError('You should type initial_state!')

    def change_state(self):
        """change the state"""
        pass

    def calculate_energy(self):
        """calculate energy in current state"""
        pass

    def set_parameters(self, parameters):
        """set the parameters"""
        self.Tmax = parameters['tmax']
        self.Tmin = parameters['tmin']
        self.steps = int(parameters['steps'])
        self.updates = int(parameters['updates'])

    def copy_state(self, state):
        return state[:]

    def update(self, step, T, E, acceptance, improvement):
        elapsed = time.time() - self.start
        if step == 0:
            print('\n Temperature        Energy    Accept   Improve     Elapsed   Remaining',
                  file=sys.stderr)
            print('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
                  .format(Temp=T,
                          Energy=E,
                          Elapsed=time_string(elapsed)),
                  file=sys.stderr, end="")
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Elapsed:s}  {Remaining:s}'
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement,
                          Elapsed=time_string(elapsed),
                          Remaining=time_string(remain)),
                  file=sys.stderr, end="")
            sys.stderr.flush()

    def anneal(self):
        """do anneal, minimize energy"""
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('minimum temperature should be greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.calculate_energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.change_state()
            if dE is None:
                E = self.calculate_energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0

        # Return best state and energy
        return self.best_state, self.best_energy

    def auto(self, minutes, steps=2000):
        """Explores the annealing landscape and
        estimates optimal temperature settings.

        Returns a dictionary suitable for the `set_schedule` method.
        """

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            E = self.calculate_energy()
            prevState = self.copy_state(self.state)
            prevEnergy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                dE = self.change_state()
                if dE is None:
                    E = self.calculate_energy()
                    dE = E - prevEnergy
                else:
                    E = prevEnergy + dE
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
            return E, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        E = self.calculate_energy()
        self.update(step, T, E, None, None)
        while T == 0.0:
            step += 1
            dE = self.change_state()
            if dE is None:
                dE = self.calculate_energy() - E
            T = abs(dE)

        # Search for Tmax - a temperature that gives 98% acceptance
        E, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T
       

        # Calculate anneal duration
        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)

        # Don't perform anneal, just return params
        return {'tmax': Tmax, 'tmin': Tmin, 'steps': duration, 'updates': self.updates}


class TSP(Annealer):

    """The annealer with TSP."""

    def __init__(self, state, distance_matrix):
        self.distance_matrix = distance_matrix
        super(TSP, self).__init__(state) 

    def change_state(self):
        """Overwrite. Random change the state"""
        initial_energy = self.calculate_energy()

        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

        return self.calculate_energy() - initial_energy

    def calculate_energy(self):
        """Overwrite. Calculates the length of the route."""
        e = 0
        for i in range(len(self.state)):
            e += self.distance_matrix[self.state[i-1]][self.state[i]]
        return e


if __name__ == '__main__':

    # csv data
    points = {'1': (38.24, 20.42), '2': (39.57, 26.15), '3': (40.56, 25.32), '4': (36.26, 23.12), '5': (33.48, 10.54),
      '6': (37.56, 12.19), '7': (38.42, 13.11), '8': (37.52, 20.44), '9': (41.23, 9.1), '10': (41.17, 13.05),
      '11': (36.08, -5.21), '12': (38.47, 15.13), '13': (38.15, 15.35), '14': (37.51, 15.17), '15': (35.49, 14.32),
      '16': (39.36, 19.56), '17': (38.09, 24.36), '18': (36.09, 23.0), '19': (40.44, 13.57), '20': (40.33, 14.15),
      '21': (40.37, 14.23), '22': (37.57, 22.56)}

    # initial state, a randomly-ordered itinerary
    init_state = list(points)
    random.shuffle(init_state)

    # create a distance matrix
    distance_matrix = defaultdict(dict)
    for ka, va in points.items():
        for kb, vb in points.items():
            distance_matrix[ka][kb] = 0.0 if kb == ka else distance(va, vb)

    tsp = TSP(init_state, distance_matrix)
    

    # # grid search
    # Tmax = [5000+i*3000 for i in range(5)]
    # Tmin = [3*math.pow(0.75,i) for i in range(5)]
    # steps = [100000+i*100000 for i in range(5)]
    # updates = [50+i*50 for i in range(5)]
    # bestE = None
    # bestParameters = None
    # count = 0
    # for i in range(5):
    #     for j in range(5):
    #         for k in range(5):
    #             for m in range(5):
    #                 count+=1
    #                 print(count/(5*5*5*5))
    #                 parameters = {"tmax":Tmax[i],"tmin":Tmin[j],"steps":steps[k],"updates":updates[m]}
    #                 tsp.set_parameters(parameters)
    #                 state, e = tsp.anneal()
    #                 if bestE == None or bestE > e:
    #                     bestE = e
    #                     bestParameters = parameters.copy()
    # print("\n Result:")
    # print("bestRoute:\n%i mile route:" % e)
    # print("bestParameter: "+str(bestParameters))

    # Record the average distance and standard deviation from the results over the 30 runs for each algorithm,
    # result = []
    # parameters = {"tmax":8000,"tmin":1.26,"steps":400000,"updates":200}
    # for _ in range(30):
    #     state, e = tsp.anneal()
    #     result.append(int(e))

    # average = sum(result)/len(result)
    # print("average:"+str(average))
    
    # total = 0.0
    # stddev = None
    # for v in result:
    #     total += (v - average)**2
    #     stddev = math.sqrt(total/len(result))

    # print("stddev:"+str(stddev))
    # print("result:"+str(result))
    



    print("\n auto generate parameters:")
    parameters = tsp.auto(minutes=0.1)
    tsp.set_parameters(parameters)
    print("\n\nParameters:  Tmax={Tmax} Tmin={Tmin} step={step}"
        .format(Tmax=parameters["tmax"],Tmin=parameters["tmin"],step=parameters["steps"]))
    print("------------------------------------------------------------")
    print("\n Simulated annealing:")
    state, e = tsp.anneal()

    while state[0] != '1':
        state = state[1:] + state[:1]  # rotate 1 to start

    print("\n Result:")
    print("\n%i mile route:" % e)
    print(" ➞  ".join(state))
