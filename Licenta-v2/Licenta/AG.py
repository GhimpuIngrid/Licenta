import copy

from Environment import FroggerEnv
import random
import numpy as np
import time


class FroggerGA:
    def __init__(self, population_size, generations, mutation_rate, environment):

        self.random_crossover = False
        self.population_size = population_size
        self.population = []
        self.selection_pool = []
        self.generations = generations

        self.BP = [0.25, 0.75, 1]
        self.BW = [0.25, 0.75, 1]
        self.cross_weak = 0

        self.mutation_rate = mutation_rate
        self.mutation_random = False
        self.max_mutation = 3
        self.mutation_nr = 3
        self.crossover2_rate = 0.35

        self.parents_elite_number = 2
        self.children_elite_number = 2
        self.roulette_number = 50
        self.tour_number = 46
        self.tour_part = 2


        self.environment = environment  # Environment-ul Frogger

        self.weights = [1 / 3, 1 / 3, 1 / 3]  # Ponderi inițiale pentru selecție
        self.initialize_population()
        self.sort()
        # print(f"Populatia nesortata: {self.population}")
        # self.calc_fitness()
        # self.afiseaza()
        # print(f"Populatia sortata: {self.population}")
        # self.fitness_scores = []
        # self.environment.gameApp.state = 'PLAYING'

    """def afiseaza(self):
        for ind in self.population:
            print(f"individul: {ind} are fitness-ul: {ind.fitness}")"""

    def initialize_population(self):
        """Inițializează populația cu acțiuni random."""

        for _ in range(self.population_size):
            not_ok = True
            while not_ok is True:
                genome = np.random.randint(0, self.environment.action_space, size=200).tolist()
                # print(genome)
                individual = Genome(genome)
                individual.lane = self.fitness(individual)

                if individual.death > 9:
                    individual.fitness = individual.lane - individual.death / 200
                    # print(individual.death)
                    not_ok = False
                    self.population.append(individual)

    def unique_ind(self):
        unique = set(tuple(id.val) for id in self.population)
        # print(len(unique))

    """def calc_fitness(self):
        for ind in self.population:
            #print(ind.val)
            ind.fitness = self.fitness(ind)"""

    def check(self, ind):
        for i in self.selection_pool:
            if i.val == ind.val:
                return False

        return True

    def sort(self):
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

    def fitness(self, genome):
        """Calculează fitness-ul unui individ rulându-l în mediu."""
        self.environment.reset()
        self.environment.gameApp.draw()
        #self.environment.gameApp.state = "PLAYING"
        fit = 1

        for i, action in enumerate(genome.val):
            _, reward, done, _ = self.environment.step(action)
            #self.environment.gameApp.draw()

            #print("lane ", self.environment.gameApp.current_lane)
            #print(reward)
            if reward >= 20:
                fit += 1
            elif reward == -5 and fit > 1:
                fit -= 1
            # print(f"Acțiune: {action}, Recompensă: {reward}")
            if done:
                genome.death = i
                #print(fit, genome.death)
                break

        # print(f"Individul: {genome} are fitnessul: {total_reward}")

        return fit

    # Mai trb de gandit care ar fi cea mai buna modalitate de mutatie
    def mutate(self):
        """Aplică mutație aleatorie pe genom."""
        for ind in reversed(self.population):
            if random.random() < self.mutation_rate:
                nr = random.randint(1, self.max_mutation) if self.mutation_random is True else self.mutation_nr
                mutations = []
                # Se poate face param global
                if random.random() < 0.7:
                    mutations.append(ind.death)
                    nr -= 1

                while nr > 0:
                    k = random.randint(0, len(ind.val) - 1)
                    if k not in mutations:
                        mutations.append(k)
                        nr -= 1

                child = copy.deepcopy(ind)
                # child = ind

                for i in mutations:
                    # print(i)
                    not_ok = True
                    while not_ok is True:
                        action = random.randint(0, self.environment.action_space)
                        if action != child.val[i]:
                            child.val[i] = action
                            not_ok = False

                self.selection_pool.append(child)

    def crossover1(self):

        # print(len(self.population))
        self.cross_weak = 0

        for i, parent1 in enumerate(reversed(self.population)):
            # print(i, parent1)
            if self.random_crossover is True:
                subpopulation = self.population
                parent2 = random.choice(subpopulation)

            elif (random.random() < self.cross_weak and i > 0) or i == self.population_size - 1:
                subpopulation = self.population[:i]
                #print(i)
                #print(subpopulation)
                parent2 = random.choice(subpopulation)

            else:
                subpopulation = self.population[i + 1:]
                #print(i)
                #print(subpopulation)
                parent2 = random.choice(subpopulation)

            # Actual crossover starts here
            k = 0
            # To avoid crossover breakpoints to override each other when the frog dies too early
            # The brakpoint based crossover needs more testing
            not_ok = True
            while not_ok is True:
                if parent1.death < 10:
                    if parent1.death != 0:
                        if parent1.death - 1 <= 1:
                            k = 1
                        else:
                            k = random.randint(1, parent1.death)

                    child = Genome(parent1.val[:k])
                    child.val.extend(parent2.val[k:])
                    """print(k)
                    print("copilul: ", child.val)
                    print("parintele: ", parent1.val)
                    print("parintele 2: ", parent2.val)"""
                    if child.val != parent1.val and child.val != parent2.val and self.check(child) is True:
                        not_ok = False
                        self.selection_pool.append(child)
                        self.cross_weak += 2 ** (((i + 1) / self.population_size) * 10 - 10)
                    else:
                        parent2 = random.choice(self.population)

                else:
                    k = random.random()

                    for it, val in enumerate(self.BW):
                        if k < val:
                            if it > 0:
                                a = int(self.BP[it - 1] * parent1.death)
                            else:
                                a = 0

                            b = int(self.BP[it] * parent1.death)

                            # print(it, k, self.BP[it - 1], self.BP[it], parent1.death, a, b)

                            if a == 0 and parent1.death != 0:
                                a += 1
                            if b >= parent1.death:
                                b = parent1.death

                            k = random.randint(a, b)

                            child = Genome(parent1.val[:k])
                            child.val.extend(parent2.val[k:])
                            """print(a, b, k)
                            print("copilul: ", child.val)
                            print("parintele: ", parent1.val)
                            print("parintele 2: ", parent2.val)"""
                            if child.val != parent1.val and child.val != parent2.val and self.check(child) is True:
                                not_ok = False
                                self.selection_pool.append(child)
                                self.cross_weak += 2 ** (((i + 1) / self.population_size) * 10 - 10)
                            else:
                                parent2 = random.choice(self.population)

    def crossover2(self):
        for index, parent1 in enumerate(self.population):
            if random.random() < self.crossover2_rate:
                not_ok = True

                while not_ok is True:
                    subpopulation = self.population[:index]
                    subpopulation.extend(self.population[index + 1:])
                    parent2 = random.choice(subpopulation)

                    k1 = random.randint(1, parent1.death - 1)
                    while k1 >= len(parent1.val) - 1:
                        k1 = random.randint(1, parent1.death - 1)
                    if parent1.death + 1 < len(parent1.val) - 1:
                        k2 = random.randint(parent1.death + 1, len(parent1.val) - 1)

                    else:
                        k2 = random.randint(k1 + 1, len(parent1.val) - 1)

                    child = Genome(parent1.val[:k1])
                    child.val.extend(parent2.val[k1: k2])
                    child.val.extend(parent1.val[k2:])

                    if self.check(child) is True and child.val != parent1.val and child.val != parent2.val:
                        self.selection_pool.append(child)
                        not_ok = False

    def tournament_selection(self, k=3):
        """Selecție prin turneu."""
        pass

    def roulette_selection(self):
        """Selecție prin ruletă."""
        pass

    def elitism_selection(self, num_elites=1):
        """Selecție prin elitism."""
        pass

    def combined_selection(self):
        """Selecție combinată, bazată pe ponderi."""
        total_fitness = 0
        # print(len(self.selection_pool))

        for ind in self.selection_pool:
            ind.lane = self.fitness(ind)
            ind.fitness = ind.lane - ind.death / 200
            #print(ind.val, "Fitness: ", ind.fitness)
            total_fitness += ind.fitness

        total_weight = 0

        for ind in self.selection_pool:
            ind.rate = ind.fitness / total_fitness
            ind.weight = total_weight + ind.rate
            total_weight = ind.weight

        # print(total_weight)

        # Elitism
        self.population[self.parents_elite_number:] = []

        '''print("Self Population", len(self.population))

        for ind in self.population:
            print(ind.val, "cu fitness-ul: ", ind.fitness, " a murit la pozitia: ", ind.death)'''

        self.selection_pool.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population.extend(self.selection_pool[:self.children_elite_number])

        aux = self.selection_pool[self.children_elite_number:]
        self.selection_pool = aux

        '''print("Self Population", len(self.population))

        for ind in self.population:
            print(ind.val, "cu fitness-ul: ", ind.fitness, " a murit la pozitia: ", ind.death)'''

        # print("Am ales prin ELITISM din COPII: ")

        # for ind in range(self.parents_elite_number, self.parents_elite_number+self.children_elite_number):
        #    print(self.population[ind].val, "cu fitness-ul: ", self.population[ind].fitness, " a murit la pozitia: ", self.population[ind].death)

        # self.population is already sorted from strongest to weakest
        # start roulette from chosen elite number

        # Roulette

        # print(len(self.selection_pool))

        # print(self.parents_elite_number + self.children_elite_number, self.population_size - self.tour_number)

        for i in range(self.roulette_number):
            no_new_member = True
            while no_new_member is True:
                k = random.random()
                # print("nr random pt wheel: ", k)
                # self.selection_pool.sort(key=lambda ind: ind.weight)
                for index, ind in enumerate(self.selection_pool):
                    # print("nr wheight al individului: ", ind.weight)
                    if k < ind.weight and ind.selected is False:
                        self.population.append(ind)
                        # print("Am ales prin ROATA NOROCULUI: ")
                        # print(ind.val, "cu fitness-ul: ", ind.fitness, " a murit la pozitia: ", ind.death)
                        ind.selected = True
                        no_new_member = False
                        break

        """counter = 0
        print(f"Selection_pool inainte de turnir: {len(self.selection_pool)}")
        for i, ind in enumerate(self.selection_pool):
            if ind.selected is True:
                counter += 1
        # print("counter: ", counter)"""

        '''print("Self Population Roulette", len(self.population))

        for ind in self.population:
            print(ind.val, "cu fitness-ul: ", ind.fitness, " a murit la pozitia: ", ind.death)'''

        aux_selection = [id for id in self.selection_pool if id.selected is False]
        self.selection_pool = []
        self.selection_pool = aux_selection

        #print(len(self.selection_pool))
        # Tournament

        # print(f"Selection_pool inainte de turnir: {len(self.selection_pool)}")

        for i in range(self.tour_number):
            # print(i)
            k = self.tour_part
            best_fit = -1
            best_ind = -1

            counter = k
            v = []
            while counter > 0:
                t = random.randint(0, len(self.selection_pool) - 1)
                if t in v:
                    continue
                else:
                    v.append(t)
                    counter -= 1

            # print(v, len(v))
            while k:
                # print(k)
                t = v[k - 1]

                # print(f"Individul: {self.selection_pool[t].val} cu fitness-ul {self.selection_pool[t].fitness} "
                #      f"a mers la turnir cu")

                # print(len(self.selection_pool), t)

                if self.selection_pool[t].fitness > best_fit:
                    # print("Hello")
                    best_fit = self.selection_pool[t].fitness
                    best_ind = t

                k -= 1

            # print(best_ind)
            # print(self.selection_pool[best_ind])
            # print(f"Am ales prin TURNIR:")
            # print(self.selection_pool[best_ind].val, "cu fitness-ul: ", self.selection_pool[best_ind].fitness, " a murit la pozitia: ", self.selection_pool[best_ind].death)
            self.population.append(self.selection_pool[best_ind])
            del self.selection_pool[best_ind]

        # print(len(self.population))

    def evolve(self):
        """Rulează algoritmul genetic."""
        for generation in range(self.generations):

            self.crossover1()
            self.crossover2()
            self.mutate()

            '''for ind in self.population:
                print(ind.val, "cu fitness-ul: ", ind.fitness, " a murit la pozitia: ", ind.death)'''

            # print(
            #    f"Generația {generation + 1}: nr_pop: {len(self.population)}, nr sel_pool: {len(self.selection_pool)}")

            self.combined_selection()
            self.sort()

            """for ind in self.population:
                print(f"Individul: {ind.val} are fitnessul: {ind.fitness}")
            # self.calc_fitness()"""
            print(f"Generația {generation + 1}: Cel mai bun fitness: {self.population[0].fitness} Lane: {self.population[0].lane}")
            self.selection_pool = []

            if generation == self.generations - 1:
                print(self.unique_ind())
                for ind in self.population:
                    print(ind.val, "cu fitness-ul: ", ind.fitness, " a murit la pozitia: ", ind.death, "Lane ", ind.lane)

            # Afișare progres

        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        return self.population[0]


def visualize_weights():
    pass


class Genome:
    def __init__(self, val):
        self.val = val
        self.fitness = None
        self.lane = None
        self.death = 199
        self.rate = None
        self.weight = None
        self.selected = False


def play(player):
    frogger_env.reset()
    frogger_env.gameApp.draw()
    frogger_env.gameApp.state = "PLAYING"
    total_reward = 0
    for i, action in enumerate(player.val):
        _, reward, done, _ = frogger_env.step(action)
        frogger_env.gameApp.draw()
        total_reward += reward

        if done:
            print("ALOOOOOO ", player.death, i)
            break

        time.sleep(0.5)

    print(total_reward)


if __name__ == "__main__":
    # Inițializează environment-ul
    frogger_env = FroggerEnv()

    # Creează și rulează algoritmul genetic
    ga = FroggerGA(
        population_size=100,
        generations=100,
        mutation_rate=0.3,
        environment=frogger_env
    )
    # ga.crossover1()
    best_player = ga.evolve()

    play(best_player)
    # IDEE: actualizeaza ponderile in functie de cele mai bune rezultate (nu creste elitismul de la sine inteles)
