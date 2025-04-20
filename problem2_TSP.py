import numpy as np
import random
from typing import List, Tuple

class GeneticAlgorithmTSP:
    def __init__(self, weight_matrix: np.ndarray, num_cities: int = 15, 
                 population_size: int = 15, max_iterations: int = 20):
        """
        Initialize the Genetic Algorithm for TSP.
        Args:
            weight_matrix: Distance matrix between cities (n x n)
            num_cities: Number of cities
            population_size: Number of chromosomes in population
            max_iterations: Maximum number of iterations to run
        """
        self.weight_matrix = weight_matrix
        self.num_cities = num_cities
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.population = []
        
        # Validate the weight matrix
        if weight_matrix.shape != (num_cities, num_cities):
            raise ValueError("Weight matrix must be n x n where n is num_cities")
            
    def initialize_population(self):
        """Initialize population with random permutations of cities."""
        base = list(range(self.num_cities))
        self.population = [random.sample(base, self.num_cities) 
                          for _ in range(self.population_size)]
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """Calculate the total distance for a given route (chromosome)."""
        total_distance = 0
        for i in range(self.num_cities - 1):
            from_city = chromosome[i]
            to_city = chromosome[i+1]
            total_distance += self.weight_matrix[from_city][to_city]

        # Add distance back to starting city
        total_distance += self.weight_matrix[chromosome[-1]][chromosome[0]]
        return 1 / total_distance  # We want to maximize fitness (minimize distance)
    
    def roulette_wheel_selection(self) -> List[List[int]]:
        """Select parents using roulette wheel selection."""
        fitness_values = [self.calculate_fitness(chrom) for chrom in self.population]
        total_fitness = sum(fitness_values)
        probabilities = [f/total_fitness for f in fitness_values]
        
        # Select two parents based on probabilities
        parents = random.choices(
            self.population, 
            weights=probabilities, 
            k=2
        )
        return parents
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform one-point crossover between two parents."""

        # Select random crossover point
        point = random.randint(1, self.num_cities - 1)
        
        # Create offspring
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Perform Inorder swap mutation (swap two cities)."""

        # Select two distinct random indices
        idx1, idx2 = random.sample(range(self.num_cities), 2)
        
        # Swap the cities at these indices
        mutated = chromosome.copy()
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def rectify_chromosome(self, chromosome: List[int]) -> List[int]:
        """Ensure each city appears exactly once in the chromosome."""
        missing = set(range(self.num_cities)) - set(chromosome)
        duplicates = []
        seen = set()
        
        # Find duplicates
        for i, city in enumerate(chromosome):
            if city in seen:
                # append index of duplicate city
                duplicates.append(i)
            seen.add(city)
        
        # Replace duplicates with missing cities
        for dup_idx, missing_city in zip(duplicates, missing):
            chromosome[dup_idx] = missing_city
            
        return chromosome
    
    def run(self) -> List[int]:
        """Run the genetic algorithm and return the best solution."""

        # Initialize population
        self.initialize_population()
        best_solution = None
        best_fitness = -np.inf
        
        for iteration in range(self.max_iterations):
            new_population = []
            
            # Elitism: keep track of best solution from previous generation
            current_fitness = [self.calculate_fitness(chrom) for chrom in self.population]
            current_best_idx = np.argmax(current_fitness)
            current_best = self.population[current_best_idx]
            current_best_fitness = current_fitness[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best.copy()
            
            # Generate new population
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.roulette_wheel_selection()
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                # Rectification
                offspring1 = self.rectify_chromosome(offspring1)
                offspring2 = self.rectify_chromosome(offspring2)
                
                # Add to new population
                new_population.extend([offspring1, offspring2])
            
            # Ensure population size remains constant
            self.population = new_population[:self.population_size]
            
            # Elitism: replace worst solution with previous best
            if iteration > 0:
                worst_idx = np.argmin([self.calculate_fitness(chrom) for chrom in self.population])
                self.population[worst_idx] = best_solution.copy()
            
            # Print progress
            total_distance = 1 / best_fitness
            print(f"Iteration {iteration + 1}: Best Distance = {total_distance:.2f}")
        
        return best_solution

def get_user_matrix():
    print("Enter your 15x15 matrix row by row (each row separated by newline):")
    print("Example for row 1: 0 10 15 20 ... (15 numbers separated by spaces)")
    matrix = []
    for i in range(15):
        while True:
            row_input = input(f"Row {i+1}: ")
            try:
                row = list(map(int, row_input.split()))
                if len(row) != 15:
                    print("Error: Each row must have exactly 15 numbers")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Error: Please enter only numbers separated by spaces")
    return np.array(matrix)

if __name__ == "__main__":
    num_cities = 15
    while True:
        choice = input("Do you want to use a random weight matrix? (yes/no): ").lower()
        if choice in ['yes', 'y']:
            # Create a random distance matrix for 15 cities
            np.random.seed(42)
            num_cities = 15
            weight_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
            # Make the matrix symmetric (distance from A to B = distance from B to A)
            weight_matrix = (weight_matrix + weight_matrix.T) // 2
            # Set diagonal to 0 (distance from a city to itself is 0)
            np.fill_diagonal(weight_matrix, 0)
            print("\nGenerated random weight matrix:")
            print(weight_matrix)
            break
        elif choice in ['no', 'n']:
            weight_matrix = get_user_matrix()
            break
        else:
            print("Please enter 'yes' or 'no'")
    
    # Run the genetic algorithm
    ga_tsp = GeneticAlgorithmTSP(weight_matrix)
    best_route = ga_tsp.run()
    
    print("\nBest route found:", best_route)
    total_distance = sum(weight_matrix[best_route[i]][best_route[i+1]] for i in range(num_cities - 1))
    total_distance += weight_matrix[best_route[-1]][best_route[0]]
    print("Total distance:", total_distance)