"""Genetic Algorithm — evolutionary optimization with selection and crossover."""

import numpy as np

def genetic_algorithm(fitness_fn, n_genes, pop_size=50, n_generations=100,
                      mutation_rate=0.1, crossover_rate=0.8, gene_range=(-5, 5)):
    # initialize population
    pop = np.random.uniform(gene_range[0], gene_range[1], (pop_size, n_genes))
    best_fitness_history = []

    for gen in range(n_generations):
        fitness = np.array([fitness_fn(ind) for ind in pop])
        best_idx = fitness.argmax()
        best_fitness_history.append(fitness[best_idx])

        # tournament selection
        new_pop = [pop[best_idx].copy()]  # elitism
        while len(new_pop) < pop_size:
            # tournament of 3
            contestants = np.random.choice(pop_size, 3, replace=False)
            winner = contestants[fitness[contestants].argmax()]
            new_pop.append(pop[winner].copy())

        pop = np.array(new_pop)

        # crossover
        for i in range(1, pop_size - 1, 2):
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, n_genes)
                pop[i, point:], pop[i+1, point:] = pop[i+1, point:].copy(), pop[i, point:].copy()

        # mutation
        for i in range(1, pop_size):  # skip elite
            mask = np.random.rand(n_genes) < mutation_rate
            pop[i, mask] += np.random.randn(mask.sum()) * 0.5

        pop = np.clip(pop, gene_range[0], gene_range[1])

    fitness = np.array([fitness_fn(ind) for ind in pop])
    best = pop[fitness.argmax()]
    return best, fitness.max(), best_fitness_history

# --- demo ---
np.random.seed(42)

# maximize negative Rastrigin (minimize Rastrigin)
def neg_rastrigin(x):
    return -(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

print("=== Genetic Algorithm ===")
print(f"Minimizing Rastrigin (2D), global min at (0,0) = 0\n")

best, best_fit, history = genetic_algorithm(neg_rastrigin, n_genes=2, pop_size=80, n_generations=200)
print(f"Best solution: ({best[0]:.4f}, {best[1]:.4f})")
print(f"Best value: {-best_fit:.4f}")
print(f"\nConvergence:")
for g in [0, 10, 25, 50, 100, 199]:
    print(f"  Gen {g:>3}: f = {-history[g]:.4f}")
