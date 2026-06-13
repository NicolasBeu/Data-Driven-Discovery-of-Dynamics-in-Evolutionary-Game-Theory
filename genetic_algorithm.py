import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import os

# Genetic algorithm for the Hawk-Dove game, following Laruelle et al. (2018).
# Two death rules are implemented and chosen with the death_rule variable below:
#   "inverse"  low-fitness individuals are more likely to die
#   "uniform"  every individual is equally likely to die
# Everything else (interaction, birth, mutation, fitness normalization) is the same,
# so running both rules isolates the effect of the death rule on the dynamics.

v = 0.3
c = 0.4

RESULTS_ROOT = "Results"
death_rule = "inverse"          # "inverse" or "uniform"


def run_single_simulation(args):
    sim_id, start_val, rule = args

    N    = 1000
    Nd   = int(N * 0.5)     # active individuals per generation (N/2)
    Ne   = int(N * 0.2)     # encounters per active individual (N/5)
    mu   = 0.005            # mutation rate
    Nc   = int(N * 0.05)    # replacements per generation (N/20)
    generations = 2000

    np.random.seed(sim_id + int(start_val * 10000))

    # initial population: a fraction start_val are pure hawks, the rest pure doves
    alphas  = np.zeros(N)
    n_hawks = int(start_val * N)
    alphas[:n_hawks] = 1.0
    np.random.shuffle(alphas)

    # initial fitness w0 = c for everyone, kept outside the loop so it carries over
    fitness = np.full(N, float(c))

    history_avg_alpha = np.zeros(generations)

    for t in range(generations):

        # interaction phase: payoffs accumulate on top of the carried-over fitness
        focal_indices = np.random.choice(N, size=Nd, replace=False)

        F_matrix = np.repeat(focal_indices[:, np.newaxis], Ne, axis=1).flatten()
        O_matrix = np.random.randint(0, N - 1, size=Nd * Ne)
        O_matrix[O_matrix >= F_matrix] += 1

        alpha_F = alphas[F_matrix]
        alpha_O = alphas[O_matrix]

        X_F = np.random.rand(Nd * Ne) < alpha_F
        X_O = np.random.rand(Nd * Ne) < alpha_O

        payoffs = np.zeros(Nd * Ne, dtype=float)
        HH = X_F & X_O
        DD = (~X_F) & (~X_O)
        HD = X_F & (~X_O)

        payoffs[HH] = np.where(np.random.rand(np.sum(HH)) < 0.5, v, -c)
        payoffs[DD] = np.where(np.random.rand(np.sum(DD)) < 0.5, v, 0.0)
        payoffs[HD] = v

        np.add.at(fitness, F_matrix, payoffs)

        # min-max normalization of the whole population to [0, 1]
        f_min = np.min(fitness)
        f_max = np.max(fitness)
        if f_max > f_min:
            fitness = (fitness - f_min) / (f_max - f_min)
        else:
            fitness[:] = 0.5

        # reproduction phase
        if rule == "inverse":
            p_death = 1.0001 - fitness      # low fitness more likely to die
        else:
            p_death = np.ones(N)            # uniform death
        p_death /= p_death.sum()

        p_birth = fitness + 0.0001          # birth proportional to fitness
        p_birth /= p_birth.sum()

        deaths = np.random.choice(N, size=Nc, replace=True, p=p_death)
        births = np.random.choice(N, size=Nc, replace=True, p=p_birth)

        # an individual cannot replace itself
        self_replacements = (deaths == births)
        while np.any(self_replacements):
            births[self_replacements] = np.random.choice(
                N, size=np.sum(self_replacements), p=p_birth)
            self_replacements = (deaths == births)

        # offspring inherits the parent's strategy and fitness
        alphas[deaths]  = alphas[births]
        fitness[deaths] = fitness[births]

        # mutation: new random strategy and fitness reset to w0 = c
        mutations = np.random.rand(Nc) < mu
        if np.any(mutations):
            alphas[deaths[mutations]]  = np.random.rand(np.sum(mutations))
            fitness[deaths[mutations]] = c

        history_avg_alpha[t] = np.mean(alphas)

    return history_avg_alpha


if __name__ == "__main__":
    n_ensembles = 40
    n_workers   = cpu_count()
    start_total = time.time()

    results_dir = os.path.join(RESULTS_ROOT, f"GA_{death_rule}_death")
    os.makedirs(results_dir, exist_ok=True)

    start_values_train = np.round(np.linspace(0.05, 0.95, 10), 3).tolist()
    start_values_test  = np.round(np.linspace(0.03, 0.97, 20), 3).tolist()

    def generate_and_save_master(start_values, prefix):
        master_data_dict = {}
        plot_dict        = {}

        for sv in start_values:
            tasks = [(i, sv, death_rule) for i in range(n_ensembles)]
            with Pool(processes=n_workers) as pool:
                trajectories = pool.map(run_single_simulation, tasks)

            for run_id, traj in enumerate(trajectories):
                master_data_dict[f"start_{sv}_run_{run_id}"] = traj

            plot_dict[f"start_{sv}"] = np.mean(trajectories, axis=0)

        full_path_csv = os.path.join(results_dir, f"ga_master_{prefix}_ens{n_ensembles}.csv")
        pd.DataFrame(master_data_dict).to_csv(full_path_csv, index=False)

        df_plot = pd.DataFrame(plot_dict)
        plt.figure(figsize=(10, 6))
        for col in df_plot.columns:
            plt.plot(df_plot[col], label=col)
        plt.axhline(y=v/c, color='black', linestyle='--', label=f'ESS={v/c:.3f}')
        plt.title(f'GA N=1000 {prefix} (n={n_ensembles}) - {death_rule} death')
        plt.ylabel(r'Mean $\alpha$')
        plt.xlabel('Generation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(full_path_csv.replace('.csv', '.png'), dpi=300)
        plt.close()

        print(f"  Saved: {full_path_csv}")

    generate_and_save_master(start_values_train, "train")
    generate_and_save_master(start_values_test,  "test")

    print(f"\nDone. Total time: {time.time() - start_total:.2f}s")
