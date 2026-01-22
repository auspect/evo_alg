import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/auspect/evo_alg/blob/master/my_notebook.py)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction aux algorithmes évolutionnaires

    On va regarder comment optimiser une fonction objectif avec des algorithmes d'optimisation globaux (et plus ou moins agnostiques) :

    * Algorithme Génétique
    * Evolution différentielle

    **Remarque :** Si la fonction objectif est différentiable, il vaut mieux utiliser des techniques basées sur le gradient (voire même la hessienne), car bien plus efficace en général.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Packages
    """)
    return


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import kendalltau, norm
    from dataclasses import dataclass
    from typing import TypeAlias, Callable

    # Thème clair pour les graphiques Altair
    alt.themes.enable("googlecharts")

    return Callable, TypeAlias, alt, dataclass, kendalltau, mo, norm, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paramètres
    """)
    return


@app.cell
def _():
    N_SAMPLES = 100
    N_NOTES = 10
    SEED = 42

    return N_NOTES, N_SAMPLES, SEED


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simuler des données
    """)
    return


@app.cell
def _(N_NOTES, N_SAMPLES, SEED, norm, np, pd):
    _np_rng = np.random.default_rng(SEED)

    # Simuler les variables explicatives
    X = _np_rng.normal(size=(N_SAMPLES, 2))
    # Simuler un score pour générer une note
    # (pour que la note soit corrélée aux variables explicatives)
    # (uniquement pour la simulation de données, n'existe pas sinon)
    scores = X[:, 0] * 0.3 + X[:, 1] * 0.7 + _np_rng.normal(0, 0.3, N_SAMPLES)
    # Simuler les notes (de 1 à N_NOTES)
    quantiles_gauss = norm.cdf(np.linspace(-2.5, 2.5, N_NOTES + 1))
    quantiles_gauss[0] = 0.0
    quantiles_gauss[-1] = 1.0
    print(f"Supposons que les notes sont réparties selon ces quantiles gaussiens : {quantiles_gauss}")
    y = pd.qcut(scores, q=quantiles_gauss, labels=False) + 1

    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["y"] = y
    df

    return df, scores, y


@app.cell
def _(alt, df):
    # replace df with your data source
    _chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X(field="x0", type="quantitative"),
            y=alt.Y(field="x1", type="quantitative"),
            color=alt.Color(field="y").scale(scheme="pinkyellowgreen", reverse=True, type="ordinal"),
            tooltip=[
                alt.Tooltip(field="x0", format=",.2f"),
                alt.Tooltip(field="x1", format=",.2f"),
                alt.Tooltip(field="y", format=",.0f")
            ]
        )
        .properties(
            title="Scatterplot",
            height=290,
            width="container",
            config={
                "axis": {
                    "grid": True
                }
            }
        )
    )
    _chart

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modélisation

    On a donc :

    * deux variables explicatives continues $x_0$ et $x_1$
    * une variable cible ordinale $y$
    * pas de valeurs manquantes pour pas faire un tutoriel trop lourd

    En risque de crédit, on aime bien discrétiser les variables continues notamment pour introduire de la non-linéarité, rendre plus stables et lisibles les résultats, etc... On pourra même faire une grille de score lisible de cette manière.

    /// details | Remarque

    On peut tout à fait s'en passer, si on n'a pas ces "contraintes"-là et donc travailler directement avec les variables explicatives, transformées ou non...

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Préparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quelques fonctions utiles :

    * Pour calculer les prédictions
    * Pour calculer la mesure qui nous intéresse (ici le Tau-b de Kendall qui nous permet d'évaluer si le ranking du score continu et de la cible sont en accord ou non)
    """)
    return


@app.cell
def _(kendalltau, np):
    def compute_raw_pred(w: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Calculer la combinaison linéaire des variables avec les coefficients w.
        C'est un score brute qui a vocation à être discrétisé ensuite en notes
        """
        return X @ w

    def objective(w: np.ndarray, X: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        """
        Fonction objectif à minimiser : kendall tau-b entre les notes vraies
        et les scores bruts (continus)
        """
        y_raw_pred = compute_raw_pred(w, X)
        tau, _ = kendalltau(y_true, y_raw_pred, variant="b", **kwargs)
        return tau  # tau-b à maximiser

    return (objective,)


@app.cell
def _(kendalltau, scores, y):
    # Juste pour tester, quel est le tau-b du vrai score vs les notes issues de ce score
    # Rappel : les notes ont été générées à partir du score simulé
    # (on est pas aussi chanceux dans la vraie vie 🙂)
    _tau_b, _ = kendalltau(scores, y, variant="b")
    print(f"Tau-b entre score simulé et notes issues du score simulé : {_tau_b:.2%}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Discrétisation des variables

    On fait simple, juste une discrétisation uniforme ici (et parce que dans cet exemple on veut utiliser des données discrétisées), mais il y a des techniques bien plus pertinentes pour discrétiser (arbres de régression...).

    On choisit arbitrairement 4 modalités par variables.
    """)
    return


@app.cell
def _(df, pd):
    features_discretized = pd.DataFrame({
        "discrete_x0": pd.qcut(df["x0"], q=4, labels=["a", "b", "c", "d"]),
        "discrete_x1": pd.qcut(df["x1"], q=4, labels=["e", "f", "g", "h"]),
    })
    features_discretized

    return (features_discretized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dummies

    Mais comme on est ici en vision discrète, on traite chaque modalité comme une variable.

    En bref, on passe de :

    | discrete_x0   | discrete_x1   |
    |:--------------|:--------------|
    | c             | e             |
    | d             | h             |
    | a             | e             |
    | c             | f             |
    | b             | e             |

    à

    |   discrete_x0_b |   discrete_x0_c |   discrete_x0_d |   discrete_x1_f |   discrete_x1_g |   discrete_x1_h |
    |----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|
    |               0 |               1 |               0 |               0 |               0 |               0 |
    |               0 |               0 |               1 |               0 |               0 |               1 |
    |               0 |               0 |               0 |               0 |               0 |               0 |
    |               0 |               1 |               0 |               1 |               0 |               0 |
    |               1 |               0 |               0 |               0 |               0 |               0 |

    /// details | Multi-colinéarité
        type: info

    A noter qu'ici, on fait `drop_first=True` pour retirer 1 modalité par variable lors de la construction des dummies.

    En réalité ce n'est pas nécessaire dans le cas présent, puisque notre algorithme peut converger avec ou sans, et dans tous les cas, on pourra construire la grille de score. Mais comme c'est souvent requis (dès lors que la multi-colinéarité empêche d'inverser une matrice par ex), on le fait quand même, et ça réduit même le nombre de coefficients à trouver (les coefficients retirés seront donc forcés à 0).
    ///
    """)
    return


@app.cell
def _(features_discretized, pd):
    features_dummies = 1 * pd.get_dummies(
        features_discretized.astype(str),
        drop_first=True
    )

    features_dummies

    return (features_dummies,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Algos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Bornes

    On doit fixer des bornes pour les coefficients que l'on cherche à obtenir.

    Dans notre cas, l'étendue des coefficients importe peu, puisque la magnitude du score n'impacte pas le ranking.
    Cela dit, il est souvent préférable d'éviter des valeurs trop petites (pour des questions de stabilité numérique), donc on peut raisonnablement choisir des coefficients allant de -5 à +5 (mais n'importe quelles borne feraient l'affaire).
    """)
    return


@app.cell
def _():
    BORNE_INITIALE = -5.0
    BORNE_FINALE = 5.0
    N_GENERATIONS = 50
    POPULATION_SIZE = 100

    return BORNE_FINALE, BORNE_INITIALE, N_GENERATIONS, POPULATION_SIZE


@app.cell
def _(BORNE_FINALE, BORNE_INITIALE, TypeAlias, dataclass, features_dummies):
    # Bornes pour les coefficients
    bound_type: TypeAlias = tuple[float, float]
    bounds_type: TypeAlias = list[bound_type]
    infs_type: TypeAlias = list[float]
    sups_type: TypeAlias = list[float]

    @dataclass
    class Bound:
        low: float
        high: float

        def as_tuple(self) -> tuple[float, float]:
            return (self.low, self.high)

    @dataclass
    class Bounds:
        bounds: list[Bound]
        _cache_as_list_of_tuples: bounds_type = None
        _cache_as_list_inf_and_sup: tuple[infs_type, sups_type] = None


        def as_list_of_tuples(self) -> bounds_type:
            """Convertir les bornes en liste de tuples"""
            if self._cache_as_list_of_tuples is None:
                self._cache_as_list_of_tuples = [b.as_tuple() for b in self.bounds]
            return self._cache_as_list_of_tuples

        def as_list_inf_and_sup(self) -> tuple[infs_type, sups_type]:
            """Convertir les bornes en deux listes : inf et sup (pour np.random.uniform, clip, etc.)"""
            if self._cache_as_list_inf_and_sup is None:
                lows = [b.low for b in self.bounds]
                highs = [b.high for b in self.bounds]
                self._cache_as_list_inf_and_sup = (lows, highs)
            return self._cache_as_list_inf_and_sup
        def __len__(self) -> int:
            return len(self.bounds)

    coeffs_bounds = Bounds([
        Bound(BORNE_INITIALE, BORNE_FINALE) for _ in range(features_dummies.shape[1])
    ])
    coeffs_bounds

    return Bounds, coeffs_bounds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Stats sur la perf de l'algo

    On crée un petit utilitaire pour le suivi de la perf des algos
    """)
    return


@app.cell
def _(alt, dataclass, pd):
    @dataclass
    class PerformanceAlgoTracker:
        """Suivi de la performance d'un algo évolutionnaire"""

        best_fitness: float = float("-inf")
        avg_fitness_by_gen: list[float] = None
        best_fitness_by_gen: list[float] = None
        title: str = "Performance de l'algorithme évolutionnaire au fil des générations"

        def __post_init__(self):
            self.avg_fitness_by_gen = []
            self.best_fitness_by_gen = []

        def update_best_fitness(self, best_fitness: float):
            """Mettre à jour la meilleure fitness"""
            if self.best_fitness is None or best_fitness > self.best_fitness:
                self.best_fitness = best_fitness

        def update_by_generation(self, current_fitnesses: list[float]):
            """Mettre à jour les performances par génération"""
            avg_fitness = sum(current_fitnesses) / len(current_fitnesses)
            self.avg_fitness_by_gen.append(avg_fitness)
            self.best_fitness_by_gen.append(max(current_fitnesses))

        def update(self, best_fitness, current_fitnesses):
            """Mettre à jour les performances"""
            self.update_best_fitness(best_fitness)
            self.update_by_generation(current_fitnesses)

        def plot(self):
            """Tracer l'évolution des performances"""
            df_perf = pd.DataFrame({
                "Génération": list(range(1, len(self.avg_fitness_by_gen) + 1)),
                "Fitness moyen": self.avg_fitness_by_gen,
                "Meilleure fitness": self.best_fitness_by_gen
            })
            chart = (
                alt.Chart(df_perf)
                .transform_fold(
                    fold=["Fitness moyen", "Meilleure fitness"],
                    as_=["Metric", "Fitness"]
                )
                .mark_line(point=True)
                .encode(
                    x="Génération:Q",
                    y=alt.Y("Fitness:Q").scale(domain=(-1, 1)),
                    color="Metric:N"
                )
                .properties(
                    title=self.title,
                    width=600,
                    height=400,
                )
            )
            return chart

    return (PerformanceAlgoTracker,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Individus

    Un individu est un objet simple, qui représente une combinaison de coefficients ici. On stocke également la valeur de la performance qui lui est associé.

    Ces coefficients sont appris progressivement au fil de l'évolution (au départ, ils sont 100% aléatoires).
    """)
    return


@app.cell
def _(Bounds, np):
    class Individual:
        """Représentation d'un individu dans un algo évolutionnaire"""

        __slots__ = ("w", "fitness")

        def __init__(self, w: np.ndarray):
            self.w = w
            self.fitness = None

        @classmethod
        def random(cls, n_coeffs: int, bounds: Bounds, rng: np.random.Generator) -> "Individual":
            """Créer un individu avec des coefficients aléatoires"""
            lows, highs = bounds.as_list_inf_and_sup()
            w = rng.uniform(lows, highs, size=n_coeffs)
            return cls(w)
        def clip_individual(self, bounds: Bounds):
            """Clipper les coefficients de l'individu dans les bornes"""
            lows, highs = bounds.as_list_inf_and_sup()
            self.w = np.clip(self.w, lows, highs)
        def evaluate(self, fitness_func, X: np.ndarray, y: np.ndarray):
            """Évaluer la fitness de l'individu"""
            if self.fitness is None:
                self.fitness = fitness_func(self.w, X, y)
            return self.fitness

    # Population type alias (liste d'individus)
    Population = list[Individual]

    return (Individual,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Squelette des algos

    On sépare la configuration (paramètres propres à l'algo utilisé) de l'algo en lui-même, pour pouvoir s'y retrouver plus facilement (c'est purement optionnel).
    """)
    return


@app.cell
def _(Bounds, Callable, Individual, PerformanceAlgoTracker, dataclass, mo, np):
    # La config
    @dataclass
    class EvolutionaryConfigTemplate:
        """Configuration pour les algos évolutionnaires"""
        population_size: int
        n_generations: int
        seed: int = 42

    # Le squelette de l'algo (ils ont tous la même structure de base, seule l'évolution diffère)
    class EvolutionaryTemplate:
        """Template pour les algos évolutionnaires"""

        def __init__(self, objective: Callable, config: EvolutionaryConfigTemplate, bounds: Bounds):
            """Initialiser l'algorithme évolutionnaire"""
            self.objective = objective
            self.config = config
            self.rng = np.random.default_rng(config.seed)
            self.bounds = bounds
            self.n_coeffs = len(self.bounds)
            self.population = self.init_pop()
            self.track_perf = PerformanceAlgoTracker(
                title=f"Performance de l'algorithme {self.__class__.__name__} au fil des générations"
            )

        def init_pop(self) -> list[Individual]:
            """Initialiser la population"""
            population = [
                Individual.random(n_coeffs=len(self.bounds), bounds=self.bounds, rng=self.rng)
                for _ in range(self.config.population_size)
            ]
            return population

        def evaluate_population(self, X: np.ndarray, y: np.ndarray):
            """Évaluer toute la population et conserver l'historique des performances"""
            current_fitnesses = []
            best_fitness = float("-inf")
            for individual in self.population:
                indiv_fit = individual.evaluate(self.objective, X, y)
                current_fitnesses.append(indiv_fit)
                if indiv_fit > best_fitness:
                    best_fitness = indiv_fit
            self.track_perf.update(best_fitness, current_fitnesses)

        def run_iter(self, X: np.ndarray, y: np.ndarray):
            """Une itération (itération=génération) de l'algorithme évolutionnaire"""
            raise NotImplementedError("Cette méthode doit être implémentée dans les sous-classes")

        def run(self, X: np.ndarray, y: np.ndarray, **kwargs):
            """Lancer l'algorithme évolutionnaire"""
            with mo.status.progress_bar(
                total=self.config.n_generations,
                subtitle="Algorithme en cours d'exécution",
                show_eta=True,
                show_rate=True,
                **kwargs,
            ) as pb:
                for gen in range(self.config.n_generations):
                    self.evaluate_population(X, y)
                    pb.update(subtitle=f"Meilleure fitness pour l'instant : {self.track_perf.best_fitness:.2%}")
                    if gen < self.config.n_generations - 1:
                        self.run_iter(X, y)

    return EvolutionaryConfigTemplate, EvolutionaryTemplate


@app.cell
def _(
    EvolutionaryConfigTemplate,
    EvolutionaryTemplate,
    N_GENERATIONS,
    POPULATION_SIZE,
    SEED,
    coeffs_bounds,
    features_dummies,
    np,
    objective,
    y,
):
    class BaselineEvolutionaryAlgo(EvolutionaryTemplate):
        """Algorithme de base (sans évolution, juste des individus aléatoires...)"""

        def run_iter(self, X: np.ndarray, y: np.ndarray):
            """Lancer l'algorithme baseline pour une itération"""
            # On remplace la population par une nouvelle population aléatoire
            self.population = self.init_pop()
    # Exemple d'utilisation
    baseline_config = EvolutionaryConfigTemplate(
        population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS,
        seed=SEED
    )
    baseline_algo = BaselineEvolutionaryAlgo(
        objective=objective,
        config=baseline_config,
        bounds=coeffs_bounds
    )
    baseline_algo.run(features_dummies.values, y)

    return (baseline_algo,)


@app.cell
def _(baseline_algo):
    baseline_algo.track_perf.plot()

    return


@app.cell
def _(
    Bounds,
    EvolutionaryConfigTemplate,
    EvolutionaryTemplate,
    Individual,
    N_GENERATIONS,
    POPULATION_SIZE,
    SEED,
    coeffs_bounds,
    dataclass,
    features_dummies,
    np,
    objective,
    y,
):
    @dataclass
    class GeneticAlgorithmConfig(EvolutionaryConfigTemplate):
        """Configuration spécifique pour l'algorithme génétique"""
        mutation_rate: float = 0.1
        tournament_size: int = 3
        elites_size: int = 2

    class GeneticAlgorithm(EvolutionaryTemplate):
        """Implémentation simple d'un algorithme génétique"""

        def __init__(self, config: GeneticAlgorithmConfig, bounds: Bounds):
            super().__init__(objective=objective, config=config, bounds=bounds)

        def tournament_selection(self) -> Individual:
            """Sélection par tournoi"""
            participants_idx = self.rng.integers(0, len(self.population), size=self.config.tournament_size)
            best_participant_idx = min(participants_idx) # la pop est déjà triée à ce stade (par fitness décroissante)
            return self.population[best_participant_idx]

        def create_child(self, parent1: Individual, parent2: Individual) -> Individual:
            """Créer un enfant par crossover et mutation"""
            # Crossover (ici moyenne des coefficients, mais
            # plein de possibilités existent)
            child_w = (parent1.w + parent2.w) / 2
            # Mutation
            mask = self.rng.binomial(1, self.config.mutation_rate, size=self.n_coeffs)
            mutation_value = self.rng.uniform(*self.bounds.as_list_inf_and_sup(), size=self.n_coeffs)
            applied_mutation = mask * mutation_value
            child_w += applied_mutation
            # S'assurer que le coefficient reste dans les bornes
            child = Individual(child_w)
            child.clip_individual(self.bounds)
            return child

        def run_iter(self, X: np.ndarray, y: np.ndarray):
            """Lancer une itération de l'algorithme génétique"""
            # 1re étape : évaluer la population
            self.evaluate_population(X, y)
            # 2e étape : création de la nouvelle population
            new_population = []
            # Sélection des élites
            sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
            new_population.extend(sorted_population[:self.config.elites_size])
            # Remplir le reste de la population
            while len(new_population) < self.config.population_size:
                # Sélection par tournoi
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                # Création de l'enfant
                child = self.create_child(parent1, parent2)
                new_population.append(child)
            self.population = new_population

    genetic_config = GeneticAlgorithmConfig(
        population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS,
        seed=SEED,
        mutation_rate=0.1,
        tournament_size=3,
        elites_size=2
    )
    genetic_algo = GeneticAlgorithm(
        config=genetic_config,
        bounds=coeffs_bounds
    )
    genetic_algo.run(features_dummies.values, y)

    return (genetic_algo,)


@app.cell
def _(genetic_algo):
    genetic_algo.track_perf.plot()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
