import matplotlib.pyplot as plt
from aco import AntColony
import time

plt.style.use("dark_background")

COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (12, 10),
    (33, 89),
    (19, 72)
)


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


plot_nodes()

start = time.time()

colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,
                   pheromone_evaporation_rate=0.40, pheromone_constant=500.0,
                   iterations=300)

end = time.time()

print(f"Running time of the algorithm: {end-start}")

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.show()

# Running time values:
# ant_count=300, alpha=0.5, beta=1.2, pheromone_evaporation_rate=0.40, pheromone_constant=1000.0, iterations=300          14.390064716339111
# ant_count=200, alpha=0.5, beta=1.2, pheromone_evaporation_rate=0.40, pheromone_constant=1000.0, iterations=300          10.019835233688354
# ant_count=300, alpha=0.5, beta=1.2, pheromone_evaporation_rate=0.40, pheromone_constant=1000.0, iterations=200          10.024266004562378
# ant_count=300, alpha=0.5, beta=1.2, pheromone_evaporation_rate=0.40, pheromone_constant=500.0, iterations=300           14.696306228637695
# ant_count=300, alpha=0.5, beta=1.2, pheromone_evaporation_rate=0.10, pheromone_constant=1000.0, iterations=300          14.399922847747803
# ant_count=300, alpha=0.8, beta=1.6, pheromone_evaporation_rate=0.40, pheromone_constant=1000.0, iterations=300          14.425570964813232


# The smaller the ant_count and iterations values, the faster the algorithm finds a solution.



