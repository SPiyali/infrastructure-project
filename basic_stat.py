import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

# ==============================
# CONNECTION
# ==============================
db_url = "postgresql+psycopg2://data_friend:301274dataAccess@ep-shiny-boat-amyktl3t-pooler.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(db_url)

# ==============================
# QUERY
# ==============================
query = """
SELECT 
    sqldate,
    eventcode,
    goldsteinscale,
    avgtone,
    nummentions,
    actor1name,
    actor2name,
    actor1countrycode,
    actor2countrycode,
    actiongeo_countrycode
FROM gdelt_events
WHERE sqldate >= 20240101
"""

df = pd.read_sql(query, engine)

# ==============================
# CLEANING + MAPPING
# ==============================
df.columns = df.columns.str.lower()

df['sqldate'] = pd.to_datetime(df['sqldate'], format='%Y%m%d', errors='coerce')

df = df.dropna(subset=['avgtone', 'goldsteinscale'])

df["actor1"]     = df["actor1name"]
df["actor2"]     = df["actor2name"]
df["event_type"] = df["eventcode"].astype(str)
df["tone"]       = df["avgtone"]
df["mentions"]   = df["nummentions"]
df["country1"]   = df["actor1countrycode"]
df["country2"]   = df["actor2countrycode"]
df["country"]    = df["actiongeo_countrycode"]

df = df.dropna(subset=["actor1", "actor2"])

# ==============================
# ANALYZER CLASS
# ==============================
class GDELT15MinAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}

    def event_distribution(self):
        p = self.df["event_type"].value_counts(normalize=True)
        H = entropy(p)
        self.results["event_distribution"] = p
        self.results["event_entropy"] = H
        return p, H

    def actor_gini(self):
        actors = pd.concat([self.df["actor1"], self.df["actor2"]])
        counts = actors.value_counts().values

        def gini(x):
            x = np.sort(x)
            n = len(x)
            return (2 * np.sum((np.arange(1, n+1) * x)) / (n * np.sum(x))) - (n+1)/n

        g = gini(counts)
        self.results["actor_gini"] = g
        return g

    def build_graph(self):
        G = nx.Graph()
        for _, row in self.df.iterrows():
            G.add_edge(row["actor1"], row["actor2"])
        self.G = G
        return G

    def graph_stats(self):
        G = self.G
        stats = {
            "nodes":      G.number_of_nodes(),
            "edges":      G.number_of_edges(),
            "density":    nx.density(G),
            "components": nx.number_connected_components(G)
        }
        self.results["graph_stats"] = stats
        return stats

    def centrality(self, top_k=10):
        centrality = nx.degree_centrality(self.G)
        top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        self.results["centrality"] = top
        return top

    def country_interaction(self):
        mat = pd.crosstab(self.df["country1"], self.df["country2"])
        self.results["country_matrix"] = mat
        return mat

    def tone_stats(self):
        mean    = self.df["tone"].mean()
        std     = self.df["tone"].std()
        by_event = self.df.groupby("event_type")["tone"].mean()
        self.results["tone"] = {
            "mean":     mean,
            "std":      std,
            "by_event": by_event
        }
        return self.results["tone"]

    def extreme_events(self, q=0.95):
        threshold = self.df["mentions"].quantile(q)
        extreme   = self.df[self.df["mentions"] > threshold]
        self.results["extreme_events"] = extreme
        return extreme

    def actor_diversity(self):
        diversity = self.df.groupby("actor1")["actor2"].nunique()
        self.results["actor_diversity"] = diversity
        return diversity

    def joint_entropy(self):
        joint = pd.crosstab(self.df["actor1"], self.df["event_type"], normalize=True)
        H = entropy(joint.values.flatten())
        self.results["joint_entropy"] = H
        return H

    def cluster_actors(self, k=3):
        features = pd.crosstab(self.df["actor1"], self.df["event_type"])
        if len(features) < k:
            k = max(1, len(features))
        model  = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(features)
        features["cluster"] = labels
        self.results["actor_clusters"] = features
        return features

    def core_periphery(self):
        degrees = dict(self.G.degree())
        if len(degrees) == 0:
            return [], []
        threshold = np.percentile(list(degrees.values()), 80)
        core      = [n for n, d in degrees.items() if d >= threshold]
        periphery = [n for n, d in degrees.items() if d < threshold]
        self.results["core"]      = core
        self.results["periphery"] = periphery
        return core, periphery

    def run_all(self):
        self.event_distribution()
        self.actor_gini()
        self.build_graph()
        self.graph_stats()
        self.centrality()
        self.country_interaction()
        self.tone_stats()
        self.extreme_events()
        self.actor_diversity()
        self.joint_entropy()
        self.cluster_actors()
        self.core_periphery()
        return self.results

# ==============================
# RUN
# ==============================
analyzer = GDELT15MinAnalyzer(df)
results  = analyzer.run_all()

print("\nEvent Entropy:",  results["event_entropy"])
print("\nGraph Stats:",    results["graph_stats"])
print("\nTop Actors:",     results["centrality"][:5])
