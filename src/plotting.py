import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def assign_colors(sdata, column_key, seed=42):
    """
    Assign unique colors to each category in a given column. 
    """
    rng = np.random.default_rng(seed)

    colors = []
    seen = set()
    n_colors = sdata.tables["table"].obs[column_key].nunique()

    while len(colors) < n_colors:
        # Sample continuous HSV
        h = rng.random()
        s = rng.uniform(0.6, 1.0)  # avoid washed-out colors
        v = rng.uniform(0.7, 1.0)

        rgb = mcolors.hsv_to_rgb((h, s, v))
        hex_color = mcolors.to_hex(rgb, keep_alpha=False)

        if hex_color not in seen:
            seen.add(hex_color)
            colors.append(hex_color)

    sdata.tables["table"].uns[f"{column_key}_colors"] = colors
    
import matplotlib.pyplot as plt

def plot_scores(data):
    scores = ["bic", "aic", "icl"]
    
    score_colors = {
        "bic": "lightblue",
        "aic": "orange",
        "icl": "black",
    }

    keys = sorted(list(data.keys()))

    fig, ax = plt.subplots()

    for score in scores:
        y = [data[k][score] for k in keys]
        
        ax.plot(
            keys,
            y,
            marker="o",
            color=score_colors[score],
            label=score,
        )

        min_score_val = min(y)              
        min_index = y.index(min_score_val)  
        best_k = keys[min_index]            
        
        # add vertical indicator of lowest
        ax.axvline(
            x=best_k, 
            color=score_colors[score], 
            linestyle=":", 
            linewidth=1.5,
            alpha=0.7 
        )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Score")
    ax.set_title("Scores by number of components")
    ax.legend()

    plt.tight_layout()
    plt.show()
