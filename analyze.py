import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import glob

def load_data(path="data/"):
    all_dfs = []

    # find number of files in data
    num_files = len(glob.glob(path + "*.csv"))
    print(f"Found {num_files} files in {path}")

    # for each *.csv, add to df
    for i, file_path in enumerate(glob.glob(path + "*.csv")):
        print(i, end="\r")
        df = pd.read_csv(file_path, header=0)
        all_dfs.append(df)

    df = pd.concat(all_dfs, axis=0, ignore_index=True)
    return all_dfs, df

def gen_features(all_dfs):
    '''
    We know that the first 10 s --> 100 timesteps are not actually controlling the car
    For each df (each rollout) calc:
    - avg vEgo
    - stdev vEgo
    - avg abs(aEgo)
    - avg roll
    - avg target_lataccel (key is actually targetLateralAcceleration lol)
    '''
    features = np.zeros((len(all_dfs), 5))
    for i, df in enumerate(all_dfs):
        # use only first 100 timesteps
        df = df.iloc[:100]
        features[i, :] = [df['vEgo'].mean(), 
                          df['vEgo'].std(), 
                          df['aEgo'].abs().mean(), 
                          df['roll'].mean(), 
                          df['targetLateralAcceleration'].mean()
                          ]
    return features

def run_gmm(features):
    gmm = GaussianMixture(n_components=3, random_state=42, verbose=2)
    gmm.fit(features)
    labels = gmm.predict(features)

    # how many points in each cluster?
    for i in range(gmm.n_components):
        print(f"Cluster {i} has {np.sum(labels == i)} points")
    return gmm, labels

def viz_gmm(gmm, n=100, path="tmp/"):
    # sample n points from gmm
    generated_features, generated_labels = gmm.sample(n) # Unpack the tuple
    
    # plot 3D scatter of generated samples
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # use generated_features as a pretend set of features
    scatter_gmm = ax.scatter(generated_features[:, 0], 
                             generated_features[:, 1], 
                             generated_features[:, 2], 
                             c=generated_labels,        # Color by component label
                             s=generated_features[:, 1] * 10, # Size by the second feature 
                             cmap='viridis')
    
    ax.set_xlabel('avg vEgo')
    ax.set_ylabel('stdev vEgo')
    ax.set_zlabel('avg abs aEgo')
    fig.suptitle(f'{n} Samples from GMM', fontsize=10)
    
    # Add colorbar for the component labels
    # Ensure ticks match the number of components in the GMM
    cbar_gmm = fig.colorbar(scatter_gmm, ticks=np.arange(gmm.n_components))
    cbar_gmm.set_label('GMM component label')
    
    plt.savefig(path + "gmm_samples.png", dpi=384)

if __name__ == "__main__":
    all_dfs, df = load_data()
    features = gen_features(all_dfs)

    gmm, labels = run_gmm(features)

    viz_gmm(gmm, n=1000, path="tmp/")

    # plot 3D scatter.
    # show (avg vEgo, stdev vEgo, avg abs aEgo)
    # and also 2D (avg rol_lataccel, target_lataccel)

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, s=features[:, 1] * 10, cmap='viridis')
    ax.set_xlabel('avg vEgo')
    ax.set_ylabel('stdev vEgo')
    ax.set_zlabel('avg abs aEgo')
    ax.view_init(elev=20, azim=-50)
    fig.suptitle('color by GMM label, size by stdev vEgo', fontsize=10)
    cbar = fig.colorbar(scatter, ticks=np.arange(gmm.n_components))
    cbar.set_label('GMM cluster label')
    plt.savefig("tmp/features_vEgo_stdev_vEgo_abs_aEgo.png", dpi=384)

    # 2d plot
    plt.figure()
    scatter_2d = plt.scatter(features[:, 3], features[:, 4], c=labels, s=features[:, 1] * 10, cmap='viridis')
    plt.xlabel('avg roll')
    plt.ylabel('avg target_lataccel')
    plt.title('color by GMM label, size by stdev vEgo', fontsize=10)
    cbar_2d = plt.colorbar(scatter_2d, ticks=np.arange(gmm.n_components))
    cbar_2d.set_label('GMM cluster label')
    plt.savefig("tmp/features_roll_target_lataccel.png", dpi=384)





