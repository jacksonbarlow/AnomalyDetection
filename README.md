

# User Guide

# Overview
This guide explains how to run the anomaly detection system that was developed as part of this project. The system takes raw traffic data from real-world road conditions and processes it to identify unusual driving behaviour, particularly related to lane-change manoeuvres.

You don't need to understand how the system works internally. This guide is a set of instructions on how to set up the system and use it to get results.

The system runs through a series of key steps:
- Preprocess the raw data: This cleans the input trajectory data and prepares it for analysis.
- Train a detection model: This model aims to learn what normal driving looks like.
- Score the data: This checks how unusual each driving sequence is.
- Visualise the results: This shows the most unusual behaviour and how it relates to lane changes.


Everything is controlled using simple commands typed into a terminal. You will be told exactly what to type and when, and what files or plots to expect at the end. No programming experience is required, just follow the steps in order.
# Installation and Requirements
Before you can run this system, you will need to set up your computer with the right tools and hardware.

# Minimum System Requirements
- Operating System: This has only been tested thoroughly on Windows, but should support macOS or linux too.
- RAM): At least 8 GB 16 GB recommended
- Storage: At least 35 GB of free disk space
- GPU): An NVIDIA GPU) with CUDA) support e.g., RTX 20xx or newer. This speeds up model training significantly. The system can technically run on a regular CPU), but it will be much slower and may not work for large datasets.


# Software Requirements
The system uses Python and a number of standard scientific computing libraries. You will need:
- Python 3.8 or later
- pip Python's package installer
- The following Python libraries:
- `torch` PyTorch
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `joblib`
- `pywt`
- `scipy`
    


# Installation Steps
# 1. Install Python
Go to the official Python website:

[https://www.python.org/downloads/]Python Download Page

Download the latest version of Python 3.8 or newer, and during installation:
- Check the box that says "Add Python to PATH"
- Then click "Install Now"

This will install Python and its package manager, `pip`
# 2. Download the Project Files
If you are familiar with Git and have it installed, you should open a terminal and run the following command:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

If you are not familiar and are a new user follow these steps:
- Go to [https://github.com/yourusername/your-repo-name.git]Project GitHub Repository
- Click the green "Code" button, then choose "Download ZIP"
- Extract unzip the folder somewhere on your computer.
- Open a terminal Command Prompt on Windows and move into that folder with this command: `cd path/to/AnomalyDetection`


# 3. Install Required Packages
In the same terminal window, type:

`pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm joblib pywt scipy`

This installs all the software the project needs. If any command fails, make sure you're connected to the internet and that Python was installed correctly. Once this is done, you're ready to run the system, now you just need the raw dataset.

# 4. Download the NGSIM Dataset
This project uses real vehicle trajectory data from the NGSIM) program.
- Go to: [https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm]NGSIM Dataset Download Page
- Scroll to "Attachments" and download the following ZIP file: "I-80-Emeryville-CA.zip".
- Once downloaded extract the ZIP file into the following location inside the project folder:
    ```
         your-project-folder/
          └── trajectory_data/
              └── I-80-Emeryville-CA/
                  └── 0400pm-0415pm/
                      ├── trajectories-0400-0415.csv
                      ├── trajectories-0400-0415.txt
                      └── ... other CSVs

    ```


# Step 1: Preprocess the Data
Before the system can analyse traffic behaviour, the raw NGSIM) dataset must be cleaned and converted into a structured format. This step prepares everything needed for model training and evaluation.

What this step does:
- Smooths noisy features
- Removes or clips unrealistic values
- Reconstructs velocity and acceleration
- Detects nearby vehicles context features
- Converts everything into fixed-length sequences


This only needs to be done once. Expect it to take some time, especially if loading and combining multiple CSVs which will be done by default if they are present in the directory.

# How to run it
- Open a terminal Command Prompt, Terminal, or shell
- Navigate to the `AnomalyDetection` directory using the `cd` command.
- Then run: `python main.py --build_data`

You will see progress bars and messages as the system processes the data. Be aware that the lane change mask generation does not have a progress bar, and will take some time.

# What to expect
Once complete, the following outputs will be created:
- A folder called `cache/`, containing:
- `Sequences.npz`: sequence metadata that is used for loading
- `Sequences_seq_memmap.npy`: memory-mapped array of input sequences
- `Sequences_tgt_memmap.npy`: memory-mapped array of corresponding targets
- `scaled_features.parquet`: full scaled DataFrame of selected features
- `scaler.pkl`: a fitted StandardScaler used for feature standardisation
- `context_cache.pkl`: precomputed context features for each vehicle-frame
- A folder called `plots/preprocessing/` which contains diagnostic plots including:
- Visual comparisons of raw vs. smoothed features
- Histograms showing clipping effects
- Distribution checks of standardised values
- Heatmaps showing delta sequence consistency
    


These files are required for training and evaluating the model. If the `cache/` folder is missing or incomplete, make sure the dataset was extracted correctly and try rerunning the command.
## Step 2: Train the Model*
Once the data has been preprocessed, the next step is to train the anomaly detection model. This model learns what *normal) driving behaviour looks like based on the training data, so it can later identify unusual patterns.

# What this step does
It loads the cleaned sequence data from the cache, then splits it into training and validation sets. The neural network model is then trained LSTM autoencoder) and saved for later use. This step may take some time, especially on slower computers or if no GPU) is available.

# How To Run Training
- Open a terminal
- Navigate to the project folder
- Run the following command: `python main.py --mode train --model autoencoder --epochs 50`


You can adjust the number of training epochs if needed e.g., use `--epochs 10` for a quicker run. Progress bars will show how the training is going. I have found that the model tends to converge in under 30 epochs.

# Optional parameters:
- `--load`: Resumes training from a previously saved model
- `--latent_dim`: Sets the size of the model's internal bottleneck e.g., `--latent_dim 128`
- `--window`: Sets how many time steps each sequence contains


You can combine options like this:

`python main.py --mode train --model autoencoder --load checkpoints/checkpoint.pt --epochs 20 --latent_dim 64`

Training may also be exited at any time, and a checkpoint will be saved automatically if you use the key-bind Ctrl+c

# What to Expect
After training, the following will be saved automatically:
- `checkpoints/`
- `autoencoder.pt`: the final trained model
- `autoencoder_epochX.pt`: intermediate checkpoints every 10 epochs.
- `reconstruction_samples/epoch_plots/epoch_x_feature.png`: Showing the reconstruction versus the target for epoch x.

## Step 3: Score the Data*
Once the model has been trained, the next step is to use it to score each sequence in the dataset. These scores represent how *unusual* or *abnormal) each sequence is, based on how much it deviates from the learned patterns of normal driving.

# What this step does
This command runs several detection methods to assign an anomaly score to each driving sequence:
- Reconstruction score: Measures how well the autoencoder was able to reconstruct the original input.
- Mahalanobis distance: Detects unusual patterns in the reconstruction errors
- Traditional methods:
- Isolation Forest
- Local Outlier Factor
    


These scores are combined into a single hybrid score, which is saved for later use and visualisation.

# How to Run Evaluation
- Open a terminal
- Navigate to the project folder `AnomalyDetection`
- Run: `python main.py --mode eval --model autoencoder`


This command uses the most recent trained model `autoencoder.pt` and the cached data. It does not retrain anything.

If you want to load a different checkpoint use the argument `--load` with the appropriate file path following.

# What to Expect
After running this step, the following files will be created:
- `data/`
- `reconstruction_scores.npy`: scores based on reconstruction error
- `mahalanobis_scores.npy`: scores based on Mahalanobis distance
- `isolation_forest_scores.npy`: scores from Isolation Forest
- `lof_scores.npy`: scores from Local Outlier Factor
- `hybrid_scores.npy`: final combined score
- `plots/evaluation/`
- Score distribution plots e.g., histograms for reconstruction and Mahalanobis scores
- Comparison of hybrid vs. individual scoring methods
- Plots of the top-N highest scoring most abnormal sequences
- Diagnostic plots seen in the results section and presentation relating to lane-change analysis.
    


These outputs can be used to analyse how the system performs and which sequences it considers most unusual.
## Final Notes*
This guide has covered the full pipeline for detecting unusual driving behaviour using trajectory data, from preprocessing to model evaluation. For most users, these steps are sufficient to run the system and generate meaningful results.

For more advanced users, the system has been designed to be modular and configurable. Developers may wish to experiment with:
- Changing the model architecture
- Adjusting context feature definitions or dimensions
- Introducing new anomaly scoring techniques
- Running cross-dataset comparisons or real-time extensions


All source code is open and documented, allowing these extensions to be made with minimal disruption to the main pipeline.
