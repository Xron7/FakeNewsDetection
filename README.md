# FakeNewsDetection

The datasets are not available in this repo. They can be downloaded from [here](https://www.dropbox.com/scl/fi/flgahafqckxtup2s9eez8/rumdetect2017.zip?e=3&file_subpath=%2Frumor_detection_acl2017&rlkey=b7v86v3q1dpvcutxqk0xi7oej&dl=0).

## Setup
The necessary libraries are installed with:

```pip install -r requirements.txt```

---

If you are creating a new environment, you may need to run:

```pip install ipykernel```

```python -m ipykernel install --user --name={env_name}```

---

so that you can select the venv python from jupyter notebook.

---

To execute python code use:

```python -m path.to.file {args}```

## Walkthrough

### preprocessing

- `feature_extraction.py`: Extracts the features from the raw data in a tabular format
- `user_dataset_creation.py`: Creates user related features
- `eda.ipynb`: Exploratory Data Analysis of the tabular data

---

### nlp

- `sentiment_analysis.ipynb`: Extracts the sentiment of the tweets (Originally run in Google Colab)

---

### models

-  `main.py`: Executes the given configuration (.json files)

---

### network

- `construct_network.py`: Constructs and saves the network as a graph using the raw data
- `extract_node_features.py`: Extracts the features of the nodes from the files generated in the preprocessing section
- `network_analysis.ipynb`: Analysis of the constructed network
- `GATE.py`: Implementation of the Graph Attention Auto-Encoder architecture
- `run_gate.py`: Executes the given configuration (.json file)
- `node_selection.ipynb`: Analysis of the results of GATE and selection of the nodes to be removed
- `node_cleaning.py`: Removal of the designated nodes and results
