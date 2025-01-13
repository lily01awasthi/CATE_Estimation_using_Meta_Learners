# Estimation of CATE using Meta-Learners

This repository contains all materials related to the project "Estimation of CATE using Meta-Learners (T, S, X, R-learners)", aimed at exploring Conditional Average Treatment Effects (CATE) using various meta-learner models on two distinct datasets: analysis data and benchmark data.

## Project Structure

The project is organized into several directories, each serving a specific purpose in the analysis and implementation of CATE estimation methods. Below is a detailed explanation of each directory and its contents:

### `data`
This directory hosts all datasets used in the project, split into two main subdirectories:

#### `analysis_data`
- `dataset.csv`: The primary dataset used for initial analysis.
- [analysis_data_description.md](./data/analysis_data/analysis_data_description.md): Describes the dataset, including its features and the preprocessing steps applied.

#### `benchmark_data`
- `checkins_lessions_checkouts_training.csv` and `construct_experiments_ates_test.csv`: Raw data files used to generate the benchmark dataset through specific filtration processes described within.
- [benchmark_data_description.md](./data/benchmark_data/benchmark_data_description.md): Provides details on the benchmark dataset, including its creation and characteristics.

### `docs`
- `Estimation_of_CATE_using_Meta_Learners.pdf`: Contains the comprehensive thesis document detailing the theoretical background, methodology, and findings of the project.

### `models`
This directory contains scripts for the meta-learner models used to analyze both datasets:

#### `Meta_Learners` and `Meta_Learners_benchmark_data`
Each contains identical scripts (`meta_learner_models.py`, `s_learner.py`, `t_learner.py`, `x_learner.py`, `r_learner.py`). These scripts implement different meta-learner models. They are functionally the same but are tuned to the specifics of the analysis and benchmark datasets respectively.

### `notebooks`
Jupyter notebooks that document the experimental process and findings for each dataset:

#### `analysis_data_notebooks` and `benchmark_data_notebooks`
Series of experiments (`Exp1` to `Exp5`) detailing data preprocessing, hyperparameter tuning, model selection, performance evaluation, and in-depth analyses of CATE estimates across various conditions and demographics.


### `results`
Contains output from the experiments, including model evaluation metrics and CATE estimates:

#### `analysis_data_results` and `benchmark_data_results`
Stores results of grid search, CATE estimates by PSM, and meta-learner predictions on test datasets.

### `scripts`
Scripts used for data preprocessing, model evaluation, hyperparameter tuning, and training across both datasets:

#### `analysis_data_scripts` and `benchmark_data_scripts`
Includes Python scripts like `data_preprocessor.py`, `evaluate_models.py`, `hyperparameter_tuning_and_model_selection.py`, and `train_models.py`. These scripts are integral to preparing the data, tuning, and training the models for both the analysis and benchmark datasets.

### `requirements.txt`
Contains all the necessary Python packages required to run the project. Ensure you install these dependencies to avoid any runtime issues.

## Getting Started

* To get started with this project, clone the repository and install the required packages using:

```bash
pip install -r requirements.txt

```

* Go to  notebooks under which you will find two directories. 
    * analysis_data_notebooks
    * benchmark_dataset_notebooks
* Each of them have five notebooks that Explore, Analyze, tune the hyperparameters,select best performing model, estimate cate using meta-learner, compare the meta-learner prformances, estimate cate using PSM method and evalute the meta-learners for individual dataset.
    * analysis_data_notebooks
        * Exp1.Data Preprocessing and Exploration.ipynb 
        * Exp2.Hyperparameter_Tuning_and_model_Selection.ipynb
        * Exp3.Performance_of_each_meta_learner_on_test_set.ipynb
        * Exp4.CATE_with_PSM.ipynb
        * Exp5.Estimated_CATE_across_demographics.ipynb
    * benchmark_dataset_notebooks
        * Exp1.Performance_of_meta_learners_on_benchmark_data.ipynb
        * Exp2.Hyperparameter_Tuning_and_model_Selection.ipynb
        * Exp3.CATE_with_PSM.ipynb
        * Exp4.correlation_in_cate_estimates_by_meta_learners.ipynb
        * Exp5.cate_accuracy_across_sample_sizes.ipynb
* please run each notebook to see the results, which are explained in the notebook itself. 
