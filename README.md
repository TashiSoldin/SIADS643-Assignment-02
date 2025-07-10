# SIADS643-Assignment-02
## Penguin Data Analysis Pipeline

### Overview 
This project converts a Jupyter notebook analysis into a modularised, reusable Python pipeline that:
- Loads penguin data from multiple sources
- Cleans and preprocesses the data
- Creates informative visualizations
- Trains a machine learning model to classify penguin species
- Saves all results for further analysis

### Requirements
- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scikit-learn

### Usage
1. Install requirements: pip install -r requirements.txt
2. Run the pipeline: python main.py <data_source> <output_directory> [exclude_columns] (e.g. python main.py seaborn output/ island)
    - <data_source>: either seaborn for the built-in datasets or the path to the input CSV file
    - <output_directory>: file path to save results and visualisations
    - [exclude_columns]: optional comma-separated list of columns to exclude from modeling (i.e. particularly those that contribute to data-leakage)

#### Examples 
```bash
# Basic Usage
python main.py seaborn output/

# With CSV file data 
python main.py data/penguin_data.csv output/

# Multiple column exclusion
python main.py data/penguin_data.csv output/ island,id
```

### Classes and Functions
| **Module** | **Function** | **Description** |
|------------|--------------|-----------------|
| **data_loader.py** | `load_data_from_seaborn(dataset_name)` | Load seaborn datasets |
| | `load_data_from_csv(file_path)` | Load data from CSV |
| | `save_data(df, output_path)` | Save DataFrame to CSV |
| **data_cleaner.py** | `clean_data(df)` | Remove duplicates and handle missing values |
| | `encode_categorical(df, target_col)` | Encode categorical variables |
| **visualizer.py** | `create_numerical_distribution_plots(df, output_path)` | Numeric histograms |
| | `create_categorical_distribution_plots(df, output_path)` | Categorical bar charts |
| | `create_correlation_heatmap(df, output_path)` | Correlation matrix |
| | `create_visualizations(df, output_dir)` | Generate all plots |
| | `save_visualization(output_path)` | Save current plot |
| **model_trainer.py** | `prepare_data(df, target_column, exclude_columns)` | Prepare features and target |
| | `train_and_evaluate(X, y, test_size)` | Train and evaluate model |
| | `save_results(results, output_path, target_column)` | Save model results |
| | `train_model(df, target_column, exclude_columns)` | Complete training pipeline |

Other files in the repo are:
- **penguin_analysis_notebook.ipynb**: original analysis notebook that was the basis for the modularised python pipeline
- **/data**: folder that holds the CSV input file
- **/output**: folder that holds the output processed file, visualisation plots and the model results

### Data Overview
The seaborn penguin dataset, whether loaded directly from seaborn or via the CSV file, has the following data variables
- `species` penguin species (target variable)
- `island` island location
- `bill_length_mm` bill length in millimeters
- `bill_depth_mm` bill depth in millimeters
- `flipper_length_mm` flipper length in millimeters
- `body_mass_g` body mass in grams
- `sex` penguin sex

### Output Files
The pipeline creates:
- **`cleaned_data.csv`** - Processed dataset
- **`model_results.txt`** - Model performance and feature importance  
- **`plots/`** - Directory containing all visualizations

### Example Output
```
Starting data analysis...
Loaded data: (344, 7)
Cleaned data: (342, 7)
Creating visualizations...
Numeric columns: ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
Categorical columns: ['species', 'island', 'sex']
Created distribution plots for 4 numeric columns
Created bar charts for 3 categorical columns
Created correlation heatmap
All visualizations created and saved!
Training model to predict 'species'...
Accuracy: 0.990
Training samples: 239
Test samples: 103
Model accuracy: 0.990
Results saved to: output/
```
