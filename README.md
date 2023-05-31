
# Trimmed Marginal Likelihood GP

Implementation of the Trimmed Marginal Likelihood GP for regression proposed in "Robust Gaussian Process Regression with the Trimmed Marginal Likelihood", UAI 2023.


## Requirements

- Python >= 3.11.2
- PyTorch >= 2.0.1
- GPyTorch >= 1.10


## Usage (Example Workflow)

Example using Friedman (n=100) dataset with 10% outliers.

-------------------------------------------
1. Prepare Data
-------------------------------------------
Create some artificial data (or add outliers to existing real data, see prepare_data_with_addded_outliers.py for details):
```bash
python prepare_data_with_addded_outliers.py
```

Datasets are saved into folder "openDatasets_prepared/".

-------------------------------------------
2. Run Experiments
-------------------------------------------
Run proposed method (with fixed prespecified $\nu$ = 0.2):
```bash
python runExperiments.py Friedman_n100 trimmed_informative_nu_withoutCV focused projectedGradient 0.1 0.2
```

Run student-t GP:
```bash
python runExperiments.py Friedman_n100 student focused None 0.1
``` 

Results for analysis are saved into folder "all_results/".

-------------------------------------------
3. Show summary of all results
-------------------------------------------

```bash
python showResults_all.py
``` 


## Other files

"showExampleData.py": get nice figures for synthetic bow-shaped

## Citation

If you use (some) of the code then please cite
"Robust Gaussian Process Regression with the Trimmed Marginal Likelihood", UAI 2023
