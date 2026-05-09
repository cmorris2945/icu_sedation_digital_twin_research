# Local Setup Instructions for ICU Sedation Ablation Study

This document walks you through running the ablation study experiments on your own computer. Following these steps in order will produce all the data and figures needed for the paper analysis without depending on any remote compute resources.

## Step One: Verify Your Python Environment

Open a terminal or command prompt on your computer and check your Python version by typing the following command. You should see version 3.10 or newer.

```
python --version
```

If your Python is too old, please update it before proceeding. Most modern operating systems make this straightforward through their normal software update mechanisms. On Windows you can download the latest version from python.org. On Mac you might use Homebrew with the command brew install python. On Linux your package manager should have a recent version.

## Step Two: Install Required Python Packages

The scripts depend on several Python packages for data manipulation, machine learning, and visualization. Install them all at once using pip with this command. Depending on your internet speed and computer this might take a few minutes, with TensorFlow being the largest download.

```
pip install pandas numpy scipy scikit-learn xgboost tensorflow matplotlib
```

If pip is not on your path you might need to use python -m pip install instead. If you are on a system that uses pip3 specifically, use that instead. The package names are the same regardless of which pip command you use.

After installation completes, verify everything imports correctly by running this quick test in Python.

```
python -c "import pandas, numpy, scipy, sklearn, xgboost, tensorflow, matplotlib; print('All packages OK')"
```

If you see the message "All packages OK" you are ready to proceed. If you get import errors, the error message will tell you which package failed and you can install just that one.

## Step Three: Set Up the Project Folder

Create a folder somewhere on your computer where the experiments will live. The exact location does not matter as long as you can find it later. Some good choices would be your Documents folder, your Desktop, or a dedicated research folder if you have one. Inside this main folder, create a subfolder named data exactly that way, all lowercase.

The final folder structure should look like the diagram below where you place the four Python scripts I provided directly in the top level folder, and the three data files go in the data subfolder.

```
icu-sedation-experiments/
    build_features.py
    run_ablation.py
    analyze_results.py
    make_figures.py
    data/
        kidney_subgroups_patients.csv
        kidney_subgroups_propofol.csv
        kidney_subgroups_sas.csv
```

The three CSV files are the patient cohort data we extracted from MIMIC-IV. You should already have these from your previous BigQuery work. If you cannot find them, let me know and we can figure out how to recreate them from your BigQuery queries.

## Step Four: Run the Feature Building Script

Open a terminal in your project folder and run the first script. This is the slow step because the pharmacokinetic ODE solver has to run for every patient in the cohort. On a typical laptop you should expect this to take fifteen to thirty minutes for the full 1,490 patient dataset.

```
python build_features.py
```

The script will print progress as it processes patients in batches of 100. When it finishes, you will see a file named cached_features.pkl in your project folder. This file contains all the processed features and will make subsequent steps much faster.

If you want to test the pipeline first with a smaller patient sample before committing to the full run, you can edit the script and change the line that says N_PATIENTS = None to something like N_PATIENTS = 200. The 200 patient version will run in about five minutes and lets you verify everything works before doing the full run.

## Step Five: Run the Ablation Study

Once features are built, run the ablation experiments. This script trains five different feature configurations against both XGBoost and LSTM, with three random seeds for the LSTM to assess stability. Total expected runtime is approximately thirty to forty-five minutes depending on your CPU speed.

```
python run_ablation.py
```

The script saves results incrementally after every single training run, so if anything goes wrong or you need to interrupt it, you can just run the same command again later and it will pick up where it left off without redoing completed work. You will see progress messages as each training run completes, with the MAE and R-squared values printed for verification.

If you have a GPU available and TensorFlow is configured to use it, the LSTM training will be substantially faster. The script does not require GPU support but will use it automatically if available.

## Step Six: Analyze the Results

After the ablation completes, run the analysis script to compute summary statistics and degradation analyses. This runs in seconds because it just processes the results file we already have.

```
python analyze_results.py
```

The script prints a comprehensive summary to the console showing the mean and standard deviation for each configuration, plus the degradation percentages relative to the full hybrid baseline. It also saves two output files. The summary_statistics.csv file contains the per-configuration statistics in a clean tabular format. The degradation_analysis.csv file contains the degradation percentages with proper error bars from the multi-seed LSTM runs.

## Step Seven: Generate Figures

Finally, run the figure generation script to produce publication-quality visualizations from the results.

```
python make_figures.py
```

This creates a figures subfolder with two main plots in both PNG and PDF formats. The ablation_main figure shows the degradation percentages with error bars, which is the key visualization for our paper. The absolute_mae figure shows the actual MAE values for context.

## What to Send Me When You Are Done

Once all four scripts have completed successfully, please send me back the following items so we can analyze the complete results together. The most important file is ablation_results.csv which contains the raw data from every training run. The summary_statistics.csv and degradation_analysis.csv files contain the processed analyses. The figures in the figures subfolder will show me visually what the results look like.

You can either zip the whole project folder and send that to me, or just send the individual CSV files and PNG figures. Either way works. Once I have the data I can help you interpret what we found and decide how to write it up for your advisor.

## Troubleshooting Common Issues

If build_features.py is very slow even at the start, your CPU might be older than typical. The PK simulation is computationally intensive and there is not much we can do to speed it up beyond using a faster computer. Letting it run overnight is a perfectly reasonable strategy if your computer is slow.

If you get out of memory errors during run_ablation.py, your computer may not have enough RAM for the LSTM training with full sequences. Try editing the script to reduce the batch_size parameter from 64 to 32 or even 16. This will train slower but use less memory.

If TensorFlow prints lots of warnings about CUDA or GPU, that is usually safe to ignore. The script will fall back to CPU training automatically. The warnings just mean your system does not have an NVIDIA GPU configured for TensorFlow, which is fine.

If you see a message about "no GPU found" or similar, that is also fine. The scripts work on CPU. They will just be slower than they would be with GPU support.

If pip install fails for any specific package, sometimes the issue is that you need to upgrade pip itself first. Try running pip install --upgrade pip and then retrying the package install.

## Estimated Total Time Investment

The total time from starting setup to having complete results depends mostly on your computer speed. On a recent laptop with a fast CPU you might be done in about an hour and a half total. On an older computer it could take three or four hours. The good news is that most of this time is unattended, where you can leave the computer running and check back later. The actual hands-on time entering commands is probably fifteen minutes total spread across the whole process.

Once you have results, the analysis and interpretation we do together should take much less time because we have already done most of the thinking. We just need to look at the numbers and figures and decide what they mean.
