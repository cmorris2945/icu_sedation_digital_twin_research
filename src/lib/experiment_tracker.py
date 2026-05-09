"""
experiment_tracker.py

This module provides a lightweight experiment tracking system that logs all
the details of each experimental run to a structured JSON file. The purpose
is to ensure that every analysis we perform is fully reproducible, with a
complete record of what data was used, what model configuration was tried,
what hyperparameters were chosen, and what results were obtained.

We chose to build this simple tracker rather than using more complex tools
like MLflow or Weights and Biases because the simplicity has real benefits
for academic research. The output is a plain JSON file that anyone can read
and verify, with no special software needed. It works on any computer
without configuration. And the entire system fits in one file, so it can be
included as a supplementary file with a paper submission to make experimental
records fully transparent.

The intended workflow is to create an ExperimentTracker instance at the
start of each experiment, log information as the experiment progresses,
and call save at the end. Multiple experiments append to the same log file
so over time you build up a complete record of all the experiments that
contributed to your research.

AUTHOR: Christopher Morris
"""

import json
import os
from datetime import datetime
import sys
import platform


class ExperimentTracker:
    """
    Tracks experimental runs and saves complete records to JSON.
    
    The tracker captures information in several categories. Environment
    information includes Python version, operating system, and package
    versions. Data source information describes where the input data came
    from and how it was extracted. Preprocessing information records how
    raw data was transformed into model inputs. Model configuration
    captures all hyperparameters and architectural choices. Results
    contain the actual numbers produced by the experiment. Notes are
    free-form observations the researcher wants to preserve.
    
    Each call to save appends the current experiment record to a JSON
    file containing all previous experiments. This creates a complete
    audit trail of the research over time.
    """
    
    def __init__(self, experiment_name, log_dir='results'):
        """
        Initialize a new experiment record. Creates the log directory if
        it does not exist and captures basic environment information that
        will be useful for reproducibility.
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'experiment_log.json')
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize the experiment record with environment info captured
        # automatically. This helps anyone reading the log later understand
        # what software environment produced the results.
        self.experiment = {
            'name': experiment_name,
            'started_at': datetime.now().isoformat(),
            'status': 'running',
            'environment': self._capture_environment(),
            'data_source': None,
            'preprocessing': None,
            'model_config': None,
            'training': None,
            'results': None,
            'notes': []
        }
    
    def _capture_environment(self):
        """
        Capture information about the computing environment. This includes
        Python version, operating system, and the versions of key packages
        we depend on. This information is essential for reproducibility
        because behavior can differ subtly between package versions.
        """
        env = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'captured_at': datetime.now().isoformat()
        }
        
        # Try to capture versions of packages we use. If a package is not
        # installed we just skip it rather than failing.
        for pkg_name, import_name in [
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('scipy', 'scipy'),
            ('scikit-learn', 'sklearn'),
            ('xgboost', 'xgboost'),
            ('tensorflow', 'tensorflow'),
        ]:
            try:
                module = __import__(import_name)
                env[pkg_name] = module.__version__
            except (ImportError, AttributeError):
                pass
        
        return env
    
    def log_data_source(self, **kwargs):
        """
        Record information about where the data came from. Common keys
        include database, query_file, n_patients, inclusion_criteria.
        """
        self.experiment['data_source'] = kwargs
    
    def log_preprocessing(self, **kwargs):
        """
        Record preprocessing decisions. Common keys include
        feature_engineering, missing_data_handling, train_test_split.
        """
        self.experiment['preprocessing'] = kwargs
    
    def log_model_config(self, **kwargs):
        """
        Record the model configuration including all hyperparameters and
        the rationale for choosing those values. The rationale is critical
        because it lets future readers understand not just what was done
        but why.
        """
        self.experiment['model_config'] = kwargs
    
    def log_training(self, **kwargs):
        """
        Record information about the training process. Common keys
        include epochs, batch_size, training_time, convergence_notes.
        """
        self.experiment['training'] = kwargs
    
    def log_results(self, **kwargs):
        """
        Record the experimental results. Common keys include mae, rmse,
        r2, comparison_baseline, statistical_significance.
        """
        self.experiment['results'] = kwargs
    
    def add_note(self, note):
        """
        Add a timestamped free-form note to the experiment record. These
        notes are valuable for capturing observations that do not fit
        into the structured fields, such as unexpected behavior or ideas
        for follow-up work.
        """
        self.experiment['notes'].append({
            'timestamp': datetime.now().isoformat(),
            'note': note
        })
    
    def save(self, status='completed'):
        """
        Save the experiment record to the log file. Appends to existing
        log if one exists, or creates a new log otherwise. Returns the
        path to the log file.
        """
        self.experiment['status'] = status
        self.experiment['completed_at'] = datetime.now().isoformat()
        
        # Load existing experiments if the log already exists
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                all_experiments = json.load(f)
        else:
            all_experiments = []
        
        all_experiments.append(self.experiment)
        
        with open(self.log_file, 'w') as f:
            json.dump(all_experiments, f, indent=2, default=str)
        
        return self.log_file
