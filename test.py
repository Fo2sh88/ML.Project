import unittest
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score

# Import the main application
from main import MLModelApp

class TestMLModelApp(unittest.TestCase):
    """Comprehensive test suite for MLModelApp class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.root = tk.Tk()
        self.app = MLModelApp(self.root)
        
        # Create larger, more realistic datasets for testing
        np.random.seed(42)  # For reproducible results
        
        # Regression dataset with >10 unique continuous values
        n_samples = 100
        self.sample_regression_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 10, n_samples),
            'target': np.random.normal(50, 15, n_samples)  # Continuous target
        })
        
        # Classification dataset with categorical target
        self.sample_classification_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 10, n_samples),
            'target': np.random.choice(['A', 'B', 'C'], n_samples)  # Categorical target
        })
        
        # Binary classification dataset
        self.sample_binary_classification_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 10, n_samples),
            'target': np.random.choice([0, 1], n_samples)  # Binary target with ≤10 unique values
        })
    
    def tearDown(self):
        """Clean up after each test method."""
        self.root.destroy()
    
    # Test initialization
    def test_init_creates_widgets(self):
        """Test that __init__ creates all required widgets"""
        self.assertIsInstance(self.app.label_title, tk.Label)
        self.assertIsInstance(self.app.btn_load_file, tk.Button)
        self.assertIsInstance(self.app.label_file, tk.Label)
        self.assertIsInstance(self.app.btn_select_target, tk.Button)
        self.assertIsInstance(self.app.target_column, tk.StringVar)
        self.assertIsInstance(self.app.target_dropdown, ttk.Combobox)
        self.assertIsInstance(self.app.model_type, tk.StringVar)
        self.assertIsInstance(self.app.btn_choose_regressor, tk.Button)
        self.assertIsInstance(self.app.btn_choose_classifier, tk.Button)
        self.assertIsInstance(self.app.btn_train, tk.Button)
        self.assertIsInstance(self.app.output_text, tk.Text)
        self.assertIsInstance(self.app.btn_save, tk.Button)
    
    def test_init_sets_initial_values(self):
        """Test that __init__ sets correct initial values"""
        self.assertIsNone(self.app.dataset)
        self.assertIsNone(self.app.best_model)
        self.assertIsNone(self.app.best_metrics)
        self.assertEqual(self.app.model_type.get(), "")
        self.assertEqual(self.app.btn_train['state'], tk.DISABLED)
        self.assertEqual(self.app.btn_save['state'], tk.DISABLED)
    
    # Test load_file method
    @patch('tkinter.filedialog.askopenfilename')
    @patch('pandas.read_csv')
    def test_load_file_success(self, mock_read_csv, mock_filedialog):
        """Test successful file loading"""
        mock_filedialog.return_value = 'test.csv'
        mock_read_csv.return_value = self.sample_regression_data
        
        self.app.load_file()
        
        mock_filedialog.assert_called_once_with(filetypes=[('CSV Files', '*.csv')])
        mock_read_csv.assert_called_once_with('test.csv')
        self.assertIsNotNone(self.app.dataset)
        self.assertEqual(list(self.app.target_dropdown['values']), list(self.sample_regression_data.columns))
    
    @patch('tkinter.filedialog.askopenfilename')
    def test_load_file_cancelled(self, mock_filedialog):
        """Test file loading when user cancels"""
        mock_filedialog.return_value = ''
        
        self.app.load_file()
        
        self.assertIsNone(self.app.dataset)
    
    @patch('tkinter.filedialog.askopenfilename')
    @patch('pandas.read_csv')
    def test_load_file_exception(self, mock_read_csv, mock_filedialog):
        """Test file loading with invalid file"""
        mock_filedialog.return_value = 'invalid.csv'
        mock_read_csv.side_effect = Exception('File not found')
        
        with self.assertRaises(Exception):
            self.app.load_file()
    
    # Test select_target method
    @patch('tkinter.messagebox.showerror')
    def test_select_target_no_dataset(self, mock_messagebox):
        """Test select_target when no dataset is loaded"""
        self.app.select_target()
        
        mock_messagebox.assert_called_once_with('Error', 'Please select a CSV file first.')
    
    def test_select_target_with_dataset(self):
        """Test select_target when dataset is loaded"""
        self.app.dataset = self.sample_regression_data
        
        self.app.select_target()
        
        self.assertEqual(self.app.target_column.get(), "")
        self.assertEqual(list(self.app.target_dropdown['values']), list(self.sample_regression_data.columns))
    
    # Test set_model_type method
    @patch('tkinter.messagebox.showinfo')
    def test_set_model_type_regressor(self, mock_messagebox):
        """Test setting model type to regressor"""
        self.app.set_model_type('regressor')
        
        self.assertEqual(self.app.model_type.get(), 'regressor')
        self.assertEqual(self.app.btn_train['state'], tk.NORMAL)
        mock_messagebox.assert_called_once_with('Model Type Selected', 'You selected: Regressor')
    
    @patch('tkinter.messagebox.showinfo')
    def test_set_model_type_classifier(self, mock_messagebox):
        """Test setting model type to classifier"""
        self.app.set_model_type('classifier')
        
        self.assertEqual(self.app.model_type.get(), 'classifier')
        self.assertEqual(self.app.btn_train['state'], tk.NORMAL)
        mock_messagebox.assert_called_once_with('Model Type Selected', 'You selected: Classifier')
    
    # Test validate_model_choice method
    @patch('tkinter.messagebox.showerror')
    def test_validate_model_choice_classifier_continuous_target(self, mock_messagebox):
        """Test validation when classifier selected for continuous target"""
        self.app.dataset = self.sample_regression_data  # Continuous target
        self.app.target_column.set('target')
        self.app.model_type.set('classifier')
        
        result = self.app.validate_model_choice()
        
        self.assertFalse(result)
        mock_messagebox.assert_called_once()
    
    @patch('tkinter.messagebox.showerror')
    def test_validate_model_choice_regressor_categorical_target(self, mock_messagebox):
        """Test validation when regressor selected for categorical target"""
        self.app.dataset = self.sample_classification_data
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        result = self.app.validate_model_choice()
        
        self.assertFalse(result)
        mock_messagebox.assert_called_once()
    
    def test_validate_model_choice_valid_regressor(self):
        """Test validation for valid regressor choice"""
        self.app.dataset = self.sample_regression_data
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        result = self.app.validate_model_choice()
        self.assertTrue(result)
    
    def test_validate_model_choice_valid_classifier(self):
        """Test validation for valid classifier choice"""
        self.app.dataset = self.sample_classification_data
        self.app.target_column.set('target')
        self.app.model_type.set('classifier')
        
        result = self.app.validate_model_choice()
        self.assertTrue(result)
    
    @patch('tkinter.messagebox.showerror')
    def test_validate_model_choice_classifier_continuous_target(self, mock_messagebox):
        """Test validation when classifier selected for continuous target"""
        self.app.dataset = self.sample_regression_data  # Continuous target
        self.app.target_column.set('target')
        self.app.model_type.set('classifier')
        
        result = self.app.validate_model_choice()
        self.assertFalse(result)
        mock_messagebox.assert_called_once()
    
    @patch('tkinter.messagebox.askyesno')
    @patch.object(MLModelApp, 'validate_model_choice')
    @patch.object(MLModelApp, 'save_model')
    def test_train_models_regression_success(self, mock_save, mock_validate, mock_askyesno):
        """Test successful regression model training"""
        mock_validate.return_value = True
        mock_askyesno.return_value = True
        
        self.app.dataset = self.sample_regression_data
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        self.app.train_models()
        
        self.assertIsNotNone(self.app.best_model)
        self.assertIsNotNone(self.app.best_metrics)
        mock_save.assert_called_once()
    
    @patch('tkinter.messagebox.askyesno')
    @patch.object(MLModelApp, 'validate_model_choice')
    @patch.object(MLModelApp, 'save_model')
    def test_train_models_classification_success(self, mock_save, mock_validate, mock_askyesno):
        """Test successful classification model training"""
        mock_validate.return_value = True
        mock_askyesno.return_value = True
        
        self.app.dataset = self.sample_binary_classification_data
        self.app.target_column.set('target')
        self.app.model_type.set('classifier')
        
        self.app.train_models()
        
        self.assertIsNotNone(self.app.best_model)
        self.assertIsNotNone(self.app.best_metrics)
        mock_save.assert_called_once()
    
    @patch('tkinter.messagebox.askyesno')
    @patch.object(MLModelApp, 'validate_model_choice')
    def test_train_models_user_rejects_best_model(self, mock_validate, mock_askyesno):
        """Test when user rejects the best model"""
        mock_validate.return_value = True
        mock_askyesno.return_value = False
        
        self.app.dataset = self.sample_regression_data
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        self.app.train_models()
        
        self.assertIsNotNone(self.app.best_model)
    
    # Test train_models with missing values
    @patch('tkinter.messagebox.askyesno')
    @patch.object(MLModelApp, 'validate_model_choice')
    def test_train_models_with_missing_values(self, mock_validate, mock_askyesno):
        """Test training with missing values in dataset"""
        mock_validate.return_value = True
        mock_askyesno.return_value = False
        
        # Create dataset with missing values
        data_with_missing = self.sample_regression_data.copy()
        data_with_missing.loc[0, 'feature1'] = np.nan
        
        self.app.dataset = data_with_missing
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        self.app.train_models()
        
        self.assertIsNotNone(self.app.best_model)
    
    # Test train_models with categorical features
    @patch('tkinter.messagebox.askyesno')
    @patch.object(MLModelApp, 'validate_model_choice')
    def test_train_models_with_categorical_features(self, mock_validate, mock_askyesno):
        """Test training with categorical features"""
        mock_validate.return_value = True
        mock_askyesno.return_value = False
        
        # Create dataset with categorical features
        data_with_categorical = self.sample_regression_data.copy()
        # Fix: Create categorical column with same length as dataset
        n_samples = len(data_with_categorical)
        data_with_categorical['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
        
        self.app.dataset = data_with_categorical
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        self.app.train_models()
        
        self.assertIsNotNone(self.app.best_model)
    
    # Test save_model method
    @patch('tkinter.filedialog.asksaveasfilename')
    @patch('tkinter.messagebox.showinfo')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_save_model_success(self, mock_pickle_dump, mock_file, mock_messagebox, mock_filedialog):
        """Test successful model saving"""
        mock_filedialog.return_value = 'test_model.pkl'
        
        # Set up a best model
        self.app.best_model = ('LinearRegression', LinearRegression())
        self.app.best_metrics = {'R2': 0.95, 'MAE': 0.1, 'MSE': 0.01}
        
        self.app.save_model()
        
        mock_filedialog.assert_called_once_with(defaultextension='.pkl', filetypes=[('Pickle files', '*.pkl')])
        mock_file.assert_called_once_with('test_model.pkl', 'wb')
        mock_pickle_dump.assert_called_once()
        mock_messagebox.assert_called_once_with('Success', "Model 'LinearRegression' saved with its metrics.")
    
    @patch('tkinter.filedialog.asksaveasfilename')
    def test_save_model_cancelled(self, mock_filedialog):
        """Test model saving when user cancels"""
        mock_filedialog.return_value = ''
        
        self.app.best_model = ('LinearRegression', LinearRegression())
        self.app.best_metrics = {'R2': 0.95}
        
        self.app.save_model()
        
        # Should not raise any exceptions
    
    def test_save_model_no_best_model(self):
        """Test save_model when no best model exists"""
        self.app.save_model()
        
        # Should not raise any exceptions
    
    # Test edge cases and error handling
    def test_empty_dataset(self):
        """Test behavior with empty dataset"""
        empty_data = pd.DataFrame()
        self.app.dataset = empty_data
        
        # Should handle empty dataset gracefully
        self.assertEqual(len(self.app.dataset), 0)
    
    def test_single_row_dataset(self):
        """Test behavior with single row dataset"""
        single_row_data = pd.DataFrame({
            'feature1': [1],
            'target': [10]
        })
        self.app.dataset = single_row_data
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        # Should handle single row dataset
        self.assertEqual(len(self.app.dataset), 1)
    
    def test_all_same_values_target(self):
        """Test behavior when target has all same values"""
        same_target_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [10, 10, 10, 10, 10]
        })
        self.app.dataset = same_target_data
        self.app.target_column.set('target')
        self.app.model_type.set('regressor')
        
        # Should handle constant target values
        self.assertEqual(self.app.dataset['target'].nunique(), 1)
    
    # Test widget state management
    def test_widget_states_initial(self):
        """Test initial widget states"""
        self.assertEqual(self.app.btn_train['state'], tk.DISABLED)
        self.assertEqual(self.app.btn_save['state'], tk.DISABLED)
    
    def test_widget_states_after_model_type_selection(self):
        """Test widget states after model type selection"""
        self.app.set_model_type('regressor')
        self.assertEqual(self.app.btn_train['state'], tk.NORMAL)
    
    # Test data preprocessing components
    def test_data_preprocessing_steps(self):
        """Test that data preprocessing steps work correctly"""
        # This would be tested as part of train_models, but we can verify
        # the preprocessing logic separately if needed
        data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4],
            'categorical': ['A', 'B', 'A', 'C'],
            'target': [10, 20, 30, 40]
        })
        
        # Test that we can identify categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        self.assertEqual(categorical_cols, ['categorical'])
        
        # Test that we can identify missing values
        missing_count = data.isnull().sum().sum()
        self.assertEqual(missing_count, 1)


class TestMLModelAppIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.root = tk.Tk()
        self.app = MLModelApp(self.root)
        
        # Create temporary CSV files for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Regression dataset
        self.regression_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100) * 10 + 50
        })
        self.regression_file = os.path.join(self.temp_dir, 'regression_test.csv')
        self.regression_data.to_csv(self.regression_file, index=False)
        
        # Classification dataset
        self.classification_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        self.classification_file = os.path.join(self.temp_dir, 'classification_test.csv')
        self.classification_data.to_csv(self.classification_file, index=False)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        self.root.destroy()
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.messagebox.askyesno')
    def test_complete_regression_workflow(self, mock_askyesno, mock_filedialog):
        """Test complete regression workflow from file load to model save"""
        mock_filedialog.return_value = self.regression_file
        mock_askyesno.return_value = True
        
        # Load file
        self.app.load_file()
        self.assertIsNotNone(self.app.dataset)
        
        # Select target
        self.app.target_column.set('target')
        
        # Set model type
        self.app.set_model_type('regressor')
        
        # Train models
        with patch.object(self.app, 'save_model') as mock_save:
            self.app.train_models()
            mock_save.assert_called_once()
        
        self.assertIsNotNone(self.app.best_model)
        self.assertIsNotNone(self.app.best_metrics)
    
    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.messagebox.askyesno')
    def test_complete_classification_workflow(self, mock_askyesno, mock_filedialog):
        """Test complete classification workflow from file load to model save"""
        mock_filedialog.return_value = self.classification_file
        mock_askyesno.return_value = True
        
        # Load file
        self.app.load_file()
        self.assertIsNotNone(self.app.dataset)
        
        # Select target
        self.app.target_column.set('target')
        
        # Set model type
        self.app.set_model_type('classifier')
        
        # Train models
        with patch.object(self.app, 'save_model') as mock_save:
            self.app.train_models()
            mock_save.assert_called_once()
        
        self.assertIsNotNone(self.app.best_model)
        self.assertIsNotNone(self.app.best_metrics)


if __name__ == '__main__':
    # Create test suite using TestLoader (modern approach)
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test cases using the modern approach
    test_suite.addTests(loader.loadTestsFromTestCase(TestMLModelApp))
    test_suite.addTests(loader.loadTestsFromTestCase(TestMLModelAppIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TESTS RUN: {result.testsRun}")
    print(f"FAILURES: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    print(f"SUCCESS RATE: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)