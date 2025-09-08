import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Regression Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")


class MLModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Model Selector")
        self.root.geometry("600x600")

        # Title Label
        self.label_title = tk.Label(root, text="ML Model Selector", font=("Arial", 16, "bold"))
        self.label_title.pack(pady=10)

        # Buttons
        self.btn_load_file = tk.Button(root, text="Select CSV File", command=self.load_file)
        self.btn_load_file.pack(pady=5)

        self.label_file = tk.Label(root, text="No file selected", fg="gray")
        self.label_file.pack(pady=5)

        self.btn_select_target = tk.Button(root, text="Select Target Column", command=self.select_target)
        self.btn_select_target.pack(pady=5)

        self.target_column = tk.StringVar()
        self.target_dropdown = ttk.Combobox(root, textvariable=self.target_column, state="readonly")
        self.target_dropdown.pack(pady=5)

        # Model selection
        self.model_type = tk.StringVar()
        self.model_type.set("")  # Initially empty
        self.btn_choose_regressor = tk.Button(root, text="Choose Regression", command=lambda: self.set_model_type("regressor"))
        self.btn_choose_classifier = tk.Button(root, text="Choose Classification", command=lambda: self.set_model_type("classifier"))
        self.btn_choose_regressor.pack(pady=5)
        self.btn_choose_classifier.pack(pady=5)

        self.btn_train = tk.Button(root, text="Train Models", command=self.train_models, state=tk.DISABLED)
        self.btn_train.pack(pady=10)

        self.output_text = tk.Text(root, height=12, width=70)
        self.output_text.pack(pady=5)

        self.btn_save = tk.Button(root, text="Save Best Model", command=self.save_model, state=tk.DISABLED)
        self.btn_save.pack(pady=5)

        self.dataset = None
        self.best_model = None
        self.best_metrics = None  # Store best model's metrics

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.dataset = pd.read_csv(file_path)
            self.label_file.config(text=f"File loaded: {file_path}")
            self.target_dropdown['values'] = list(self.dataset.columns)

    def select_target(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        self.target_column.set("")
        self.target_dropdown['values'] = list(self.dataset.columns)

    def set_model_type(self, model_type):
        """Set user choice for regression or classification"""
        self.model_type.set(model_type)
        self.btn_train.config(state=tk.NORMAL)
        messagebox.showinfo("Model Type Selected", f"You selected: {model_type.capitalize()}")

    def validate_model_choice(self):
        """Ensure that regression models are trained on continuous targets, and classifiers on categorical targets."""
        target_col = self.target_column.get()
        unique_values = self.dataset[target_col].nunique()
        target_dtype = self.dataset[target_col].dtype

        if self.model_type.get() == "classifier":
            if unique_values > 10 and target_dtype != "object":
                messagebox.showerror("Error", "You selected Classification, but the target seems continuous. Please select Regression instead.")
                return False
        elif self.model_type.get() == "regressor":
            if unique_values <= 10 or target_dtype == "object":
                messagebox.showerror("Error", "You selected Regression, but the target seems categorical. Please select Classification instead.")
                return False

        return True

    def train_models(self):
        if self.dataset is None or not self.target_column.get():
            messagebox.showerror("Error", "Please select a dataset and target column.")
            return
        if not self.model_type.get():
            messagebox.showerror("Error", "Please choose Regression or Classification first.")
            return
        if not self.validate_model_choice():
            return

        target_col = self.target_column.get()
        X = self.dataset.drop(columns=[target_col])
        y = self.dataset[target_col]

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            self.output_text.insert(tk.END, "Handling missing values...\n")
            imputer = SimpleImputer(strategy='most_frequent')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            self.output_text.insert(tk.END, "Encoding categorical variables...\n")
            X = pd.get_dummies(X, columns=categorical_cols)

        # Scale data
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Convert classification target to numeric
        if self.model_type.get() == "classifier" and y.dtype == "object":
            y = y.astype("category").cat.codes

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.output_text.insert(tk.END, f"Training {self.model_type.get()} models...\n")
        self.root.update()

        best_score = float('-inf')
        self.best_model = None

        if self.model_type.get() == "regressor":
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "SVR": SVR()
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)

                self.output_text.insert(tk.END, f"{name} - R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}\n")

                if r2 > best_score:
                    best_score = r2
                    self.best_model = (name, model)
                    self.best_metrics = {"R2": r2, "MAE": mae, "MSE": mse}

        elif self.model_type.get() == "classifier":
            models = {
                "LogisticRegression": LogisticRegression(),
                "KNN": KNeighborsClassifier(),
                "SVC": SVC()
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                report = classification_report(y_test, predictions)

                self.output_text.insert(tk.END, f"{name} - Accuracy: {accuracy:.4f}\n")

                if accuracy > best_score:
                    best_score = accuracy
                    self.best_model = (name, model)
                    self.best_metrics = {"Accuracy": accuracy, "Report": report}

        if self.best_model:
            self.output_text.insert(tk.END, f"\nBest Model: {self.best_model[0]}\n")
            user_response = messagebox.askyesno("Best Model Confirmation", f"Do you agree with the best model: {self.best_model[0]}?")
            if user_response:
                self.save_model()

    def save_model(self):
        if self.best_model:
            file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
            if file_path:
                with open(file_path, 'wb') as f:
                    pickle.dump({"model": self.best_model[1], "metrics": self.best_metrics}, f)
                messagebox.showinfo("Success", f"Model '{self.best_model[0]}' saved with its metrics.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLModelApp(root)
    root.mainloop()
