import customtkinter as ctk
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('new_datasetcsv.csv')

df = pd.DataFrame(data)

categorical_cols = ['City', 'Type']
numeric_cols = ['Year', 'Population']
target_col = 'Crime Rate'

preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_cols)],remainder='passthrough')

model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])

X = df[categorical_cols + numeric_cols]
y = df[target_col]

model.fit(X, y)

class CrimeRatePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Crime Rate Predictor")
        self.dataset = pd.read_csv('new_datasetcsv.csv')
        self.df = pd.DataFrame(self.dataset)

        ctk.set_appearance_mode("light")  
        ctk.set_default_color_theme("blue") 

        ctk.CTkLabel(root, text="City:").grid(row=0, column=0, padx=10, pady=10)
        self.city = ctk.CTkComboBox(root, values=sorted(self.df['City'].unique()))
        self.city.grid(row=0, column=1, padx=10, pady=10)

        ctk.CTkLabel(root, text="Type:").grid(row=1, column=0, padx=10, pady=10)
        self.type = ctk.CTkComboBox(root, values=sorted(self.df['Type'].unique()))
        self.type.grid(row=1, column=1, padx=10, pady=10)

        ctk.CTkLabel(root, text="Year:").grid(row=2, column=0, padx=10, pady=10)
        self.year = ctk.CTkEntry(root)
        self.year.grid(row=2, column=1, padx=10, pady=10)

        ctk.CTkLabel(root, text="Population:").grid(row=3, column=0, padx=10, pady=10)
        self.population = ctk.CTkEntry(root)
        self.population.grid(row=3, column=1, padx=10, pady=10)

        self.predict_button = ctk.CTkButton(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.result = ctk.CTkLabel(root, text="")
        self.result.grid(row=5, column=0, columnspan=2, pady=10)

    def predict(self):
        new_data = {
            'City': [self.city.get()],
            'Type': [self.type.get()],
            'Year': [int(self.year.get())],
            'Population': [int(self.population.get())]
        }

        new_df = pd.DataFrame(new_data)
        prediction = model.predict(new_df)
        self.result.configure(text=f'Prediction: {prediction[0]:.2f}')

root = ctk.CTk()
app = CrimeRatePredictor(root)
root.mainloop()
