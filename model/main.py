import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import pickle as pickle



def get_clean_data():
    data =pd.read_csv('data/yacht_hydro.csv')
    print(data.head())
    print(data.describe())
    return data

def create_model(data):
    X = data.drop(columns=['Rr'])
    y = data['Rr']
    
    #scale data range
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
   
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    # # Train and evaluate models
    # results = {}

    # for name, model in models.items():
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     results[name] = {
    #         'MAE': mean_absolute_error(y_test, y_pred),
    #         'MSE': mean_squared_error(y_test, y_pred),
    #         'R²': r2_score(y_test, y_pred)
    #     }

   
    #print(results)
    
    #select the best model : Random Forest
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print('Performance of the model \n')
    print('MAE:  ', mean_absolute_error(y_test, y_pred))
    print('MSE : ', mean_squared_error(y_test, y_pred))
    print('R² : ', r2_score(y_test, y_pred))
    
    return model, scaler



def main():
    
   # load and clean dataset before using
   data = get_clean_data()
   
   #create the model
   model, scaler  = create_model(data)
   
   #export model
   with open('model/model.pkl', 'wb') as f:
       pickle.dump(model, f)
   
   with open('model/scaler.pkl', 'wb') as f:
       pickle.dump(scaler, f)
       
if __name__ == '__main__':
    main()