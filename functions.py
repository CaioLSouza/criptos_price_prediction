# Libraries and modules

# standard libraries
import os
import time
from datetime import datetime
import itertools
from itertools import chain, combinations
import warnings
warnings.filterwarnings('ignore')

# data analysis and manipulation
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning and model selection
from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor

# statistical tools and analysis
from scipy.stats import pearsonr, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# APIs and requests
import requests
import json

import time



# Data visualization

def scatter_plots(df):
    columns = df.columns
    combinations = list(itertools.combinations(columns, 2))  
    for col1, col2 in combinations:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col1], y=df[col2])
        plt.title(f'Scatter Plot: {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        plt.show()

def plot_residuals(residuals_df):
    num_columns = len(residuals_df.columns)
    
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 6 * num_columns), sharex=True)
    
    for i, column in enumerate(residuals_df.columns):
        axes[i].plot(residuals_df.index, residuals_df[column], label=f'Resíduo - {column}')
        axes[i].axhline(0, color='black', linewidth=1)
        axes[i].set_ylabel('Resíduos')
        axes[i].set_title(f'Resíduos - {column}')
        axes[i].legend()
    
    axes[-1].set_xlabel('Tempo')  
    plt.tight_layout()  
    plt.show()

def plot_autocorrelation(df, n_lags, autocorrelation_function):

    if autocorrelation_function == "ACF":
        for column in df.columns:
            plot_acf(df[column], lags = n_lags)
            plt.title(f"ACF for {column}")
            plt.tight_layout() 
            
    elif autocorrelation_function == "PACF":
        for column in df.columns:
                plot_pacf(df[column], lags = n_lags)
                plt.title(f"PACF for {column}")
                plt.tight_layout()
            
    else:
        print("Invalid autocorrelation function. Please try 'ACF' or 'PACF'")



# Statistics and data analysis

def correlation_with_lags(target_series, series_dict, max_lag):
    correlations = {}
    
    # Iterar sobre cada série no dicionário
    for series_name, other_series in series_dict.items():
        
        # Dicionário para armazenar as correlações de uma única série
        series_correlations = {}
        
        for lag in range(max_lag + 1):
            # Defasagem da outra série
            shifted_series = other_series.shift(lag)
            
            # Calcular correlação entre a série alvo e a série defasada
            correlation = target_series.corr(shifted_series)
            
            # Armazenar a correlação com o nome da coluna correspondente ao lag
            series_correlations[f'corr_lag_{lag}'] = correlation
        
        # Armazenar as correlações dessa série
        correlations[series_name] = series_correlations
    
    # Converter o dicionário em DataFrame
    result_df = pd.DataFrame(correlations).T 
    
    return result_df

def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] > 0.05:
        print("A série não é estacionária (p > 0.05)")
    else:
        print("A série é estacionária (p <= 0.05)")


def shapiro_test(residuals_df):
    for column in residuals_df.columns:
        stat, p_value = shapiro(residuals_df[column])
        if p_value < 0.05:
            print(f"Variável: {column}, estatística: {stat}, p-value: {p_value}. Hipótese nula rejeitada. Os resíduos não são normalmente distribuídos")
        else: 
            print(f"Variável: {column}, estatística: {stat}, p-value: {p_value}. Hipótese nula não foi rejeitada. Os resíduos são normalmente distribuídos")


# Data preprocessing and manipulation

def split_train_test(X, y, train_size=0.8):
    split_index = int(len(X) * train_size)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

def convert_unix_milliseconds_to_date(unix_milliseconds):
    # Convert milliseconds to seconds
    unix_seconds = unix_milliseconds / 1000
    # Create a datetime object from the Unix timestamp
    dt = datetime.fromtimestamp(unix_seconds)
    # Format the datetime object to a string in day/month/year format
    return dt.strftime('%d/%m/%Y')


def convert_to_date(df, date_column, freq='h'):
    df = df.assign(date=list(map(lambda x: convert_unix_milliseconds_to_date(x), 
                                        df[date_column])))
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['hour'] = pd.date_range(start='00:00', periods=len(df), freq=freq).time
    df['date_hour'] = df['date'] + pd.to_timedelta(df.groupby('date').cumcount(), unit=freq)
    df.index = df['date_hour']
    df = df.drop('date_hour', axis=1)
    return df


def get_all_combinations(columns):
    return list(chain.from_iterable(combinations(columns, r) for r in range(1, len(columns) + 1)))


def add_lags(df, num_lags):
    
    lagged_df = df.copy()

    for lag in range(1, num_lags + 1):
        lagged_columns = {
            f"{col}_lag_{lag}": df[col].shift(lag)
            for col in df.columns
        }
        lagged_df = pd.concat([lagged_df, pd.DataFrame(lagged_columns)], axis=1)

    lagged_df.dropna(inplace=True)

    return lagged_df


# Model training

def one_step_ahead_forecasting_AR(y, starting_point_percent, lags):
    '''Faz previsão one-step-ahead utilizando modelo autoregressivo (AR) e um ponto de início baseado em porcentagem'''
    
    predictions = np.array([])
    print(f'Treinando modelo AR com {lags} lags')

    # Definindo o ponto de início como porcentagem dos dados
    starting_point = int(len(y) * starting_point_percent)
    
    # Dividindo o conjunto de treino e teste a partir do ponto de início
    y_train = y[:starting_point]
    y_test = y[starting_point:]

    # Criando o modelo autoregressivo inicial
    model = AutoReg(y_train, lags=lags)
    model_fit = model.fit()

    # Previsão one-step-ahead
    for i in range(len(y_test)):
        # Prevendo o próximo valor (one-step-ahead)
        pred = model_fit.predict(start=len(y_train), end=len(y_train), dynamic=False)
        predictions = np.append(predictions, pred)
        
        # Adiciona o valor real ao conjunto de treino
        y_train = np.append(y_train, y_test.iloc[i])
        
        # Ajusta o modelo novamente com o novo valor adicionado
        model = AutoReg(y_train, lags=lags)
        model_fit = model.fit()
    
    return predictions


def fit_ar_model(df, starting_point_percent, lags_list):
    '''Treina o modelo AR com uma lista de lags e retorna DataFrames de performance e resíduos'''
    
    performance_dict = {}
    residuals_dict = {}

    for lags in lags_list:
        print(f'Treinando para {lags} lags')
        
        for column in df.columns:
            print(f'Treinando com a variável: {column}')
            
            # Previsão com AR
            predictions = one_step_ahead_forecasting_AR(y=df[column], starting_point_percent=starting_point_percent, lags=lags)
            
            # Avaliando a performance
            actual = df[column].iloc[int(len(df[column]) * starting_point_percent):]
            mse, mae = evaluate_performance(predictions=predictions, actual=actual)
            performance_dict[(column, lags)] = [mse, mae]

            # Calculando os resíduos
            residuals = predictions - actual.values
            residuals_dict[(column, lags)] = residuals

            # Modelo Naive: previsão t = valor de t-1 para a variável
            predictions_naive = df[column].shift(1).iloc[int(len(df[column]) * starting_point_percent):]
            predictions_naive = predictions_naive.dropna()
            mse_naive, mae_naive = evaluate_performance(predictions=predictions_naive, actual=actual)
            performance_dict[('Naive_' + column, lags)] = [mse_naive, mae_naive]
    
    # Convertendo os dicionários de performance e resíduos para DataFrames
    performance_df = pd.DataFrame.from_dict(performance_dict, orient='index', columns=['MSE', 'MAE'])
    performance_df.index = pd.MultiIndex.from_tuples(performance_df.index, names=["Variable", "Lags"])
    
    # Resíduos: Cada variável terá seus resíduos para cada lag
    residuals_df = pd.DataFrame(residuals_dict)
    residuals_df.columns = pd.MultiIndex.from_tuples(residuals_df.columns, names=["Variable", "Lags"])
    residuals_df.index = df[column].iloc[int(len(df[column]) * starting_point_percent):].index[:len(residuals_df)]

    return performance_df, residuals_df

def one_step_ahead_forecasting_AR_exog(y, X, starting_point_percent, lags):
    """Faz previsão one-step-ahead utilizando modelo autoregressivo (AR) com variáveis exógenas e lags."""
    predictions = []
    print('Treinando modelo AR com variáveis exógenas e lags')
    
    # Definir o ponto de início
    starting_point = int(len(y) * starting_point_percent)
    
    # Dividindo o conjunto de treino e teste
    y_train = y[:starting_point].values  # Convertendo para numpy array
    y_test = y[starting_point:].values  # Convertendo para numpy array
    
    X_train = X.iloc[:starting_point].values  # Garantindo formato correto
    X_test = X.iloc[starting_point:].values  # Garantindo formato correto
    
    # Criar e ajustar o modelo inicial
    model = AutoReg(y_train, lags=lags, exog=X_train)
    model_fit = model.fit()
    print('Modelo ajustado')
    
    for i in range(len(y_test)):
        print(f'Predição passo {i+1}')
        
        if i >= len(X_test):  # Evita erro de indexação
            break
        
        exog_oos = X_test[i].reshape(1, -1)  # Garantir que exog seja 2D
        pred = model_fit.predict(start=len(y_train), end=len(y_train), exog_oos=exog_oos, dynamic=False)
        
        if pred.size > 0:
            predictions.append(pred[0])
        else:
            predictions.append(np.nan)  # Evita erro caso a previsão falhe
        
        # Atualizar conjunto de treino
        y_train = np.append(y_train, y_test[i])
        X_train = np.vstack([X_train, X_test[i]])
        
        # Ajustar modelo novamente
        model = AutoReg(y_train, lags=lags, exog=X_train)
        model_fit = model.fit()
    
    return np.array(predictions)

def fit_ar_model_exog(df, target_column, starting_point_percent, lags_list):
    """Treina o modelo AR com variáveis exógenas para diferentes lags e retorna DataFrames de performance e resíduos."""
    performance_dict = {}
    residuals_dict = {}
    
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    # Gerar todas as combinações possíveis das variáveis exógenas
    exog_combinations = get_all_combinations(X.columns)
    
    for lags in lags_list:
        print(f'Treinando para {lags} lags')
        
        for combo in exog_combinations:
            print(f'Treinando com variáveis exógenas: {combo}')
            
            X_subset = X[list(combo)]  # Seleciona as colunas da combinação
            
            predictions = one_step_ahead_forecasting_AR_exog(y=y, X=X_subset, starting_point_percent=starting_point_percent, lags=lags)
            
            actual = y.iloc[int(len(y) * starting_point_percent):].values
            actual = actual[:len(predictions)]  # Garantir tamanhos compatíveis
            
            if len(predictions) != len(actual):
                print(f"Inconsistência de tamanho! Predictions: {len(predictions)}, Actual: {len(actual)}")
                continue
            
            mse, mae = evaluate_performance(predictions=predictions, actual=actual)
            performance_dict[(combo, lags)] = [mse, mae]
            
            residuals = predictions - actual
            residuals_dict[(combo, lags)] = residuals
            
            # Modelo Naive
            predictions_naive = y.shift(1).iloc[int(len(y) * starting_point_percent):].dropna().values
            predictions_naive = predictions_naive[:len(actual)]
            
            if len(predictions_naive) != len(actual):
                continue
            
            mse_naive, mae_naive = evaluate_performance(predictions=predictions_naive, actual=actual)
            performance_dict[('Naive_' + '_'.join(combo), lags)] = [mse_naive, mae_naive]
    
    # Convertendo os dicionários de performance e resíduos para DataFrames
    performance_df = pd.DataFrame.from_dict(performance_dict, orient='index', columns=['MSE', 'MAE'])
    performance_df.index = pd.MultiIndex.from_tuples(performance_df.index, names=["Variables", "Lags"])
    
    residuals_df = pd.DataFrame(residuals_dict)
    residuals_df.columns = pd.MultiIndex.from_tuples(residuals_df.columns, names=["Variables", "Lags"])
    residuals_df.index = y.iloc[int(len(y) * starting_point_percent):].index[:len(residuals_df)]
    
    return performance_df, residuals_df


# Model validation

def time_series_cross_validation(X_train, y_train, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []

    # Iniciar o contador de tempo
    start_time = time.time()

    for train_index, val_index in tscv.split(X_train):
        X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_t, y_t)
        predictions = model.predict(X_val)

        mse = mean_squared_error(y_val, predictions)
        errors.append(mse)

    # Finalizar o contador de tempo
    end_time = time.time()
    total_time = end_time - start_time

    # Exibir o tempo total de treinamento
    print(f"Total training time: {total_time:.2f} seconds")

    return np.mean(errors)

def hyperparameter_optimization(X_train, y_train, model, param_grid):
    results = []
    for params in ParameterGrid(param_grid):
        print(f"Testando a combinação {params}")
        model.set_params(**params)
        mse = time_series_cross_validation(X_train, y_train, model, n_splits=5)

        result = params.copy()
        result["mse"] = mse
        results.append(result)
    return pd.DataFrame(results)


def final_model_evaluation(X_train, y_train, X_test, y_test, model, best_params):
    """
    Trains the model on the training data and evaluates on the test set.
    Filters out invalid hyperparameters before setting them.
    """
    # Get valid hyperparameters for the model
    valid_params = model.get_params().keys()
    
    # Filter only valid hyperparameters
    filtered_params = {k: v for k, v in best_params.items() if k in valid_params}

    # Set the valid hyperparameters
    model.set_params(**filtered_params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate MSE
    final_mse = mean_squared_error(y_test, predictions)
    
    return final_mse, predictions


def optimize_and_evaluate(X, y, models_and_params, train_size=0.8):
    X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=0.8)
    all_results = pd.DataFrame()

    for model_name, model_info in models_and_params.items():
        print(f"Otimizando {model_name}...")
        
        # Medir o tempo de início do treinamento
        start_time = time.time()

        # Otimização dos hiperparâmetros
        results_df = hyperparameter_optimization(
            X_train, y_train, model_info["model"], model_info["params"]
        )
        
        # Tempo de treinamento
        training_time = time.time() - start_time
        print(f"Tempo de treinamento para {model_name}: {training_time:.2f} segundos")
        
        results_df["model"] = model_name
        results_df["training_time"] = training_time  # Adicionando a coluna de tempo de treinamento

        best_params = results_df.sort_values(by="mse").iloc[0].drop("mse").to_dict()
        
        # Avaliação final com os melhores parâmetros
        final_mse, predictions = final_model_evaluation(
            X_train, y_train, X_test, y_test, model_info["model"], best_params
        )

        print(f"{model_name} - MSE Final no Teste: {final_mse:.4f}")
        results_df["final_test_mse"] = final_mse
        all_results = pd.concat([all_results, results_df], ignore_index=True)

    return all_results


def optimize_and_evaluate_one_lag(df, models_and_params):
    all_results = pd.DataFrame()

    for column in df.columns:
        print(f"\n🔍 Processing column: {column}")

        # Create lagged feature (lag 1)
        lagged_df = pd.DataFrame({
            f"{column}_lag_1": df[column].shift(1),
            column: df[column]
        }).dropna()

        # Define X and y
        X = lagged_df[[f"{column}_lag_1"]]
        y = lagged_df[column]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=0.8)

        # Apply models to each column
        for model_name, model_info in models_and_params.items():
            print(f"⚙️ Optimizing model: {model_name} for {column}")

            # Hyperparameter optimization
            results_df = hyperparameter_optimization(
                X_train, y_train, model_info["model"], model_info["params"]
            )
            results_df["model"] = model_name
            results_df["series"] = column

            # Best hyperparameters
            best_params = results_df.sort_values(by="mse").iloc[0].drop("mse").to_dict()

            # Final evaluation
            final_mse, predictions = final_model_evaluation(
                X_train, y_train, X_test, y_test, model_info["model"], best_params
            )

            print(f"✅ {model_name} for {column} - Final MSE on Test: {final_mse:.4f}")

            # Store results
            results_df["final_test_mse"] = final_mse
            all_results = pd.concat([all_results, results_df], ignore_index=True)

    return all_results


def evaluate_all_feature_combinations(df, target, exogenous_vars, model, param_grid):
    results = pd.DataFrame()

    # Generate all possible combinations of exogenous variables + lagged target
    all_features = [f"{target}_lag_1"] + exogenous_vars
    feature_combinations = []
    for i in range(1, len(all_features) + 1):
        feature_combinations.extend(combinations(all_features, i))

    # Loop through all feature combinations
    for feature_set in feature_combinations:
        print(f"\n🔍 Evaluating feature set: {feature_set}")
        X = df[list(feature_set)]
        y = df[target]
        
        # Drop rows with NaNs caused by lags
        data = pd.concat([X, y], axis=1).dropna()
        X = data[list(feature_set)]
        y = data[target]

        # Train-test split
        X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=0.8)

        # Optimize hyperparameters for the current feature set
        results_df = hyperparameter_optimization(X_train, y_train, model, param_grid)
        results_df["features"] = [feature_set] * len(results_df)

        # Get best hyperparameters based on MSE
        best_params = results_df.sort_values(by="mse").iloc[0].drop("mse").to_dict()
        final_mse, predictions = final_model_evaluation(
            X_train, y_train, X_test, y_test, model, best_params
        )

        print(f"✅ Features: {feature_set} - Final MSE on Test: {final_mse:.4f}")
        results_df["final_test_mse"] = final_mse

        # Store the results
        results = pd.concat([results, results_df], ignore_index=True)

    return results


def final_model_evaluation(X_train, y_train, X_test, y_test, model, best_params):
    valid_params = model.get_params().keys()
    
    # Filter valid hyperparameters
    filtered_params = {k: v for k, v in best_params.items() if k in valid_params}

    # Convert floats to integers if necessary
    integer_params = [
        'max_depth', 'max_leaf_nodes', 'min_samples_split', 'min_samples_leaf', 'n_estimators'
    ]
    for param in integer_params:
        if param in filtered_params and isinstance(filtered_params[param], float):
            filtered_params[param] = int(filtered_params[param])

    # Train the model
    model.set_params(**filtered_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    final_mse = mean_squared_error(y_test, predictions)

    return final_mse, predictions


def evaluate_performance(predictions, actual):
    '''Evaluates model's performance using MSE, MAE e R²'''
    
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    
    return [mse, mae]




