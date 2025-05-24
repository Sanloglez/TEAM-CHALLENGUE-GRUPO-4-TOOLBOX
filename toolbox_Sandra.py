import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal, ttest_ind

def tipifica_variables(df, umbral_categoria=10, umbral_continua=0.2):
    """
    Clasifica las columnas de un DataFrame según su tipo de variable: Binaria, Categórica, Numérica Discreta o Continua.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    umbral_categoria (int): Máximo número de valores únicos para que una variable sea considerada categórica.
    umbral_continua (float): Porcentaje mínimo (sobre total de filas) para considerar una variable como continua.

    Devuelve:
    pd.DataFrame: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'.
    """
    resultado = []
    n = len(df)

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = round(cardinalidad / n, 2)

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo = "Numerica Continua"
        else:
            tipo = "Numerica Discreta"

        resultado.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultado)


def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve una lista de columnas categóricas que presentan una relación significativa
    con la variable numérica target_col usando t-test o ANOVA según corresponda.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo (numérica continua o discreta con alta cardinalidad).
    pvalue (float): Nivel de significación estadística (default = 0.05).

    Retorna:
    list or None: Lista de variables categóricas relacionadas, o None si hay error en los argumentos.
    """
    
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("❌ 'df' debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"❌ La columna '{target_col}' no está en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"❌ La columna '{target_col}' no es numérica.")
        return None

    if not (0 < pvalue < 1):
        print("❌ 'pvalue' debe estar entre 0 y 1.")
        return None

    cardinalidad = df[target_col].nunique()
    porcentaje = cardinalidad / len(df)

    if cardinalidad < 10 or porcentaje < 0.05:
        print(f"❌ La variable '{target_col}' no tiene suficiente cardinalidad para considerarse continua.")
        print(f"Cardinalidad única: {cardinalidad} ({round(porcentaje * 100, 2)}%)")
        return None

    # Selección de columnas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        print("⚠️ No hay variables categóricas en el DataFrame.")
        return []

    relacionadas = []

    for col in cat_cols:
        niveles = df[col].dropna().unique()
        grupos = [df[df[col] == nivel][target_col].dropna() for nivel in niveles]

        if any(len(grupo) < 2 for grupo in grupos):
            continue  # no hay suficientes datos en alguno de los grupos

        try:
            if len(niveles) == 2:
                stat, p = ttest_ind(*grupos)
            elif len(niveles) > 2:
                stat, p = f_oneway(*grupos)
            else:
                continue

            if p < pvalue:
                relacionadas.append(col)
        except Exception as e:
            print(f"⚠️ Error evaluando la columna '{col}': {e}")
            continue

    return relacionadas