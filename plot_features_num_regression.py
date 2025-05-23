import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Genera pairplots con variables numéricas del dataframe que cumplan ciertas condiciones de correlación con target_col.

    Parámetros:
    - df (DataFrame): DataFrame de entrada.
    - target_col (str): Columna objetivo para el análisis de correlación.
    - columns (list): Lista de columnas a considerar; si está vacía, se tomarán todas las numéricas del dataframe.
    - umbral_corr (float): Umbral mínimo absoluto de correlación para incluir variables.
    - pvalue (float or None): Nivel de significación estadística para el test de correlación. Si es None, no se aplica.

    Retorna:
    - Lista de columnas seleccionadas según las condiciones.
    """

    # Validación de target_col (debe existir enel dataframe)
    if target_col not in df.columns:
        raise ValueError(f"La columna target_col '{target_col}' no existe en el dataframe.")

    # Si columns está vacío, tomamos todas las variables numéricas excepto target_col
    if not columns:
        columns = df.select_dtypes(include=np.number).columns.tolist() #añado todas las columnas numéricas a la lista de columnas vacía
        columns.remove(target_col) #le quito el target
    #si hay columnas en parámetros tomará esas

    # Filtrar columnas por correlación
    selected_columns = [] #creo una lista vacía de columnas seleccionadas que iré rellenando
    for col in columns: #recorro las columnas de columns
        if col == target_col: #si la col es el target me la salto
            continue
        corr = df[[target_col, col]].dropna().corr().iloc[0, 1]  # tomo las columnas target_col y col del dataframe, elimina los NaN y calcula la matriz de correlación y extrae el valor con el iloc
        
        if abs(corr) > umbral_corr: #si la correlación en valor absoluto es mayor del umbral
            # Si se especifica pvalue, verificar significación estadística
            if pvalue is not None: #si el pvalue no es None
                _, pval = pearsonr(df[target_col].dropna(), df[col].dropna()) #calculo la correlación entre target_col y col y devuelve el vp_val porque el corr_coef no me hace falta
                if pval < 1 - pvalue: #si la probabilidad pval de que la correlación ocurra al azar es menor de 1-pvalue
                    selected_columns.append(col) #es estadísticamente signifcativa y lo meto en la lista
            else:
                selected_columns.append(col) # si no hay pvalue agrega la columna a la lista para verificarla

    # Graficar en grupos de máximo 5 columnas por gráfico
    if selected_columns: #si selected_columns no está vacía
        for i in range(0, len(selected_columns), 4):  # Genero números e 0 a la longitud de selected_columns de 4 en 4. Máximo 5 con target_col, proceso 4 columnas de cada iteración
            subset = [target_col] + selected_columns[i:i+4] #creo este subset que tiene el target y las 4 columnas
            sns.pairplot(df[subset].dropna(), diag_kind='kde') #hago el pairplot habiendo eliminado las filas con Nan con el dropna
            plt.show() #lo muestro

    return selected_columns #devuelvo las columnas que superaron el filtro de correlación y significancia

# Ejemplo de uso:
data = {
    'target': [1, 2, 3, 4, 5, 6, 7],
    'A': [2, 4, 6, 8, 10, 12, 14],
    'B': [1, 3, 3, 5, 5, 7, 7],
    'C': [5, 4, 3, 2, 1, 0, -1],
    'D': [10, 20, 30, 40, 50, 60, 70]
}
df = pd.DataFrame(data)

result = plot_features_num_regression(df, target_col="target", umbral_corr=0.5, pvalue=0.05)
print("Columnas seleccionadas:", result)

