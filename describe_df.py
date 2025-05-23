import pandas as pd

def describe_df(df):
    """
    Recibe un dataframe y devuelve otro dataframe con los nombres 
    de las columnas del dataframe transferido a la función. En filas contiene 
    los parámetros descriptivos del dataframe: 
    - Tipo de dato
    - % Nulos (Porcentaje de valores nulos)
    - Valores Únicos
    - % Cardinalidad (Relación de valores únicos con el total de registros)
    
    Argumentos:
    - df (DataFrame): DataFrame de trabajo
    
    Retorna:
    - DataFrame con los parámetros descriptivos
    """
    
    summary_df = pd.DataFrame(index=['Tipo', '% Nulos', 'Valores Únicos', '% Cardinalidad'])

    for column in df.columns:
        tipo = df[column].dtype
        porcentaje_nulos = df[column].isnull().mean() * 100
        valores_unicos = df[column].nunique()
        cardinalidad = (valores_unicos / len(df)) * 100

        summary_df[column] = [tipo, f"{porcentaje_nulos:.2f}%", valores_unicos, f"{cardinalidad:.2f}%"]

    return summary_df


