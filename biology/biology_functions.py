import pandas as pd
import numpy as np
from haversine import haversine


def formatear_fecha(fecha):
    if pd.notnull(fecha):
        try:
            return pd.to_datetime(fecha, format='%d/%m/%Y %H:%M').strftime('%d/%m/%Y %H:%M')
        except ValueError:
            return pd.to_datetime(fecha, format='mixed').strftime('%d/%m/%Y %H:%M')
    else:
        return fecha


def lat_long(lat_lon):
    # Dividir la cadena en grados, minutos y segundos
    lat_lon_split = lat_lon.str.split(" ", n=2, expand=True)

    # Convertir las partes a números y realizar cálculos
    lat_lon_final = lat_lon_split.apply(lambda x: pd.to_numeric(x.str.replace("[' °]*", "", regex=True)), axis=1)
    lat_lon_final[0] = -1 * lat_lon_final[0] - lat_lon_final[1] / 60 - lat_lon_final[2] / 3600

    return lat_lon_final[0]
  

def length_weight(length, a, b):
    length = pd.to_numeric(length)
    w = a * (length**b)
    return w

def ponderacion(data, tallas, captura_column, a, b):
    
    talla = data[tallas]
    catch = data[captura_column]

    peso = length_weight(tallas, a, b) * talla
    sum_pesos = peso.sum(axis=1, skipna=True)
    sum_pesos.replace(0, np.nan, inplace=True)

    fp = catch / (1000 * sum_pesos)

    resultados = pd.DataFrame((fp.values.reshape(-1, 1) * talla.values),
                              index=fp.index, columns=talla.columns)

    return resultados


def number_to_weight(data, tallas, a, b):
    talla = data[tallas]
    peso = length_weight(tallas, a, b) * talla

    return peso


def porc_juveniles(data, tallas_names=None, juv_lim=12):

    if tallas_names is None:
        raise ValueError("Se requiere especificar los nombres de las columnas de tallas.")
    
    total = data[tallas_names].sum(axis = 1, skipna=True)
    selected_columns = [col for col in tallas_names if float(col) < juv_lim]
    juv = data[selected_columns].sum(axis = 1, skipna = True)
    
    juv = juv*100/total

    return juv


def min_range(data):
    min_column_names = []
    for index, row in data.iterrows():
        # Encontrar el índice del primer valor no nulo en cada fila
        first_non_null_index = row.first_valid_index()
        
        # Obtener el nombre de la columna mínima
        min_column_name = data.columns[data.columns.get_loc(first_non_null_index)]
        min_column_names.append(min_column_name)
        
    return min_column_names
  
def max_range(data):
    min_column_names = []
    for index, row in data.iterrows():
        # Encontrar el índice del primer valor no nulo en cada fila
        first_non_null_index = row.last_valid_index()
        
        # Obtener el nombre de la columna mínima
        min_column_name = data.columns[data.columns.get_loc(first_non_null_index)]
        min_column_names.append(min_column_name)
        
    return min_column_names


def remove_high_row(row, prop=0.9):
    non_null_elements = row.dropna().values
    normalized_values = non_null_elements / np.sum(non_null_elements)

    return any(normalized_values > prop)


def distancia_costa(lat, lon, costa):
    lat_lon = np.column_stack((lat, lon))
    
    def calcular_distancia(punto):
        distancias_km = np.array([haversine(punto, (lat, lon)) for lat, lon in costa.values])
        distancia_minima_km = np.min(distancias_km)
        distancia_minima_mn = distancia_minima_km * 0.539957  # Conversión a millas náuticas
        return distancia_minima_mn
    
    distancias_minimas_mn = np.apply_along_axis(calcular_distancia, 1, lat_lon)
    
    return distancias_minimas_mn
  

def puntos_tierra(x, y, shoreline):
    x = pd.to_numeric(x, errors='coerce').round(3)
    y = pd.to_numeric(y, errors='coerce').round(3)
    resultados = []

    for i in range(len(y)):
        if pd.notna(x.iloc[i]) and pd.notna(y.iloc[i]):
            base_corr1 = shoreline[np.isclose(shoreline['Lat'].round(3), y.iloc[i])]
            base_corr2 = base_corr1.iloc[0, :].values if not base_corr1.empty else [np.nan, np.nan]

            base_corr00 = pd.DataFrame([base_corr2], columns=['LonL', 'LatL'])
            base_corr00['LonP'] = x.iloc[i]
            distancia = (base_corr00['LonP'].astype(float) - base_corr00['LonL'].astype(float)) * -1

            if distancia.iloc[0] < 0:
                resultados.append('tierra')
            else:
                resultados.append('ok')
        else:
            resultados.append(np.nan)

    return resultados


def estima_FP(data, catch, tallas, a, b):
  
    talla = data[tallas]
    catch = data[catch]
    peso = length_weight(tallas, a, b) * talla
    fp = catch / (1000 * peso.sum(axis=1, skipna=True))  
  
    return fp


def remover_extremos_altos(dataframe, col_tallas, condition_type='first', threshold=20):
    
    if condition_type not in ['first', 'last']:
        raise ValueError("El argumento 'condition_type' debe ser 'first' o 'last' ")

    condition = lambda row: row.dropna().to_list()[0] > threshold if not row.dropna().empty else False
    
    if condition_type == 'last':
        condition = lambda row: row.dropna().to_list()[-1] > threshold if not row.dropna().empty else False

    mask = dataframe[col_tallas].apply(condition, axis=1)
    dataframe.loc[mask, col_tallas] = np.nan
    return dataframe


def salto_indices_de_1(index):
    try:
        index = [float(i) for i in index]
    except ValueError:
        return False  
    return all(index[i] + 1 == index[i + 1] for i in range(len(index) - 1))


def salto_de_1_llenar_nan(dataframe, col_tallas):
    for index, row in dataframe.iterrows():
        selected_columns = row[col_tallas].dropna(axis=0).index
        if salto_indices_de_1(selected_columns):
            dataframe.loc[index, col_tallas] = np.nan
    return dataframe


def get_modes(row):
    modes_result = []

    if len(row.dropna()) >= 1:
        values = row.dropna().values
        xdif = np.diff(np.sign(np.diff(values)))
        modes_indices = values[np.where(xdif == -2)[0] + 1]
        modes_result.extend(row.index[np.isin(row.values, modes_indices)])

    return modes_result[0] if modes_result else np.nan


def calcular_porcentaje_incidental(data_total):
    # Buscar la columna que contiene 'catch_ANCHOVETA'
    anchoveta_column = data_total.filter(like='catch_ANCHOVETA').columns[0]

    # Filtrar las columnas que contienen 'catch_'
    catch_columns = data_total.filter(like='catch_')

    # Calcular la captura total
    total_catch = catch_columns.sum(axis=1, skipna=True)

    # Calcular la captura incidental excluyendo la columna de anchoveta
    other_catch = catch_columns.drop(columns=[anchoveta_column]).sum(axis=1, skipna=True)

    # Calcular el porcentaje incidental
    porcentaje_incidental = other_catch * 100 / total_catch

    # Crear una columna con los nombres de las especies incidentales
    data_total['nombre_especie_incidental'] = catch_columns.drop(columns=[anchoveta_column]).apply(
        lambda row: ', '.join([col.replace('catch_', '') for col in row.index[row.notna()].tolist()]), axis=1
    )

    # Crear una columna con el porcentaje incidental
    data_total['porcentaje_incidental'] = porcentaje_incidental

    return data_total

