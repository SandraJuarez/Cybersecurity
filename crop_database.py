import pandas as pd
import random

# Cargar el archivo CSV
df = pd.read_csv('benign.csv')
print('len original',len(df))

# Obtener el número total de bloques de 25 renglones
total_blocks = len(df) // 25

# Lista para almacenar las muestras de bloques
sampled_blocks = []


# Iterar sobre los bloques de 25 renglones y agregarlos a la lista de muestras
for i in range(total_blocks):
    start_index = i * 25
    end_index = start_index + 25
    block = df.iloc[start_index:end_index]
    sampled_blocks.append(block)

# Revolver los bloques
random.shuffle(sampled_blocks)

# Obtener el 20% de las muestras
sample_size = int(total_blocks * 0.1)
sampled_data = sampled_blocks[:sample_size]

# Concatenar los bloques de muestra en un DataFrame
sampled_df = pd.concat(sampled_data)

print('cropped_dataset size',len(sampled_df))

# Ahora 'sampled_df' contiene una muestra del 20% del total de bloques de datos de 25 renglones
sampled_df.to_csv('benign_sample.csv', index=False)
