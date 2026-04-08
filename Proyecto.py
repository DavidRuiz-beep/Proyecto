import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

ruta_base = r"C:\Users\kfeli\Documents\Universidad Externado\8vo SEMESTRE\Inteligencia artificial\Proyecto\GEIH_2023"
archivo = os.path.join(ruta_base, "GEIH_2023_limpia_hogares.csv")

# Crear carpeta para resultados
os.makedirs(os.path.join(ruta_base, "EDA_resultados"), exist_ok=True)

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA) - GEIH 2023")
print("="*80)

# 1. CARGA DE DATOS
print("\n1. CARGANDO DATOS...")
df = pd.read_csv(archivo)
print(f"Base cargada: {len(df):,} hogares")
print(f"Variables: {len(df.columns)}")

# 2. VISIÓN GENERAL DE LOS DATOS
print("\n2. VISIÓN GENERAL DE LOS DATOS")
print("-" * 50)

print("\nPrimeras 5 filas:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes.value_counts())

print("\nInformación general:")
print(f"   Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"   Valores nulos totales: {df.isnull().sum().sum():,}")
print(f"   Filas duplicadas: {df.duplicated().sum()}")

# 3. ANÁLISIS DE VALORES NULOS
print("\n3. ANÁLISIS DE VALORES NULOS")
print("-" * 50)

nulos = df.isnull().sum()
nulos = nulos[nulos > 0].sort_values(ascending=False)

if len(nulos) > 0:
    print("\nVariables con valores nulos:")
    for var, count in nulos.items():
        pct = (count / len(df)) * 100
        print(f"   • {var}: {count:,} ({pct:.1f}%)")
    
    # Gráfico de nulos
    plt.figure(figsize=(10, 6))
    nulos.plot(kind='bar', color='coral')
    plt.title('Valores Nulos por Variable', fontsize=14, fontweight='bold')
    plt.xlabel('Variables')
    plt.ylabel('Cantidad de Valores Nulos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_base, "EDA_resultados", "valores_nulos.png"), dpi=300)
    plt.close()
    print("Gráfico guardado: valores_nulos.png")
else:
    print("No hay valores nulos en la base")

# 4. ESTADÍSTICAS DESCRIPTIVAS
print("\n4. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 50)

# Variables numéricas
vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nVariables numéricas ({len(vars_numericas)}):")
print(df[vars_numericas].describe().round(2))

# Variables categóricas
vars_categoricas = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nVariables categóricas ({len(vars_categoricas)}):")
for var in vars_categoricas[:5]:
    print(f"\n   • {var}:")
    print(f"     {df[var].value_counts().head(3)}")

# 5. DISTRIBUCIÓN DE VARIABLES CLAVE
print("\n5. DISTRIBUCIÓN DE VARIABLES CLAVE")
print("-" * 50)

# 5.1 Ingreso per cápita
ingresos_positivos = df[df['ingreso_per_capita'] > 0]['ingreso_per_capita']
print(f"\nINGRESO PER CÁPITA:")
print(f"   Media: ${ingresos_positivos.mean():,.0f}")
print(f"   Mediana: ${ingresos_positivos.median():,.0f}")
print(f"   Desv. estándar: ${ingresos_positivos.std():,.0f}")
print(f"   Mínimo: ${ingresos_positivos.min():,.0f}")
print(f"   Máximo: ${ingresos_positivos.max():,.0f}")
print(f"   Hogares sin ingreso: {(df['ingreso_per_capita'] == 0).sum():,} ({(df['ingreso_per_capita'] == 0).mean()*100:.1f}%)")

# Histograma de ingresos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Escala normal
axes[0].hist(ingresos_positivos, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Ingreso Per Cápita ($)', fontsize=11)
axes[0].set_ylabel('Número de Hogares', fontsize=11)
axes[0].set_title('Distribución del Ingreso (Escala Normal)', fontsize=12, fontweight='bold')
axes[0].axvline(ingresos_positivos.median(), color='red', linestyle='--', label=f'Mediana: ${ingresos_positivos.median():,.0f}')
axes[0].axvline(ingresos_positivos.mean(), color='green', linestyle='--', label=f'Media: ${ingresos_positivos.mean():,.0f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Escala logarítmica
axes[1].hist(ingresos_positivos, bins=50, edgecolor='black', alpha=0.7, color='steelblue', log=True)
axes[1].set_xlabel('Ingreso Per Cápita ($)', fontsize=11)
axes[1].set_ylabel('Número de Hogares (escala log)', fontsize=11)
axes[1].set_title('Distribución del Ingreso (Escala Logarítmica)', fontsize=12, fontweight='bold')
axes[1].axvline(ingresos_positivos.median(), color='red', linestyle='--')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "EDA_resultados", "distribucion_ingreso.png"), dpi=300)
plt.close()
print("Gráfico guardado: distribucion_ingreso.png")

# 5.2 Número de personas por hogar
print(f"\nNÚMERO DE PERSONAS POR HOGAR:")
print(f"   Media: {df['num_personas'].mean():.2f}")
print(f"   Mediana: {df['num_personas'].median():.0f}")
print(f"   Moda: {df['num_personas'].mode().iloc[0]:.0f}")
print(f"   Mínimo: {df['num_personas'].min():.0f}")
print(f"   Máximo: {df['num_personas'].max():.0f}")

# Distribución de tamaño de hogar
fig, ax = plt.subplots(figsize=(10, 6))
tamano_counts = df['num_personas'].value_counts().sort_index()
ax.bar(tamano_counts.index, tamano_counts.values, color='mediumseagreen', edgecolor='black')
ax.set_xlabel('Número de Personas por Hogar', fontsize=12)
ax.set_ylabel('Número de Hogares', fontsize=12)
ax.set_title('Distribución del Tamaño del Hogar', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Agregar etiquetas
for i, (x, y) in enumerate(tamano_counts.items()):
    ax.text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "EDA_resultados", "distribucion_tamano_hogar.png"), dpi=300)
plt.close()
print("Gráfico guardado: distribucion_tamano_hogar.png")

# 5.3 Acceso a servicios públicos
print(f"\nACCESO A SERVICIOS PÚBLICOS:")
print(f"   Índice promedio: {df['acceso_servicios'].mean():.2f}/5")
print(f"   Hogares con acceso total (5/5): {(df['acceso_servicios'] == 5).sum():,} ({(df['acceso_servicios'] == 5).mean()*100:.1f}%)")
print(f"   Hogares sin servicios (0/5): {(df['acceso_servicios'] == 0).sum():,} ({(df['acceso_servicios'] == 0).mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución de servicios
servicios_dist = df['acceso_servicios'].value_counts().sort_index()
axes[0].bar(servicios_dist.index, servicios_dist.values, color='cornflowerblue', edgecolor='black')
axes[0].set_xlabel('Número de Servicios', fontsize=11)
axes[0].set_ylabel('Número de Hogares', fontsize=11)
axes[0].set_title('Distribución del Acceso a Servicios', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Servicios específicos
servicios_especificos = ['energia_electrica', 'gas_natural', 'alcantarillado', 'recoleccion_basuras', 'acueducto']
servicios_pct = [df[s].mean() * 100 for s in servicios_especificos if s in df.columns]
nombres_servicios = [s.replace('_', ' ').title() for s in servicios_especificos if s in df.columns]

axes[1].barh(nombres_servicios, servicios_pct, color='lightcoral', edgecolor='black')
axes[1].set_xlabel('Porcentaje de Hogares (%)', fontsize=11)
axes[1].set_title('Acceso por Tipo de Servicio', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

for i, pct in enumerate(servicios_pct):
    axes[1].text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "EDA_resultados", "acceso_servicios.png"), dpi=300)
plt.close()
print("Gráfico guardado: acceso_servicios.png")

# 6. ANÁLISIS POR NIVEL EDUCATIVO
print("\n6. ANÁLISIS POR NIVEL EDUCATIVO DEL JEFE")
print("-" * 50)

# Filtrar solo hogares con ingreso > 0 para este análisis
df_con_ingreso = df[df['ingreso_per_capita'] > 0]

educacion_stats = df_con_ingreso.groupby('nivel_educativo').agg({
    'ingreso_per_capita': 'median',
    'num_personas': 'mean',
    'acceso_servicios': 'mean',
    'hogar_con_desempleo': 'mean'
}).round(2)

educacion_stats.columns = ['Ingreso_Mediano', 'Prom_Personas', 'Prom_Servicios', 'Tasa_Desempleo']
educacion_stats['Tasa_Desempleo'] = educacion_stats['Tasa_Desempleo'] * 100
print("\nEstadísticas por nivel educativo (solo hogares con ingreso):")
print(educacion_stats)

# Gráfico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
educacion_stats_sorted = educacion_stats.sort_values('Ingreso_Mediano', ascending=False)

# Ingreso por educación
educacion_stats_sorted['Ingreso_Mediano'].plot(kind='bar', ax=axes[0,0], color='coral')
axes[0,0].set_title('Ingreso Mediano por Nivel Educativo', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Ingreso Per Cápita ($)')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

# Tamaño hogar por educación
educacion_stats_sorted['Prom_Personas'].plot(kind='bar', ax=axes[0,1], color='mediumseagreen')
axes[0,1].set_title('Tamaño Promedio del Hogar', fontsize=12, fontweight='bold')
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Personas por Hogar')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# Servicios por educación
educacion_stats_sorted['Prom_Servicios'].plot(kind='bar', ax=axes[1,0], color='cornflowerblue')
axes[1,0].set_title('Acceso a Servicios Públicos', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('')
axes[1,0].set_ylabel('Número de Servicios (0-5)')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# Desempleo por educación
educacion_stats_sorted['Tasa_Desempleo'].plot(kind='bar', ax=axes[1,1], color='tomato')
axes[1,1].set_title('Tasa de Desempleo en el Hogar', fontsize=12, fontweight='bold')
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('Hogares con Desempleo (%)')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "EDA_resultados", "analisis_por_educacion.png"), dpi=300)
plt.close()
print("Gráfico guardado: analisis_por_educacion.png")

# 7. MATRIZ DE CORRELACIÓN
print("\n7. MATRIZ DE CORRELACIÓN")
print("-" * 50)

vars_corr = ['num_personas', 'ingreso_per_capita', 'acceso_servicios', 
             'num_ocupados', 'num_desocupados', 'tasa_desempleo_hogar']
vars_existentes = [v for v in vars_corr if v in df.columns]

# Hogares con ingreso > 0 para correlaciones
df_corr = df[df['ingreso_per_capita'] > 0]
corr_matrix = df_corr[vars_existentes].corr()

print("\nCorrelaciones principales con ingreso per cápita:")
corr_con_ingreso = corr_matrix['ingreso_per_capita'].sort_values(ascending=False)
for var, corr in corr_con_ingreso.items():
    print(f"   • {var}: {corr:.3f}")

# Mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación entre Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "EDA_resultados", "matriz_correlacion.png"), dpi=300)
plt.close()
print("Gráfico guardado: matriz_correlacion.png")

# 8. ANÁLISIS DE CONDICIONES DE VIVIENDA
print("\n8. ANÁLISIS DE CONDICIONES DE VIVIENDA")
print("-" * 50)

if 'tipo_vivienda' in df.columns:
    print("\nTipo de vivienda:")
    tipo_counts = df['tipo_vivienda'].value_counts()
    for tipo, count in tipo_counts.items():
        print(f"   • {tipo}: {count:,} ({count/len(df)*100:.1f}%)")

if 'material_paredes' in df.columns:
    print("\nMaterial de paredes:")
    paredes_counts = df['material_paredes'].value_counts().head(5)
    for material, count in paredes_counts.items():
        print(f"   • {material}: {count:,} ({count/len(df)*100:.1f}%)")

if 'material_pisos' in df.columns:
    print("\nMaterial de pisos:")
    pisos_counts = df['material_pisos'].value_counts().head(5)
    for material, count in pisos_counts.items():
        print(f"   • {material}: {count:,} ({count/len(df)*100:.1f}%)")

# Gráfico de condiciones de vivienda
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

if 'tipo_vivienda' in df.columns:
    df['tipo_vivienda'].value_counts().plot(kind='bar', ax=axes[0], color='lightblue', edgecolor='black')
    axes[0].set_title('Tipo de Vivienda', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

if 'material_paredes' in df.columns:
    df['material_paredes'].value_counts().head(5).plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].set_title('Material de Paredes (Top 5)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

if 'material_pisos' in df.columns:
    df['material_pisos'].value_counts().head(5).plot(kind='bar', ax=axes[2], color='lightsalmon', edgecolor='black')
    axes[2].set_title('Material de Pisos (Top 5)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "EDA_resultados", "condiciones_vivienda.png"), dpi=300)
plt.close()
print("Gráfico guardado: condiciones_vivienda.png")

# 9. ANÁLISIS SITUACIÓN LABORAL
print("\n9. ANÁLISIS DE SITUACIÓN LABORAL")
print("-" * 50)

print(f"\nEstadísticas laborales:")
print(f"   Hogares con al menos un ocupado: {(df['num_ocupados'] > 0).sum():,} ({(df['num_ocupados'] > 0).mean()*100:.1f}%)")
print(f"   Hogares sin ocupados: {(df['num_ocupados'] == 0).sum():,} ({(df['num_ocupados'] == 0).mean()*100:.1f}%)")
print(f"   Hogares con desempleo: {df['hogar_con_desempleo'].sum():,} ({df['hogar_con_desempleo'].mean()*100:.1f}%)")
print(f"   Promedio de ocupados por hogar: {df['num_ocupados'].mean():.2f}")
print(f"   Promedio de desocupados por hogar: {df['num_desocupados'].mean():.2f}")

# Gráfico de situación laboral
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución de ocupados
ocupados_dist = df['num_ocupados'].value_counts().sort_index().head(10)
axes[0].bar(ocupados_dist.index, ocupados_dist.values, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Número de Ocupados en el Hogar', fontsize=11)
axes[0].set_ylabel('Número de Hogares', fontsize=11)
axes[0].set_title('Distribución de Ocupados por Hogar', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Tasa de desempleo por rangos de ingreso (solo hogares con ingreso)
df_con_ingreso = df[df['ingreso_per_capita'] > 0].copy()
if len(df_con_ingreso) > 0:
    try:
        df_con_ingreso['rango_ingreso'] = pd.qcut(df_con_ingreso['ingreso_per_capita'], 
                                                   q=4, labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])
        desempleo_por_ingreso = df_con_ingreso.groupby('rango_ingreso')['hogar_con_desempleo'].mean() * 100
        axes[1].bar(range(len(desempleo_por_ingreso)), desempleo_por_ingreso.values, color='tomato', edgecolor='black')
        axes[1].set_xticks(range(len(desempleo_por_ingreso)))
        axes[1].set_xticklabels(desempleo_por_ingreso.index)
        axes[1].set_xlabel('Rango de Ingreso Per Cápita', fontsize=11)
        axes[1].set_ylabel('Hogares con Desempleo (%)', fontsize=11)
        axes[1].set_title('Incidencia del Desempleo por Nivel de Ingreso', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        for i, (rango, pct) in enumerate(desempleo_por_ingreso.items()):
            axes[1].text(i, pct + 0.5, f'{pct:.1f}%', ha='center', fontsize=10)
    except Exception as e:
        print(f"No se pudo generar el gráfico de desempleo por ingreso: {e}")

plt.tight_layout()
plt.savefig(os.path.join(ruta_base, "EDA_resultados", "situacion_laboral.png"), dpi=300)
plt.close()
print("Gráfico guardado: situacion_laboral.png")

# 10. ANÁLISIS RANGOS DE INGRESO
print("\n10. ANÁLISIS POR RANGOS DE INGRESO")
print("-" * 50)

# Crear rangos de ingreso solo con hogares que tienen ingreso
df_con_ingreso = df[df['ingreso_per_capita'] > 0].copy()
if len(df_con_ingreso) > 0:
    df_con_ingreso['rango_ingreso'] = pd.cut(df_con_ingreso['ingreso_per_capita'], 
                                              bins=[0, 200000, 400000, 600000, 1000000, float('inf')],
                                              labels=['Muy bajo', 'Bajo', 'Medio', 'Medio-Alto', 'Alto'])
    
    ingreso_analysis = df_con_ingreso.groupby('rango_ingreso').agg({
        'num_personas': 'mean',
        'acceso_servicios': 'mean',
        'num_ocupados': 'mean',
        'hogar_con_desempleo': 'mean'
    }).round(2)
    
    ingreso_analysis['hogar_con_desempleo'] = ingreso_analysis['hogar_con_desempleo'] * 100
    ingreso_analysis.columns = ['Personas_Hogar', 'Servicios', 'Ocupados', 'Desempleo_%']
    
    print("\nCaracterísticas por rango de ingreso:")
    print(ingreso_analysis)

# 11. FINAL
print("\n" + "="*80)
print("REPORTE FINAL DEL EDA")
print("="*80)

# Estadísticas adicionales
hogares_sin_ingreso = (df['ingreso_per_capita'] == 0).sum()
pct_sin_ingreso = (hogares_sin_ingreso / len(df)) * 100

nivel_superior = df['nivel_educativo'].isin(['Superior universitaria', 'Posgrado', 'Superior no universitaria'])
pct_superior = nivel_superior.mean() * 100

servicios_4_mas = (df['acceso_servicios'] >= 4).mean() * 100 if 'acceso_servicios' in df.columns else 0

# Comparación ingresos * educación
if len(df_con_ingreso) > 0:
    ingreso_superior = df_con_ingreso[df_con_ingreso['nivel_educativo'].isin(['Superior universitaria', 'Posgrado'])]['ingreso_per_capita'].median()
    ingreso_primaria = df_con_ingreso[df_con_ingreso['nivel_educativo'] == 'Primaria (básica primaria)']['ingreso_per_capita'].median()
    if ingreso_primaria > 0:
        diferencia_pct = ((ingreso_superior / ingreso_primaria) * 100 - 100)
        texto_diferencia = f"{diferencia_pct:.0f}% más"
    else:
        texto_diferencia = "N/A"
else:
    texto_diferencia = "N/A"

print(f"""
RESUMEN EJECUTIVO - GEIH 2023

DATOS GENERALES:
   • Total de hogares analizados: {len(df):,}
   • Total de personas (estimado): {df['num_personas'].sum():,}
   • Variables analizadas: {len(df.columns)}

INGRESOS:
   • Ingreso per cápita promedio: ${df[df['ingreso_per_capita'] > 0]['ingreso_per_capita'].mean():,.0f}
   • Ingreso per cápita mediano: ${df[df['ingreso_per_capita'] > 0]['ingreso_per_capita'].median():,.0f}
   • Hogares sin ingreso reportado: {hogares_sin_ingreso:,} ({pct_sin_ingreso:.1f}%)

DEMOGRAFÍA:
   • Promedio personas por hogar: {df['num_personas'].mean():.2f}
   • Hogares unipersonales: {(df['num_personas'] == 1).sum():,} ({(df['num_personas'] == 1).mean()*100:.1f}%)
   • Hogares de 5+ personas: {(df['num_personas'] >= 5).sum():,} ({(df['num_personas'] >= 5).mean()*100:.1f}%)

SERVICIOS PÚBLICOS:
   • Índice promedio de acceso: {df['acceso_servicios'].mean():.2f}/5
   • Hogares con acceso total: {(df['acceso_servicios'] == 5).sum():,} ({(df['acceso_servicios'] == 5).mean()*100:.1f}%)

EMPLEO:
   • Hogares con desempleo: {df['hogar_con_desempleo'].sum():,} ({df['hogar_con_desempleo'].mean()*100:.1f}%)
   • Promedio de ocupados por hogar: {df['num_ocupados'].mean():.2f}

EDUCACIÓN (JEFE DE HOGAR):
   • Nivel más común: {df['nivel_educativo'].mode().iloc[0]}
   • Con educación superior: {nivel_superior.sum():,} ({pct_superior:.1f}%)

PRINCIPALES HALLAZGOS:
   1. El {servicios_4_mas:.0f}% de los hogares tiene acceso a 4 o más servicios públicos
   2. Los hogares con jefe de educación superior ganan {texto_diferencia} que los de primaria
   3. El {df['hogar_con_desempleo'].mean()*100:.1f}% de los hogares tiene al menos un miembro desempleado
   4. {df[df['material_pisos'] == 'Tierra'].shape[0]:,} hogares ({((df['material_pisos'] == 'Tierra').sum()/len(df))*100:.1f}%) tienen piso de tierra
   5. El {((df['acceso_servicios'] >= 4) & (df['ingreso_per_capita'] == 0)).sum():,} de hogares sin ingreso tiene buen acceso a servicios""")

print("\n" + "="*80)
print("EDA COMPLETADO CON ÉXITO")
print(f"Todos los gráficos y resultados se guardaron en: {os.path.join(ruta_base, 'EDA_resultados')}")
print("="*80)

# Guardar estadísticas
df.describe().to_csv(os.path.join(ruta_base, "EDA_resultados", "estadisticas_descriptivas.csv"))
print("\n Estadísticas descriptivas guardadas en: estadisticas_descriptivas.csv")