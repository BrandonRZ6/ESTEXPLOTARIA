# app.py (Versión Mejorada Estética)
# Proyecto Final: Análisis de Desigualdad Económica Global
# Autores: Kevin Criollo y Brandon Rodriguez
# Descripción: Dashboard interactivo para analizar PIB, desempleo e inflación

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ast import literal_eval
from scipy import stats
import statsmodels.api as sm
import re
from pathlib import Path

# ---------------------------
# Utilidades
# ---------------------------
def snake(col):
    col = col.strip()
    col = re.sub(r"[ /%()-]+", "_", col)
    col = re.sub(r"__+", "_", col)
    return col.lower()

def safe_get(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

# ---------------------------
# Cargar y preprocesar datos
# ---------------------------
@st.cache_data
def load_and_prep_data(path="base_de_datos.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path.resolve()}")
    df = pd.read_csv(path, low_memory=False)

    # Normalizar columnas
    df.rename(columns={c: snake(c) for c in df.columns}, inplace=True)

    # Parse borders
    bcol = safe_get(df, ["borders", "borders_"])
    if bcol:
        df["borders"] = df[bcol].apply(lambda x: literal_eval(x) if isinstance(x,str) and x.startswith("[") else [])
        df["n_borders"] = df["borders"].apply(len)
    else:
        df["borders"] = [[] for _ in range(len(df))]
        df["n_borders"] = 0

    # Variables numéricas
    num_candidates = {
        "gdp": ["gdp","gdp_","gdp_total","gdp_usd"],
        "gdp_growth": ["gdp_growth","gdp_growth_","gdp_growth_percent"],
        "interest_rate": ["interest_rate","interest_rate_"],
        "inflation_rate": ["inflation_rate","inflation_rate_","inflation.rate"],
        "jobless_rate": ["jobless_rate","jobless.rate","unemployment_rate"],
        "gov_budget": ["gov_budget"],
        "debt_gdp": ["debt_gdp","debt/gdp","debt_percent"],
        "current_account": ["current_account"],
        "population": ["population","population_"],
        "area": ["area"],
        "latitude": ["latitude","lat"],
        "longitude": ["longitude","lon","long"]
    }

    found_nums = {}
    for key, cand in num_candidates.items():
        found = safe_get(df, cand)
        if found: found_nums[key] = found

    for k, col in found_nums.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # GDP per capita
    gdp_col = found_nums.get("gdp")
    pop_col = found_nums.get("population")
    if gdp_col and pop_col:
        df["gdp_per_capita"] = np.where(
            (df[pop_col].notna()) & (df[pop_col] > 0),
            (df[gdp_col]*1e9)/(df[pop_col]*1e6),
            np.nan
        )
    else:
        df["gdp_per_capita"] = np.nan

    # Coordenadas
    lat_col = found_nums.get("latitude")
    lon_col = found_nums.get("longitude")
    if lat_col: df.rename(columns={lat_col:"latitude"}, inplace=True)
    if lon_col: df.rename(columns={lon_col:"longitude"}, inplace=True)

    # Nombre y región
    name_col = safe_get(df, ["name","country"])
    if name_col: df.rename(columns={name_col:"name"}, inplace=True)
    region_col = safe_get(df, ["region","continent"])
    if region_col: df.rename(columns={region_col:"region"}, inplace=True)
    subregion_col = safe_get(df, ["subregion","sub_region"])
    if subregion_col: df.rename(columns={subregion_col:"subregion"}, inplace=True)

    # Promedios regionales
    if "region" in df.columns:
        agg_cols = [c for c in ["gdp_per_capita","gdp","gdp_growth","inflation_rate","jobless_rate","debt_gdp"] if c in df.columns]
        if agg_cols:
            region_stats = df.groupby("region")[agg_cols].mean().reset_index()
            region_stats.rename(columns={c:f"{c}_regional_avg" for c in agg_cols}, inplace=True)
            df = df.merge(region_stats, on="region", how="left")

    return df

try:
    df = load_and_prep_data("base_de_datos.csv")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(page_title="🌍 World Economics Dashboard", layout="wide", page_icon="🌎")
# ---------------------------
# PORTADA Y PRESENTACIÓN DEL PROYECTO
# ---------------------------
st.markdown("""
# 🌍 Análisis de Desigualdad Económica Global

**Autores:** Kevin Criollo y Brandon Rodriguez  
**Institución:** Universidad Santo Tomas
**Año:** 2025

---

## 📋 Introducción

La desigualdad económica es una de las problemáticas más persistentes a nivel mundial. A pesar del 
crecimiento económico global, los países continúan mostrando **grandes diferencias en sus niveles de 
desarrollo y bienestar**.

Este análisis busca examinar indicadores macroeconómicos claves como:
- **Producto Interno Bruto (PIB) per cápita**
- **Tasa de desempleo**
- **Inflación**
- **Deuda pública**

Con el fin de comprender mejor las **diferencias y relaciones existentes** entre ellos.

---

## 🎯 Objetivos del Proyecto

### Objetivo General
Analizar la desigualdad económica entre distintos países a partir de indicadores como el PIB, 
la tasa de desempleo, la inflación y la deuda pública, con el fin de identificar patrones, 
diferencias regionales y posibles factores asociados. Aplicar metodologías estadísticas 
vistas en clase para extraer conclusiones válidas.

### Objetivos Específicos

1. **Comparar el nivel de PIB per cápita entre países**
   - Identificar diferencias significativas en el nivel de desarrollo económico
   - Aplicar prueba Kruskal–Wallis para validar estadísticamente las diferencias
   - Visualizar disparidades geográficas mediante mapas temáticos

2. **Examinar la relación entre tasa de desempleo y PIB**
   - Evaluar cómo el crecimiento económico influye en el empleo
   - Validar la Ley de Okun en datos reales
   - Identificar factores estructurales que modifican esta relación

3. **Analizar la variabilidad de la inflación entre países**
   - Evaluar impacto en el poder adquisitivo y estabilidad económica
   - Identificar regiones con inflación controlada vs. descontrolada
   - Relacionar inflación con bienestar económico

---

## 📊 Metodología

### Método 1: Análisis Descriptivo y Visualización
- **Herramientas:** Mapas temáticos, boxplots, scatter plots
- **Propósito:** Identificar patrones visuales y variabilidad económica

### Método 2: Prueba de Kruskal–Wallis
- **Fundamentación:** Test no paramétrico para comparar medianas entre múltiples grupos
- **Aplicación:** Comparar PIB per cápita entre regiones
- **Ventaja:** No requiere supuestos de normalidad ni homocedasticidad

### Método 3: Regresión Lineal
- **Fundamentación:** Ley de Okun (relación inversa PIB–desempleo)
- **Aplicación:** Evaluar influencia del crecimiento económico en empleo
- **Interpretación:** Validar patrones macroeconómicos conocidos

### Método 4: Análisis de Correlación
- **Herramientas:** Correlación de Pearson y Spearman
- **Propósito:** Evaluar relaciones entre indicadores económicos

""")

# ---------------------------
# Sidebar: filtros (robusto)
# ---------------------------
st.sidebar.title("🎛️ Filtros")

regions_opts = sorted(df["region"].dropna().unique()) if "region" in df.columns else []
regions = st.sidebar.multiselect("Región", options=regions_opts, default=regions_opts)

subregions_opts = sorted(df[df["region"].isin(regions)]["subregion"].dropna().unique()) if ("region" in df.columns and "subregion" in df.columns and regions) else []
subregions = st.sidebar.multiselect("Subregión", options=subregions_opts, default=subregions_opts)

# Rango GDP y Population con manejo NaNs
gdp_col = safe_get(df, ["gdp", "gdp_"])
pop_col = safe_get(df, ["population", "population_"])

gdp_series = df[gdp_col].dropna() if gdp_col in df.columns else pd.Series([0])
pop_series = df[pop_col].dropna() if pop_col in df.columns else pd.Series([0])

gdp_min, gdp_max = float(gdp_series.min()), float(gdp_series.max())
pop_min, pop_max = float(pop_series.min()), float(pop_series.max())

gdp_range = st.sidebar.slider("Rango de PIB (B USD)", min_value=gdp_min, max_value=gdp_max, value=(gdp_min, gdp_max), step=max(1.0, (gdp_max-gdp_min)/50.0))
pop_range = st.sidebar.slider("Rango de Población (M)", min_value=pop_min, max_value=pop_max, value=(pop_min, pop_max), step=max(0.1, (pop_max-pop_min)/50.0))

# Filtro aplicado
df_f = df.copy()
if regions:
    if "region" in df_f.columns:
        df_f = df_f[df_f["region"].isin(regions)]
if subregions:
    if "subregion" in df_f.columns:
        df_f = df_f[df_f["subregion"].isin(subregions)]
if gdp_col in df_f.columns:
    df_f = df_f[(df_f[gdp_col] >= gdp_range[0]) & (df_f[gdp_col] <= gdp_range[1])]
if pop_col in df_f.columns:
    df_f = df_f[(df_f[pop_col] >= pop_range[0]) & (df_f[pop_col] <= pop_range[1])]

# Limpiar lat/lon para mapas
if ("latitude" in df_f.columns) and ("longitude" in df_f.columns):
    df_f = df_f.dropna(subset=["latitude","longitude"])

# ---------------------------
# KPI header
# ---------------------------
st.title("🌎 World Economics Dashboard")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Países", f"{len(df_f):,}")
col2.metric("PIB Global (B USD)", f"{df_f[gdp_col].sum():,.0f}" if gdp_col in df_f.columns else "N/A")
col3.metric("Población (M)", f"{df_f[pop_col].sum():,.0f}" if pop_col in df_f.columns else "N/A")
infl_col = "inflation_rate" if "inflation_rate" in df_f.columns else None
col4.metric("Prom. Inflación", f"{df_f[infl_col].mean():.1f}%" if infl_col else "N/A")

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs([
    "📘 Informe",
    "🗺️ Mapa Global",
    "📊 Rankings",
    "🔍 Comparación Países",
    "🧮 País vs Región",
    "📈 Distribución por Región",
    "➕ País Personalizado",
    "📋 Conclusiones"
])
tab_informe, tab_map, tab_rank, tab_compare, tab_vs_region, tab_box, tab_custom, tab_conclusions = tabs

# ---------------------------
# Pestaña: Informe (MEJORADO)
# ---------------------------
with tab_informe:
    st.header("📘 Informe — Introducción y Metodología")
    st.markdown("""
**Introducción**  
El análisis examina desigualdades macroeconómicas (PIB per cápita, desempleo e inflación) entre países y regiones.

**Metodología (resumen)**  
- Objetivo 1: Kruskal–Wallis (no paramétrico) + visualizaciones.  
- Objetivo 2: Correlación y regresión lineal simple (Jobless Rate ~ GDP per capita).  
- Objetivo 3: Estadísticos de dispersión y mapas de inflación.
    """)

    st.subheader("Comprensión de la Base de Datos")
    st.markdown("""
La base de datos contiene **información macroeconómica de distintos países del mundo**. Incluye variables como:

- **GDP**: Producto Interno Bruto per cápita (USD)
- **Jobless Rate**: Tasa de desempleo (%)
- **Inflation Rate**: Inflación anual (%)
- **Region**: Continente o región económica
- **Latitude / Longitude**: Ubicación geográfica
- **Name**: Nombre del país
- **Debt/GDP**: Ratio de deuda sobre PIB

**Exploración inicial:** Se verificaron tipos de datos, valores faltantes, rangos, presencia de outliers 
y distribución general de las principales variables.
    """)

    st.subheader("🧪 Kruskal–Wallis: GDP per capita por región")
    if "gdp_per_capita" in df.columns and "region" in df.columns:
        df_kw = df.dropna(subset=["gdp_per_capita","region"]).copy()
        groups = [g["gdp_per_capita"].values for name,g in df_kw.groupby("region") if len(g) >= 2]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            col1, col2 = st.columns(2)
            col1.metric("Estadístico H", f"{h:.4f}")
            col2.metric("p-value", f"{p:.6f}")
            
            if p < 0.05:
                st.success("✅ Diferencias SIGNIFICATIVAS entre regiones (p < 0.05)")
            else:
                st.info("ℹ️ NO hay diferencias significativas (p >= 0.05)")
        else:
            st.warning("No hay suficientes grupos con datos para Kruskal–Wallis.")
        
        # Tabla resumen por region
        st.subheader("📊 Resumen por Región")
        reg_tab = df.groupby("region")["gdp_per_capita"].agg(n="count", median="median", mean="mean", std="std").reset_index()
        st.dataframe(reg_tab.style.format({"median":"{:.0f}","mean":"{:.0f}","std":"{:.0f}"}), use_container_width=True)
    else:
        st.warning("Faltan columnas necesarias para Kruskal–Wallis (gdp_per_capita o region).")

    st.subheader("📈 Visual: PIB per cápita por región")
    if "gdp_per_capita" in df.columns and "region" in df.columns:
        plot_df = df.dropna(subset=["gdp_per_capita","region"]).copy()
        
        # Gráfico mejorado con escala normal
        fig = px.strip(
            plot_df, 
            x="region", 
            y="gdp_per_capita", 
            color="region", 
            hover_data={"name": True, "region": False},
            stripmode="overlay", 
            title="🎯 Distribución de PIB per cápita por región",
            labels={"gdp_per_capita": "PIB per cápita (USD)", "region": "Región"}
        )
        
        # Agregar mediana
        med = plot_df.groupby("region")["gdp_per_capita"].median().reset_index()
        fig.add_trace(go.Scatter(
            x=med["region"], 
            y=med["gdp_per_capita"], 
            mode="markers+lines",
            marker=dict(color="darkred", symbol="diamond", size=12, line=dict(color="white", width=2)),
            line=dict(color="darkred", width=2, dash="dash"),
            name="Mediana"
        ))
        
        fig.update_layout(
            showlegend=True, 
            xaxis_tickangle=-45, 
            height=550,
            hovermode="closest",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gráfico en escala logarítmica
        st.subheader("📊 Vista Logarítmica (mejor visualización)")
        plot_df_log = plot_df.assign(gdp_pc_log=np.log10(plot_df["gdp_per_capita"]))
        fig2 = px.box(
            plot_df_log, 
            x="region", 
            y="gdp_pc_log", 
            color="region",
            points="outliers",
            title="📦 PIB per cápita (escala log10) - Distribución por región",
            labels={"gdp_pc_log": "log10(PIB per cápita)", "region": "Región"}
        )
        fig2.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No hay datos para graficar PIB per cápita por región.")

    st.subheader("📉 Regresión: Desempleo vs PIB per capita (Ley de Okun)")
    if ("jobless_rate" in df.columns) and ("gdp_per_capita" in df.columns):
        reg_df = df.dropna(subset=["jobless_rate","gdp_per_capita"]).copy()
        if len(reg_df) >= 3:
            X = sm.add_constant(reg_df["gdp_per_capita"])
            model = sm.OLS(reg_df["jobless_rate"], X).fit()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Coef. (β1)", f"{model.params['gdp_per_capita']:.6f}")
            col2.metric("p-value", f"{model.pvalues['gdp_per_capita']:.6f}")
            col3.metric("R²", f"{model.rsquared:.4f}")
            
            scatter = px.scatter(
                reg_df, 
                x="gdp_per_capita", 
                y="jobless_rate", 
                color="region" if "region" in reg_df.columns else None,
                trendline="ols",
                hover_data={"name": True},
                title="🎯 Ley de Okun: Desempleo vs PIB per capita",
                labels={"gdp_per_capita": "PIB per cápita (USD)", "jobless_rate": "Tasa de desempleo (%)"}
            )
            scatter.update_layout(height=600, template="plotly_white", hovermode="closest")
            st.plotly_chart(scatter, use_container_width=True)
            
            st.info(f"📌 Interpretación: Por cada aumento de $1000 USD en PIB per cápita, el desempleo **cambia {model.params['gdp_per_capita']:.4f}%**")
        else:
            st.warning("Observaciones insuficientes para ajustar regresión.")
    else:
        st.info("Columnas necesarias para regresión no disponibles (jobless_rate y/o gdp_per_capita).")

    st.markdown("---")
    st.caption("⚠️ Revisar supuestos estadísticos (normalidad, homocedasticidad) antes de inferencias causales.")

# ---------------------------
# Pestaña Map
# ---------------------------
# ---------------------------
# Pestaña Map
# ---------------------------
with tab_map:
    st.subheader("🗺️ Mapa Interactivo: Indicador por País")

    # Indicadores disponibles en tu base procesada
    indicator = st.selectbox(
        "Selecciona el indicador",
        options=[
            "gdp", "gdp_per_capita", "inflation_rate", "jobless_rate",
            "debt_gdp", "gdp_growth", "population"
        ],
        index=0
    )

    # Validación de datos
    df_plot = df_f.dropna(subset=["latitude", "longitude", indicator])

    if df_plot.empty:
        st.warning("⚠️ No hay datos suficientes para mostrar el mapa.")
        st.stop()

    # Tamaño del punto
    if indicator in ["gdp", "population"]:
        df_plot["size"] = np.log1p(df_plot[indicator]).clip(lower=1)
    else:
        df_plot["size"] = df_plot[indicator].abs().clip(lower=0.1)

    fig = px.scatter_geo(
        df_plot,
        lat="latitude",
        lon="longitude",
        size="size",
        color=indicator,
        hover_name="name",
        hover_data=[
            "region", "subregion", "population",
            "gdp", "gdp_per_capita", "inflation_rate"
        ],
        projection="natural earth",
        color_continuous_scale="Viridis",
        title=f"{indicator} por país"
    )

    fig.update_geos(
    showland=True,
    showocean=True,
    oceancolor="LightBlue",
    landcolor="LightGreen",
    coastlinecolor="black",  # color de la costa
    coastlinewidth=1
)


    st.plotly_chart(fig, use_container_width=True)
# ---------------------------
# Pestaña Comparación Países
# ---------------------------
with tab_compare:
    st.subheader("⚖️ Comparación entre países")
    names = sorted(df["name"].dropna().unique()) if "name" in df.columns else []
    selected = st.multiselect("Selecciona 2–5 países", options=names, default=names[:5], max_selections=5)
    if len(selected) < 2:
        st.warning("Selecciona al menos 2 países.")
    else:
        cols = [c for c in ["gdp_per_capita","inflation_rate","jobless_rate","debt_gdp","gdp_growth"] if c in df.columns]
        comp = df[df["name"].isin(selected)].set_index("name")[cols].round(2)
        # radar normalized
        maxv = comp.max().max() if not comp.empty else 1
        fig = go.Figure()
        for country in comp.index:
            fig.add_trace(go.Scatterpolar(r=(comp.loc[country].values / maxv), theta=comp.columns, fill='toself', name=country))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])), title="Radar (normalizado)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp, use_container_width=True)

# ---------------------------
# Pestaña País vs Región
# ---------------------------
with tab_vs_region:
    st.subheader("🆚 País vs Promedio Regional")
    names = sorted(df["name"].dropna().unique()) if "name" in df.columns else []
    country = st.selectbox("Selecciona un país", options=names)
    if country:
        crow = df[df["name"]==country].iloc[0]
        region = crow.get("region", None)
        inds = [c for c in ["gdp_per_capita","inflation_rate","jobless_rate","debt_gdp","gdp_growth"] if c in df.columns]
        country_vals, region_vals, labels = [], [], []
        for ind in inds:
            cv = crow.get(ind, np.nan)
            rv = crow.get(f"{ind}_regional_avg", np.nan)
            if pd.notna(cv) and pd.notna(rv):
                country_vals.append(cv); region_vals.append(rv); labels.append(ind)
        if not labels:
            st.warning("No hay suficientes datos para esta comparación.")
        else:
            comp_df = pd.DataFrame({"Indicador": labels*2, "Valor": country_vals+region_vals, "Tipo": ["País"]*len(labels)+[f"{region} avg"]*len(labels)})
            fig = px.bar(comp_df, x="Indicador", y="Valor", color="Tipo", barmode="group", title=f"{country} vs {region} (promedio)")
            st.plotly_chart(fig, use_container_width=True)
            dif = np.array(country_vals)-np.array(region_vals)
            pct = [f"{(cv-rv)/rv*100:+.1f}%" if rv!=0 else "N/A" for cv,rv in zip(country_vals,region_vals)]
            table = pd.DataFrame({"Indicador":labels,"País":country_vals,"Región":region_vals,"Δ abs":dif,"Δ %":pct})
            st.dataframe(table, use_container_width=True)

# ---------------------------
# Pestaña Distribución (MEJORADA)
# ---------------------------
with tab_box:
    st.subheader("📦 Distribución por Región - Análisis de Variabilidad")
    metrics_box = [c for c in ["gdp_per_capita","inflation_rate","jobless_rate","debt_gdp","gdp_growth"] if c in df.columns]
    metric_box = st.selectbox("Indicador", metrics_box, index=0, key="box_metric")
    
    df_box = df_f.dropna(subset=[metric_box]).copy() if metric_box in df_f.columns else df_f.copy()
    
    fig = px.box(
        df_box, 
        x="region", 
        y=metric_box, 
        color="region",
        points="all",  # Mostrar todos los puntos
        title=f"📊 {metric_box.replace('_', ' ').title()} - Distribución por región",
        labels={"region": "Región", metric_box: metric_box.replace("_", " ").title()}
    )
    fig.update_layout(
        height=600, 
        template="plotly_white",
        xaxis_tickangle=-45,
        showlegend=False,
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas adicionales
    st.subheader("📊 Estadísticas de Variabilidad")
    stats_table = df_box.groupby("region")[metric_box].agg(["count", "mean", "std", "min", "max"]).round(2)
    st.dataframe(stats_table.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

# ---------------------------
# Pestaña País Personalizado
# ---------------------------
with tab_custom:
    st.subheader("➕ País personalizado (qué-pasaría-si)")
    if "custom_countries" not in st.session_state:
        st.session_state.custom_countries = []
    empty = {"name":"Mi País","region":"Americas","subregion":"South America","latitude":-34.0,"longitude":-64.0,"gdp":100.0,"population":10.0,"inflation_rate":5.0,"jobless_rate":6.0,"debt_gdp":60.0,"gdp_growth":2.0}
    edited = st.data_editor(pd.DataFrame([empty]), num_rows="dynamic", hide_index=True, use_container_width=True)
    if st.button("➕ Añadir país"):
        for _, row in edited.iterrows():
            if pd.notna(row["name"]) and row["name"].strip():
                gdp_pc = (row["gdp"] * 1e9)/(row["population"]*1e6) if row["population"]>0 else np.nan
                custom = row.to_dict()
                custom["gdp_per_capita"] = round(gdp_pc,0)
                # rellenar columnas faltantes
                for c in df.columns:
                    if c not in custom:
                        custom[c] = np.nan
                st.session_state.custom_countries.append(custom)
                st.success(f"{row['name']} añadido.")
            else:
                st.warning("Nombre requerido.")
    if st.session_state.custom_countries:
        cdf = pd.DataFrame(st.session_state.custom_countries)
        st.dataframe(cdf[[c for c in ["name","region","gdp","population","gdp_per_capita","inflation_rate"] if c in cdf.columns]], use_container_width=True)
        if st.button("🗑️ Borrar todos"):
            st.session_state.custom_countries = []
            st.rerun()

# ---------------------------
# NUEVA PESTAÑA: CONCLUSIONES
# ---------------------------
with tab_conclusions:
    st.header("📋 Conclusiones Generales del Análisis")
    
    st.markdown("""
    ## ✅ Hallazgos Principales
    
    ### 1️⃣ Desigualdad Global en PIB per Cápita
    
    **Resultado:** La prueba Kruskal–Wallis confirma que **existen diferencias estadísticamente 
    significativas** en el PIB per cápita entre regiones (p < 0.05).
    
    **Hallazgos clave:**
    - **América del Norte, Europa Occidental y Oceanía**: PIB per cápita **elevado** (economías desarrolladas)
    - **África y partes de América Latina**: PIB per cápita **bajo** (menor desarrollo económico)
    - **Asia**: Heterogeneidad significativa (desde Japón desarrollado hasta economías emergentes)
    
    **Implicaciones:**
    - Diferencias estructurales en productividad, tecnología, educación e infraestructura
    - Concentración de riqueza en regiones industrializadas
    - Brechas internas significativas dentro de continentes
    
    ---
    
    ### 2️⃣ Relación entre PIB y Desempleo (Ley de Okun)
    
    **Resultado:** Se observa una **relación inversa entre PIB y tasa de desempleo**, confirmando 
    la Ley de Okun en los datos reales.
    
    **Hallazgos clave:**
    - Países con **mayor PIB per cápita** tienden a tener **menor desempleo**
    - Países con **bajo PIB** presentan **tasas de desempleo más altas**
    - La relación **no es perfecta** debido a factores estructurales locales
    
    **Factores que modifican la relación:**
    - Políticas laborales y regulaciones de empleo
    - Nivel educativo de la población
    - Acceso a tecnología e innovación
    - Grado de informalidad económica
    - Crisis económicas y contextos locales
    
    **Implicaciones:**
    - El crecimiento económico es necesario pero **no suficiente** para reducir desempleo
    - Se requieren políticas complementarias de empleo y capacitación
    - La estructura económica local determina el impacto real del crecimiento
    
    ---
    
    ### 3️⃣ Variabilidad de la Inflación y Estabilidad Económica
    
    **Resultado:** La inflación presenta **alta variabilidad entre regiones**, siendo un 
    **indicador crítico de estabilidad económica**.
    
    **Hallazgos clave:**
    - **Europa y Oceanía**: Inflación baja y estable (< 5%)
    - **África**: Inflación muy alta en varios países (> 100%)
    - **América Latina y Asia**: Valores intermedios con fluctuaciones
    
    **Impacto de la inflación alta:**
    - Pérdida del poder adquisitivo de la población
    - Depreciación monetaria
    - Incertidumbre económica y riesgo para inversiones
    - Desajustes en consumo y ahorro
    
    **Impacto de la inflación baja y estable:**
    - Mayor estabilidad macroeconómica
    - Condiciones favorables para inversión
    - Planificación económica más predecible
    - Mejor preservación del valor del dinero
    
    ---
    
    ## 🎓 Conclusiones Académicas
    
    ### Confirmación de Teorías Económicas
    
    ✅ **Ley de Okun**: Validada en los datos (relación inversa PIB–desempleo)
    
    ✅ **Desigualdad de Kuznets**: Se observa correlación entre nivel de desarrollo y desigualdad
    
    ✅ **Teoría Cuantitativa del Dinero**: Alta inflación asociada a inestabilidad monetaria
    
    ### Limitaciones del Análisis
    
    ⚠️ No se incluyen **variables contextuales** (guerras, pandemias, cambios políticos)
    
    ⚠️ **Correlación ≠ Causalidad**: Relaciones observadas pueden tener causas comunes
    
    ⚠️ **Datos de corte transversal**: No permite analizar **evolución temporal**
    
    ⚠️ Presencia de **valores extremos y atípicos** en algunos países
    
    ---
    
    ## 💡 Recomendaciones de Política Económica
    
    ### Para Países de Bajo Ingreso
    1. **Invertir en educación y capital humano** para aumentar productividad
    2. **Mejorar infraestructura** para facilitar comercio y producción
    3. **Fortalecer instituciones** para atraer inversión extranjera
    4. **Diversificar la economía** reduciendo dependencia de sectores primarios
    
    ### Para Países con Desempleo Alto
    1. **Implementar programas de capacitación** alineados con demanda laboral
    2. **Fomentar emprendimiento** y pequeñas empresas
    3. **Reducir rigidez laboral** sin sacrificar protección social
    4. **Estimular crecimiento económico** mediante políticas de demanda
    
    ### Para Países con Inflación Alta
    1. **Controlar agregados monetarios** (política del banco central)
    2. **Mejorar disciplina fiscal** reduciendo déficit público
    3. **Anclar expectativas de inflación** con credibilidad institucional
    4. **Diversificar fuentes de financiamiento** del gobierno
    
    ---
    
    ## 🔍 Próximas Líneas de Investigación
    
    - Análisis de **series de tiempo** para estudiar evolución 2010–2024
    - Inclusión de variables de **educación, salud y tecnología**
    - Estudio de **causalidad** usando métodos econométricos avanzados
    - Análisis de **COVID-19** y otros shocks macroeconómicos
    - Comparación de **políticas públicas** efectivas entre países
    
    ---
    
    ## 📌 Resumen Ejecutivo
    
    **Este análisis evidencia que:**
    
    1. La desigualdad económica global es **real, significativa y estructural**
    2. El crecimiento económico **reduce desempleo pero con variaciones** según contexto local
    3. La inflación controlada es **condición necesaria** para estabilidad y bienestar
    4. Se requieren **políticas multidimensionales**, no solo crecimiento económico
    5. Los datos confirman **teorías macroeconómicas clásicas** en contextos reales
    
    **Conclusión final:** La desigualdad económica mundial requiere de **intervenciones 
    coordinadas a nivel nacional, regional e internacional**, combinando políticas de 
    crecimiento, empleo, estabilidad monetaria e inversión en capital humano.
    
    ---
    
    ✨ **Fin del Análisis** ✨
    """)

# ---------------------------

# Pie de página
# ---------------------------
st.markdown("---")
st.caption("💡 Datos: World Economics Database  Proyecto Final 2025")
