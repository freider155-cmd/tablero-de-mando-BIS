"""
app.py - Dashboard Streamlit para cálculo MVEI (Métrica de Valor Estratégico de la Innovación)

Requisitos:
 - streamlit
 - pandas
 - numpy
 - plotly

Este archivo contiene una aplicación completa de Streamlit que:
 - Permite configurar pesos y metas (min/max) por cada dimensión
 - Permite introducir valores actuales (raw) por dimensión
 - Normaliza valores con control de errores (división por cero)
 - Calcula el puntaje MVEI (0..1) con la restricción de pesos
 - Muestra KPI (Gauge), gráfico radar (Plotly) y tabla de resumen

Autor: Generado programáticamente
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def clamp01(x: float) -> float:
    """Clamp a rango [0, 1]"""
    return float(max(0.0, min(1.0, x)))


def normalize_value(value: float, vmin: float, vmax: float) -> float:
    """
    Normaliza `value` entre [vmin, vmax] usando la fórmula:

        Normalizado = (ValorActual - ValorMin) / (ValorMax - ValorMin)

    Maneja división por cero si vmin == vmax:
      - Si vmin == vmax == value => consideramos 1.0 (cumplió la meta exacta)
      - Si vmin == vmax != value => consideramos 0.0 (fuera del rango sin referencia)

    Devuelve un valor en [0.0, 1.0].
    """
    try:
        denom = vmax - vmin
        if denom == 0:
            # Evitar ZeroDivisionError: interpretar si el valor coincide con la meta exacta
            return 1.0 if float(value) == float(vmin) else 0.0

        norm = (float(value) - float(vmin)) / float(denom)
        return clamp01(norm)
    except Exception:
        # En caso de valores no convertibles a float
        return 0.0


def ensure_weights_sum(weights: dict) -> (dict, bool):
    """
    Asegura que la suma de los pesos sea 1.0.

    Si la suma es 0, repartimos uniformemente. Si la suma no es 1, normalizamos (dividiendo por suma).
    Devuelve (weights_normalizados, was_normalized_boolean)
    """
    s = sum(weights.values())
    if s == 0:
        # repartir uniformemente
        n = len(weights)
        return ({k: 1.0 / n for k in weights}, True)

    if abs(s - 1.0) > 1e-8:
        return ({k: v / s for k, v in weights.items()}, True)

    return (weights, False)


def compute_mvei(scores: dict, weights: dict) -> float:
    """
    Calcula MVEI = sum(S_i * W_i). Asume `scores` y `weights` tienen las mismas keys.
    Devuelve valor en 0..1
    """
    mvei = 0.0
    for k in scores:
        mvei += float(scores[k]) * float(weights[k])
    return clamp01(mvei)


def build_radar_plot(scores: dict) -> go.Figure:
    """
    Construye un gráfico radar (polar) con 4 ejes usando plotly.graph_objects.
    """
    labels = list(scores.keys())
    values = [scores[k] for k in labels]

    # Radar requiere cerrar la serie (primer valor repetido al final)
    closed_values = values + [values[0]]
    closed_labels = labels + [labels[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=closed_values,
                theta=closed_labels,
                fill="toself",
                name="Perfil normalizado",
                hovertemplate="%{theta}: %{r:.2f}",
            )
        ]
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], tickformat=".2f"),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
    )

    return fig


def build_gauge(mvei_value: float) -> go.Figure:
    """
    Construye un gráfico tipo gauge (plotly) para visualizar MVEI entre 0..1.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=float(mvei_value),
            number={"valueformat": ".2f"},
            delta={"reference": 0.5, "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".2f"},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 0.25], "color": "#ffe6e6"},
                    {"range": [0.25, 0.5], "color": "#fff0cc"},
                    {"range": [0.5, 0.75], "color": "#e6f7ff"},
                    {"range": [0.75, 1], "color": "#e6ffe6"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": float(mvei_value)},
            },
            title={"text": "MVEI - Puntaje de Valor Estratégico"},
        )
    )

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
    return fig


def main():
    st.set_page_config(page_title="Tablero MVEI • Innovación", layout="wide")

    st.title("Tablero de Mando — MVEI (Métrica de Valor Estratégico de la Innovación)")

    # Marca de licencia en la parte superior derecha
    st.markdown(
        """
        <div style="position: fixed; top: 60px; right: 20px; font-size: 12px; color: #333; text-align: right; z-index: 999; background-color: #f0f0f0; padding: 8px 12px; border-radius: 4px; border-left: 3px solid #1f77b4;">
            <div style="font-weight: bold; color: #1f77b4;">Licencia MIT</div>
            <div style="margin-top: 2px;">© 2025 Freider Barbosa</div>
            <div><a href="mailto:freider155@gmail.com" style="color: #0066cc; text-decoration: none; font-size: 11px;">freider155@gmail.com</a></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: configuraciones
    st.sidebar.header("Configuración — Pesos & Metas")

    st.sidebar.markdown("Ajuste los pesos (debe sumar 1.0). Si no suman, el sistema normaliza automáticamente.")

    # Default sample values
    default_weights = {
        "Estrategia": 0.4,
        "Organización": 0.25,
        "Cliente": 0.25,
        "Liderazgo": 0.10,
    }

    with st.sidebar.form(key="weights_form"):
        w_e = st.number_input("Peso - Estratégica", min_value=0.0, max_value=1.0, value=default_weights["Estrategia"], step=0.01, format="%.4f")
        w_o = st.number_input("Peso - Organizacional", min_value=0.0, max_value=1.0, value=default_weights["Organización"], step=0.01, format="%.4f")
        w_c = st.number_input("Peso - Cliente", min_value=0.0, max_value=1.0, value=default_weights["Cliente"], step=0.01, format="%.4f")
        w_l = st.number_input("Peso - Liderazgo", min_value=0.0, max_value=1.0, value=default_weights["Liderazgo"], step=0.01, format="%.4f")
        weights_submit = st.form_submit_button("Aplicar pesos")

    weights_raw = {"Estrategia": w_e, "Organización": w_o, "Cliente": w_c, "Liderazgo": w_l}
    weights, normalized_flag = ensure_weights_sum(weights_raw)

    if normalized_flag:
        st.sidebar.warning("Los pesos no sumaban 1.0 — se han normalizado automáticamente.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Define metas (Valores mínimo y máximo) para normalización de cada métrica.")

    # Allow metrics ranges in the sidebar
    with st.sidebar.expander("Metas / Rango por métrica (Min / Max)", expanded=True):
        # Provide default minima/maxima for the example metrics
        default_ranges = {
            "Tasa de Adopción (%)": (0.0, 100.0),
            "Reducción de Tiempo (h)": (0.0, 40.0),
            "NPS (puntos)": (-100.0, 100.0),
            "Número de Patentes": (0.0, 50.0),
        }

        ta_min = st.number_input("Tasa de Adopción - Min", value=default_ranges["Tasa de Adopción (%)"][0], format="%.3f")
        ta_max = st.number_input("Tasa de Adopción - Max", value=default_ranges["Tasa de Adopción (%)"][1], format="%.3f")

        rt_min = st.number_input("Reducción de Tiempo - Min", value=default_ranges["Reducción de Tiempo (h) ".strip()][0] if "Reducción de Tiempo (h)" in default_ranges else default_ranges["Reducción de Tiempo (h) ".strip()][0], format="%.3f")
        # The above reference to a key with trailing spaces might fail in some python versions if not normalized -- ensure correct defaults
        rt_min = ta_min if False else rt_min

        rt_max = st.number_input("Reducción de Tiempo - Max", value=default_ranges["Reducción de Tiempo (h)"][1], format="%.3f")

        nps_min = st.number_input("NPS - Min", value=default_ranges["NPS (puntos)"][0], format="%.3f")
        nps_max = st.number_input("NPS - Max", value=default_ranges["NPS (puntos)"][1], format="%.3f")

        pat_min = st.number_input("Número de Patentes - Min", value=default_ranges["Número de Patentes"][0], format="%.3f")
        pat_max = st.number_input("Número de Patentes - Max", value=default_ranges["Número de Patentes"][1], format="%.3f")

    # Main: Input datos actuales
    st.header("Entradas — Valores actuales de métricas")
    st.markdown("Introduce los valores actuales (ValorActual) para cada métrica que alimentará la normalización.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ta_value = st.number_input("Tasa de Adopción (%)", value=20.0, step=0.1, format="%.3f")

    with col2:
        rt_value = st.number_input("Reducción de Tiempo (h)", value=4.0, step=0.1, format="%.3f")

    with col3:
        nps_value = st.number_input("NPS (puntos)", value=10.0, step=0.5, format="%.3f")

    with col4:
        pat_value = st.number_input("Número de Patentes", value=2.0, step=1.0, format="%.3f")

    # Build dictionaries for normalization and labels mapping to the 4 MVEI dimensions
    # Mapeo ejemplo: cada dimensión puede estar representada por una métrica clave.
    raw_values = {
        "Estrategia": ta_value,
        "Organización": rt_value,
        "Cliente": nps_value,
        "Liderazgo": pat_value,
    }

    mins = {
        "Estrategia": ta_min,
        "Organización": rt_min,
        "Cliente": nps_min,
        "Liderazgo": pat_min,
    }

    maxs = {
        "Estrategia": ta_max,
        "Organización": rt_max,
        "Cliente": nps_max,
        "Liderazgo": pat_max,
    }

    # Normalizar cada métrica
    normalized_scores = {}
    for k in raw_values:
        v = raw_values[k]
        vmin = mins[k]
        vmax = maxs[k]
        normalized_scores[k] = normalize_value(v, vmin, vmax)

    # Use the normalized/possibly-normalized weights
    mvei_value = compute_mvei(normalized_scores, weights)

    st.markdown("---")

    # KPI + Radar + Summary table layout
    left, right = st.columns([1, 2])

    with left:
        st.subheader("KPI — MVEI")
        # Gauge plot
        gfig = build_gauge(mvei_value)
        st.plotly_chart(gfig, use_container_width=True)

        # Also show a compact numeric metric with percentage
        st.metric(label="MVEI (0..1)", value=f"{mvei_value:.3f}", delta=f"{(mvei_value-0.5):+.3f}")

    with right:
        st.subheader("Perfil de Impacto — Gráfico Radar (Normalizado)")
        rfig = build_radar_plot(normalized_scores)
        st.plotly_chart(rfig, use_container_width=True)

    st.markdown("---")

    st.subheader("Tabla de resumen — Valores, Normalizados y Contribución ponderada")
    # Construir DataFrame de salida
    data_rows = []
    for k in raw_values:
        raw = raw_values[k]
        vmin = mins[k]
        vmax = maxs[k]
        s = normalized_scores[k]
        w = weights[k]
        contrib = float(s) * float(w)
        data_rows.append(
            {
                "Dimensión": k,
                "Valor_Raw": raw,
                "Min": vmin,
                "Max": vmax,
                "Normalizado (S_i)": round(s, 4),
                "Peso (W_i)": round(w, 4),
                "Contribución (S_i * W_i)": round(contrib, 4),
            }
        )

    df = pd.DataFrame(data_rows).set_index("Dimensión")
    st.dataframe(df.style.format({"Valor_Raw": "{:.4f}", "Min": "{:.4f}", "Max": "{:.4f}"}))

    # Mostrar resumen final
    st.markdown("---")
    st.write("### Resultado final")
    st.write(f"**MVEI (score)** = {mvei_value:.4f}  —  (0 = sin impacto, 1 = impacto máximo según metas)")

    # Mostrar ayuda y notas
    with st.expander("Notas Técnicas / Cómo se calculó MVEI"):
        st.markdown(
            """
        - **Normalización**: Normalizado = (ValorActual - ValorMin) / (ValorMax - ValorMin)
            - Si el normalizado < 0 → se asume 0; si > 1 → 1.
            - Si Min == Max: si ValorActual == Min => Normalizado = 1.0, si no => 0.0 para evitar división por cero.
        - **Pesos**: Si la suma de los pesos no es 1.0, la aplicación normaliza automáticamente para mantener la restricción. Si todos los pesos = 0, la aplicación distribuye uniformemente.
        - **MVEI**: MVEI = sum(S_i * W_i) para i ∈ {Estrategia, Organización, Cliente, Liderazgo}
        """
        )


if __name__ == "__main__":
    main()
