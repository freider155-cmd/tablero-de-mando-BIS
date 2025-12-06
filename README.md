# Tablero MVEI — Métrica de Valor Estratégico de la Innovación

Aplicación Streamlit para calcular la MVEI (score 0..1) basada en cuatro dimensiones: Estratégica, Organizacional, Cliente y Liderazgo.

Características:
- Interfaz lateral para ajustar pesos (W) y metas (min/max)
- Panel principal para introducir valores actuales de métricas clave
- Cálculo y normalización segura de las métricas (manejo de división por cero)
- Visualizaciones con Plotly: Gauge y Radar (Araña)
- Tabla de resumen con valores brutos, normalizados y contribución ponderada

Archivos:
- `app.py` — App principal (Streamlit + Plotly)
- `requirements.txt` — Dependencias recomendadas

Instalación y ejecución (recomendado dentro de un entorno virtual):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Notas:
- La aplicación normaliza automáticamente los pesos si la suma no es 1.0.
- Si el rango (min == max) se define de forma idéntica, la normalización evita división por cero: si Value == min se considera 1.0, si no => 0.0.
