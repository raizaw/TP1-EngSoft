from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import kruskal, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import statsmodels.api as sm
except Exception:
    sm = None

st.set_page_config(
    page_title="Dashboard Imobiliário SP",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET = "VALOR DO M2 DO TERRENO"
AREA_COL = "AREA DO TERRENO"
ZONE_COL = "ZONEAMENTO"

DISTANCE_COLUMNS = {
    "Metrô": "DIST_METRO_M",
    "Trem": "DIST_TREM_M",
    "Ônibus": "DIST_ONIBUS_M",
    "Parque / área verde": "DIST_PARQUE_M",
}

SOCIO_FEATURES = ["cd_indice_", "qt_populac"]
MODEL_FEATURES = [
    AREA_COL,
    "DIST_METRO_M",
    "DIST_TREM_M",
    "DIST_ONIBUS_M",
    "DIST_PARQUE_M",
    "cd_indice_",
    "qt_populac",
]

ZONE_GROUP_COL = "GRUPO_ZONEAMENTO"
ZONE_GROUP_DESC_COL = "DESCRICAO_GRUPO_ZONEAMENTO"
ZONE_GROUP_ORDER = [
    "ZEU — Eixo de estruturação urbana",
    "ZEM — Eixo de estruturação metropolitana",
    "ZC — Centralidade",
    "ZCOR — Corredor",
    "ZM — Zona mista",
    "ZER — Exclusivamente residencial",
    "ZPR — Predominantemente residencial",
    "ZEIS — Interesse social",
    "ZPI — Predominantemente industrial",
    "ZDE — Desenvolvimento econômico",
    "ZEPAM — Proteção ambiental",
    "ZEP — Preservação",
    "ZOE — Ocupação especial",
    "ZPDS — Preservação e desenvolvimento sustentável",
    "Outros / não classificado",
]

FEATURE_EXPLANATIONS = {
    AREA_COL: "Área do terreno do lote. Ajuda o modelo a capturar diferenças estruturais entre terrenos pequenos e grandes.",
    "DIST_METRO_M": "Distância, em metros, até a estação de metrô mais próxima. Quanto menor, maior é a proximidade ao metrô.",
    "DIST_TREM_M": "Distância, em metros, até a estação de trem mais próxima.",
    "DIST_ONIBUS_M": "Distância, em metros, até o ponto/corredor/equipamento de ônibus mais próximo disponível na base.",
    "DIST_PARQUE_M": "Distância, em metros, até parque ou área verde mais próxima disponível na base.",
    "cd_indice_": "Código/índice socioeconômico do IPVS associado à região do lote. Representa uma dimensão de vulnerabilidade social.",
    "qt_populac": "Quantidade de população associada à área/setor do IPVS cruzado com o lote.",
}

DEFAULT_DATA_PATH = os.getenv(
    "IPTU_PARQUET",
    "/content/drive/MyDrive/engsoft/processed/lotes_iptu_master.parquet",
)
DEFAULT_IPVS_PATH = os.getenv(
    "IPVS_ZIP",
    "/content/drive/MyDrive/engsoft/processed/geoportal_indice_paulista_vulnerabilidadesocial.zip",
)
DEFAULT_CULTURA_PATHS = {
    "Museus": os.getenv(
        "CULTURA_MUSEUS_ZIP",
        "/content/drive/MyDrive/engsoft/processed/geoportal_equipamento_cultura_museus_v3.zip",
    ),
    "Espaços culturais": os.getenv(
        "CULTURA_ESPACOS_ZIP",
        "/content/drive/MyDrive/engsoft/processed/geoportal_equipamento_cultura_espacos_culturais_v3.zip",
    ),
    "Teatros, cinemas e shows": os.getenv(
        "CULTURA_TEATROS_ZIP",
        "/content/drive/MyDrive/engsoft/processed/geoportal_equipamento_cultura_teatro_cinema_show_v3.zip",
    ),
}


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def file_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def br_money(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def br_number(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pvalue_text(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value == 0:
        return "< 1e-300"
    return f"{value:.2e}"


def conclusion_from_p(p_value: float, alpha: float = 0.05) -> str:
    if p_value is None or pd.isna(p_value):
        return "Indisponível"
    return "Validada" if p_value < alpha else "Não validada"


def clean_column_name(c: object) -> str:
    return str(c).strip()


def unique_existing_columns(cols: List[str], df: pd.DataFrame) -> List[str]:
    out = []
    seen = set()
    for col in cols:
        if col in df.columns and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def ensure_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Usa CRS métrico. GeoSampa costuma usar EPSG:31983."""
    if "geometry" not in gdf.columns:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(epsg=31983, allow_override=True)
    try:
        if gdf.crs.to_epsg() != 31983:
            return gdf.to_crs(epsg=31983)
    except Exception:
        return gdf.set_crs(epsg=31983, allow_override=True)
    return gdf


def add_lat_lon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" not in gdf.columns:
        return gdf
    gdf = ensure_metric_crs(gdf).copy()
    valid_geom = gdf.geometry.notna()
    gdf["lat"] = np.nan
    gdf["lon"] = np.nan
    if valid_geom.any():
        pts = gdf.loc[valid_geom, "geometry"].representative_point()
        pts_wgs = gpd.GeoSeries(pts, crs=gdf.crs).to_crs(epsg=4326)
        gdf.loc[valid_geom, "lon"] = pts_wgs.x.values
        gdf.loc[valid_geom, "lat"] = pts_wgs.y.values
    return gdf


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def metric_cards(items: List[Tuple[str, str, Optional[str]]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value, delta) in zip(cols, items):
        c.metric(label, value, delta=delta)


def summary_card(title: str, question: str, indicator: str, result: str, reading: str) -> None:
    # Usamos componentes nativos do Streamlit em vez de HTML fixo.
    # Assim o card fica legível tanto em tema claro quanto em tema escuro.
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(f"**Pergunta:** {question}")
        st.markdown(f"**Indicador:** {indicator}")
        st.markdown(f"**Resultado:** {result}")
        st.markdown(f"**Leitura:** {reading}")


def short_evidence_status(is_favorable: bool) -> str:
    return "Favorável" if is_favorable else "Fraca/mista"


def direction_text_for_distance(corr: float) -> str:
    if pd.isna(corr):
        return "sem resultado disponível"
    if corr < -0.05:
        return "relação negativa: imóveis mais próximos tendem a apresentar maior valor/m²"
    if corr > 0.05:
        return "relação positiva: imóveis mais distantes tendem a apresentar maior valor/m²"
    return "relação fraca ou próxima de zero"


def normalize_zone_text(zone: object) -> str:
    z = str(zone).upper().strip()
    z = (
        z.replace("Á", "A")
        .replace("À", "A")
        .replace("Â", "A")
        .replace("Ã", "A")
        .replace("É", "E")
        .replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ô", "O")
        .replace("Õ", "O")
        .replace("Ú", "U")
        .replace("Ç", "C")
    )
    return z


def zone_family(zone: object) -> Tuple[str, str]:
    """Agrupa códigos e descrições de zoneamento em famílias legíveis.

    A base pode trazer zoneamentos como 'ZCOR-2', 'Zona Corredor 2',
    'ZONA MISTA', etc. Por isso a classificação abaixo procura tanto códigos
    quanto palavras-chave. É um guia interpretativo, não uma fonte legal oficial.
    """
    z = normalize_zone_text(zone)
    z_compact = re.sub(r"[^A-Z0-9]", "", z)

    checks = [
        (
            lambda: "ZEIS" in z_compact or "INTERESSE SOCIAL" in z or "HABITACAO DE INTERESSE SOCIAL" in z,
            "ZEIS — Interesse social",
            "Áreas voltadas à moradia popular, regularização fundiária e produção de habitação de interesse social.",
        ),
        (
            lambda: "ZEPAM" in z_compact or "PROTECAO AMBIENTAL" in z or "PROTECAO AMBIENT" in z,
            "ZEPAM — Proteção ambiental",
            "Áreas com função ambiental, proteção de vegetação, recursos naturais ou restrições relevantes de ocupação.",
        ),
        (
            lambda: "ZPDS" in z_compact or "DESENVOLVIMENTO SUSTENTAVEL" in z,
            "ZPDS — Preservação e desenvolvimento sustentável",
            "Áreas em que a ocupação deve compatibilizar preservação ambiental e desenvolvimento urbano controlado.",
        ),
        (
            lambda: "ZEM" in z_compact or "ESTRUTURACAO DA TRANSFORMACAO METROPOLITANA" in z or "TRANSFORMACAO METROPOLITANA" in z,
            "ZEM — Eixo de estruturação metropolitana",
            "Áreas estratégicas de transformação urbana/metropolitana, ligadas a adensamento e reestruturação urbana.",
        ),
        (
            lambda: "ZEU" in z_compact or "EIXO DE ESTRUTURACAO" in z or "EIXO ESTRUTURACAO" in z,
            "ZEU — Eixo de estruturação urbana",
            "Áreas ao longo de eixos de transporte coletivo, normalmente associadas a maior adensamento e uso misto.",
        ),
        (
            lambda: "ZCOR" in z_compact or "CORREDOR" in z,
            "ZCOR — Corredor",
            "Áreas em corredores urbanos, geralmente com maior presença de comércio/serviços e transição entre zonas residenciais e centralidades.",
        ),
        (
            lambda: re.search(r"(^|[^A-Z])ZC([0-9]|[^A-Z]|$)", z) is not None or "CENTRALIDADE" in z,
            "ZC — Centralidade",
            "Áreas com concentração de comércio, serviços, empregos e maior diversidade de usos.",
        ),
        (
            lambda: re.search(r"(^|[^A-Z])ZM([0-9]|[^A-Z]|$)", z) is not None or "ZONA MISTA" in z or " MISTA" in z,
            "ZM — Zona mista",
            "Áreas que admitem mistura de usos residenciais, comerciais e de serviços, com intensidade variável.",
        ),
        (
            lambda: "ZER" in z_compact or "EXCLUSIVAMENTE RESIDENCIAL" in z,
            "ZER — Exclusivamente residencial",
            "Áreas predominantemente residenciais, normalmente com regras mais restritivas de uso e adensamento.",
        ),
        (
            lambda: "ZPR" in z_compact or "PREDOMINANTEMENTE RESIDENCIAL" in z,
            "ZPR — Predominantemente residencial",
            "Áreas com predominância residencial, mas com alguma possibilidade de usos complementares.",
        ),
        (
            lambda: "ZPI" in z_compact or "INDUSTRIAL" in z,
            "ZPI — Predominantemente industrial",
            "Áreas voltadas a usos industriais, logísticos e produtivos.",
        ),
        (
            lambda: "ZDE" in z_compact or "DESENVOLVIMENTO ECONOMICO" in z,
            "ZDE — Desenvolvimento econômico",
            "Áreas destinadas ao incentivo de atividades produtivas, econômicas, industriais ou logísticas.",
        ),
        (
            lambda: re.search(r"(^|[^A-Z])ZEP([0-9]|[^A-Z]|$)", z) is not None or "PRESERVACAO" in z,
            "ZEP — Preservação",
            "Áreas com objetivos de preservação ambiental, cultural, paisagística ou urbana.",
        ),
        (
            lambda: "ZOE" in z_compact or "OCUPACAO ESPECIAL" in z,
            "ZOE — Ocupação especial",
            "Áreas com características específicas, regras próprias ou usos urbanos especiais.",
        ),
    ]
    for predicate, label, desc in checks:
        if predicate():
            return label, desc
    return "Outros / não classificado", "Categoria não mapeada automaticamente no guia. Consulte a legislação urbanística para interpretação detalhada."


def zone_group_sort_key(group: str) -> int:
    try:
        return ZONE_GROUP_ORDER.index(group)
    except ValueError:
        return len(ZONE_GROUP_ORDER)


def add_zone_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    if ZONE_COL not in df.columns:
        return df
    df = df.copy()
    mapped = df[ZONE_COL].apply(zone_family)
    df[ZONE_GROUP_COL] = mapped.apply(lambda x: x[0])
    df[ZONE_GROUP_DESC_COL] = mapped.apply(lambda x: x[1])
    return df


def render_zone_guide(df: pd.DataFrame) -> None:
    if ZONE_COL not in df.columns:
        return
    tmp = add_zone_group_columns(df[[ZONE_COL]].dropna().copy())
    if tmp.empty:
        return

    rows = []
    for group, g in tmp.groupby(ZONE_GROUP_COL, sort=False):
        desc = g[ZONE_GROUP_DESC_COL].iloc[0]
        examples = sorted(g[ZONE_COL].astype(str).unique())[:8]
        suffix = "..." if g[ZONE_COL].nunique() > 8 else ""
        rows.append(
            {
                "Grupo": group,
                "Descrição geral": desc,
                "Códigos encontrados no filtro": ", ".join(examples) + suffix,
                "Registros": len(g),
                "_ordem": zone_group_sort_key(group),
            }
        )
    guide = pd.DataFrame(rows).sort_values(["_ordem", "Grupo"]).drop(columns="_ordem")

    with st.expander("Guia rápido dos grupos de zoneamento presentes no filtro", expanded=False):
        st.caption(
            "Este guia agrupa códigos parecidos para facilitar a leitura. É uma explicação geral e não substitui a consulta oficial à legislação de zoneamento."
        )
        st.dataframe(guide, use_container_width=True, hide_index=True)

def render_feature_dictionary() -> None:
    rows = [{"Variável": k, "O que significa": v} for k, v in FEATURE_EXPLANATIONS.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_missing_files(paths: Dict[str, str]) -> None:
    missing = {k: v for k, v in paths.items() if v and not file_exists(v)}
    if missing:
        with st.sidebar.expander("⚠️ Arquivos não encontrados", expanded=True):
            for k, v in missing.items():
                st.write(f"**{k}:** `{v}`")


# -----------------------------------------------------------------------------
# Carregamento com cache somente por caminho/parâmetros simples
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando GeoParquet principal...")
def load_main_geoparquet(path: str) -> gpd.GeoDataFrame:
    if not file_exists(path):
        raise FileNotFoundError(
            f"Arquivo principal não encontrado: {path}. Monte o Drive ou corrija o caminho na barra lateral."
        )
    gdf = gpd.read_parquet(path)
    gdf.columns = [clean_column_name(c) for c in gdf.columns]
    if "geometry" not in gdf.columns:
        raise ValueError("O arquivo principal não tem coluna geometry; confirme se é um GeoParquet válido.")
    for col in [TARGET, AREA_COL]:
        if col not in gdf.columns:
            raise ValueError(f"Coluna obrigatória ausente no arquivo principal: {col}")
    return ensure_metric_crs(gdf)


@st.cache_data(show_spinner="Carregando IPVS...")
def load_ipvs(path: str) -> Optional[gpd.GeoDataFrame]:
    if not file_exists(path):
        return None
    gdf = gpd.read_file(path)
    gdf.columns = [clean_column_name(c) for c in gdf.columns]
    gdf = ensure_metric_crs(gdf)
    keep = unique_existing_columns(["cd_indice_", "qt_populac", "geometry"], gdf)
    if "geometry" not in keep:
        return None
    return gdf[keep].copy()


@st.cache_data(show_spinner="Carregando equipamentos culturais...")
def load_cultura(paths_tuple: Tuple[Tuple[str, str], ...]) -> Optional[gpd.GeoDataFrame]:
    frames = []
    for nome, path in paths_tuple:
        if file_exists(path):
            tmp = gpd.read_file(path)
            tmp.columns = [clean_column_name(c) for c in tmp.columns]
            if "geometry" not in tmp.columns:
                continue
            tmp = ensure_metric_crs(tmp[["geometry"]].copy())
            tmp["tipo_cultura"] = nome
            frames.append(tmp)
    if not frames:
        return None
    cultura = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(cultura, geometry="geometry", crs=frames[0].crs)


@st.cache_data(show_spinner="Preparando amostra analítica...")
def prepare_data(
    data_path: str,
    ipvs_path: str,
    cultura_paths_tuple: Tuple[Tuple[str, str], ...],
    sample_n: int,
    random_state: int,
    enrich_ipvs: bool,
    enrich_cultura: bool,
) -> gpd.GeoDataFrame:
    base = load_main_geoparquet(data_path)

    wanted = [
        TARGET,
        AREA_COL,
        ZONE_COL,
        "geometry",
        *DISTANCE_COLUMNS.values(),
        *SOCIO_FEATURES,
        "DIST_CULTURA_M",
    ]
    cols = unique_existing_columns(wanted, base)
    df = base[cols].copy()
    df = safe_numeric(df, [TARGET, AREA_COL, *DISTANCE_COLUMNS.values(), *SOCIO_FEATURES, "DIST_CULTURA_M"])

    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    df = df[df[AREA_COL].notna() & (df[AREA_COL] > 0)]
    for col in DISTANCE_COLUMNS.values():
        if col in df.columns:
            df = df[df[col].notna() & (df[col] >= 0)]

    df = ensure_metric_crs(df)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=random_state)

    missing_socio = [col for col in SOCIO_FEATURES if col not in df.columns or df[col].isna().all()]
    if enrich_ipvs and missing_socio:
        ipvs = load_ipvs(ipvs_path)
        if ipvs is not None and "geometry" in ipvs.columns:
            df = gpd.sjoin(df, ipvs, how="left", predicate="within")
            if "index_right" in df.columns:
                df = df.drop(columns=["index_right"])
            for col in SOCIO_FEATURES:
                left = f"{col}_left"
                right = f"{col}_right"
                if right in df.columns:
                    if col in df.columns:
                        df[col] = df[col].combine_first(df[right])
                    elif left in df.columns:
                        df[col] = df[left].combine_first(df[right])
                    else:
                        df[col] = df[right]
                drop_cols = [c for c in [left, right] if c in df.columns]
                if drop_cols:
                    df = df.drop(columns=drop_cols)

    need_cultura = "DIST_CULTURA_M" not in df.columns or df["DIST_CULTURA_M"].isna().all()
    if enrich_cultura and need_cultura:
        cultura = load_cultura(cultura_paths_tuple)
        if cultura is not None and len(cultura) > 0:
            df = gpd.sjoin_nearest(
                df,
                cultura[["geometry", "tipo_cultura"]],
                how="left",
                distance_col="DIST_CULTURA_M",
            )
            if "index_right" in df.columns:
                df = df.drop(columns=["index_right"])
            df = df[~df.index.duplicated(keep="first")]

    df = safe_numeric(df, [TARGET, AREA_COL, *DISTANCE_COLUMNS.values(), *SOCIO_FEATURES, "DIST_CULTURA_M"])
    df = add_lat_lon(df)
    return df


# -----------------------------------------------------------------------------
# Estatísticas e modelo
# -----------------------------------------------------------------------------
def compute_stats(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}

    h1 = []
    for label, col in DISTANCE_COLUMNS.items():
        if col in df.columns and TARGET in df.columns:
            tmp = df[[col, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) >= 20 and tmp[col].nunique() > 1 and tmp[TARGET].nunique() > 1:
                pearson = tmp[[col, TARGET]].corr(method="pearson").iloc[0, 1]
                spearman, p = spearmanr(tmp[col], tmp[TARGET])
                h1.append({"Variável": label, "Coluna": col, "Pearson": pearson, "Spearman": spearman, "p-valor": p})
    out["h1"] = pd.DataFrame(h1)

    if ZONE_COL in df.columns and TARGET in df.columns:
        groups = [
            g[TARGET].dropna().values
            for _, g in df.groupby(ZONE_COL)
            if len(g[TARGET].dropna()) >= 30
        ]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
        else:
            stat, p = np.nan, np.nan
        out["h2_stat"] = stat
        out["h2_p"] = p
        out["h2_resumo"] = (
            df.groupby(ZONE_COL)[TARGET]
            .agg(contagem="count", media="mean", mediana="median", desvio="std")
            .sort_values("mediana", ascending=False)
            .reset_index()
        )

    if "DIST_CULTURA_M" in df.columns and TARGET in df.columns:
        tmp = df[["DIST_CULTURA_M", TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) >= 20 and tmp["DIST_CULTURA_M"].nunique() > 1 and tmp[TARGET].nunique() > 1:
            corr, p = spearmanr(tmp["DIST_CULTURA_M"], tmp[TARGET])
        else:
            corr, p = np.nan, np.nan
        out["h4_corr"] = corr
        out["h4_p"] = p
        bins = [0, 500, 1000, 2000, 5000, np.inf]
        labels = ["Até 500m", "500m–1km", "1km–2km", "2km–5km", "Mais de 5km"]
        tmp = tmp.copy()
        tmp["Faixa"] = pd.cut(tmp["DIST_CULTURA_M"], bins=bins, labels=labels, include_lowest=True)
        resumo = tmp.groupby("Faixa", observed=False)[TARGET].agg(contagem="count", mediana="median", media="mean").reset_index()
        if len(resumo) and pd.notna(resumo["mediana"].iloc[0]) and resumo["mediana"].iloc[0] != 0:
            ref = resumo["mediana"].iloc[0]
            resumo["Variação vs. até 500m"] = (resumo["mediana"] - ref) / ref * 100
        out["h4_resumo"] = resumo
    return out


def train_model(
    df: pd.DataFrame,
    max_rows: int,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> Tuple[Optional[RandomForestRegressor], Dict[str, object]]:
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    info: Dict[str, object] = {"missing": missing}
    if missing:
        return None, info

    model_df = df[MODEL_FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model_df = model_df[(model_df[TARGET] > 0) & (model_df[AREA_COL] > 0)]
    if len(model_df) < 500:
        info["error"] = "Amostra insuficiente para treinar. Use pelo menos 500 registros válidos."
        return None, info

    low, high = model_df[TARGET].quantile([0.01, 0.99])
    model_df = model_df[(model_df[TARGET] > low) & (model_df[TARGET] < high)]
    if len(model_df) > max_rows:
        model_df = model_df.sample(max_rows, random_state=random_state)

    X = model_df[MODEL_FEATURES]
    y = model_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    info.update({
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "predictions": pd.DataFrame({"Real": y_test.values, "Predito": pred}),
        "importance": pd.DataFrame({"Variável": MODEL_FEATURES, "Importância": rf.feature_importances_}).sort_values("Importância", ascending=False),
        "ranges": X.describe().T[["min", "50%", "max"]],
    })
    return rf, info


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuração")
st.sidebar.caption("Se o Drive estiver montado, os caminhos abaixo já devem funcionar.")

data_path = st.sidebar.text_input("GeoParquet principal", DEFAULT_DATA_PATH)
ipvs_path = st.sidebar.text_input("Base IPVS", DEFAULT_IPVS_PATH)

with st.sidebar.expander("Bases culturais", expanded=False):
    cultura_paths = {
        "Museus": st.text_input("Museus", DEFAULT_CULTURA_PATHS["Museus"]),
        "Espaços culturais": st.text_input("Espaços culturais", DEFAULT_CULTURA_PATHS["Espaços culturais"]),
        "Teatros, cinemas e shows": st.text_input("Teatros/cinemas/shows", DEFAULT_CULTURA_PATHS["Teatros, cinemas e shows"]),
    }

render_missing_files({"principal": data_path, "IPVS": ipvs_path, **cultura_paths})

st.sidebar.divider()
sample_n = st.sidebar.slider("Tamanho da amostra", 5_000, 300_000, 80_000, step=5_000)
random_state = st.sidebar.number_input("Random state", 0, 9999, 42, step=1)
enrich_ipvs = st.sidebar.checkbox("Completar IPVS por spatial join, se faltar", value=True)
enrich_cultura = st.sidebar.checkbox("Calcular distância cultural, se faltar", value=True)

try:
    df = prepare_data(
        data_path=data_path,
        ipvs_path=ipvs_path,
        cultura_paths_tuple=tuple(cultura_paths.items()),
        sample_n=int(sample_n),
        random_state=int(random_state),
        enrich_ipvs=enrich_ipvs,
        enrich_cultura=enrich_cultura,
    )
except Exception as exc:
    st.error("Não foi possível carregar o dashboard.")
    st.exception(exc)
    st.stop()

if df.empty:
    st.error("A amostra ficou vazia depois da limpeza. Revise filtros, caminhos e colunas da base.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Filtros")
if ZONE_COL in df.columns:
    zones = sorted([str(z) for z in df[ZONE_COL].dropna().unique()])
    selected_zones = st.sidebar.multiselect("Zoneamentos", zones, default=zones)
    if selected_zones:
        df_view = df[df[ZONE_COL].astype(str).isin(selected_zones)].copy()
    else:
        df_view = df.copy()
else:
    df_view = df.copy()

if TARGET in df_view.columns and not df_view.empty:
    p99 = float(df_view[TARGET].quantile(0.99))
    upper_limit = st.sidebar.number_input(
        "Teto visual de valor/m²",
        min_value=0.0,
        max_value=max(float(df_view[TARGET].max()), p99, 1.0),
        value=float(p99),
        step=100.0,
    )
    df_plot = df_view[df_view[TARGET] <= upper_limit].copy()
else:
    df_plot = df_view.copy()

# Colunas auxiliares para organizar visualmente os zoneamentos em famílias.
df_view = add_zone_group_columns(df_view)
df_plot = add_zone_group_columns(df_plot)

stats = compute_stats(df_view)

# -----------------------------------------------------------------------------
# Cabeçalho
# -----------------------------------------------------------------------------
st.title("🏙️ Dashboard Imobiliário — Valor venal por m² em São Paulo")
st.markdown(
    "Dashboard interativo das quatro hipóteses analisadas no notebook, com filtros, testes estatísticos, "
    "modelo preditivo e mapa amostral."
)

metric_cards([
    ("Registros filtrados", f"{len(df_view):,}".replace(",", "."), None),
    ("Mediana valor/m²", br_money(df_view[TARGET].median()), None),
    ("Média valor/m²", br_money(df_view[TARGET].mean()), None),
    ("Zoneamentos", str(df_view[ZONE_COL].nunique()) if ZONE_COL in df_view.columns else "—", None),
])

tab_resumo, tab_h1, tab_h2, tab_h3, tab_h4, tab_mapa, tab_dados = st.tabs([
    "Resumo executivo",
    "H1 Transporte/verde",
    "H2 Zoneamento",
    "H3 Predição",
    "H4 Cultura",
    "Mapa",
    "Dados",
])

# -----------------------------------------------------------------------------
# Resumo
# -----------------------------------------------------------------------------
with tab_resumo:
    st.header("Resumo executivo")
    h1 = stats.get("h1", pd.DataFrame())
    h2_p = stats.get("h2_p", np.nan)
    h4_p = stats.get("h4_p", np.nan)
    h4_corr = stats.get("h4_corr", np.nan)

    h1_best_label = "—"
    h1_best_corr = np.nan
    h1_best_p = np.nan
    if isinstance(h1, pd.DataFrame) and not h1.empty:
        h1_tmp = h1.dropna(subset=["Spearman"]).copy()
        if not h1_tmp.empty:
            h1_tmp["abs_corr"] = h1_tmp["Spearman"].abs()
            best = h1_tmp.sort_values("abs_corr", ascending=False).iloc[0]
            h1_best_label = str(best["Variável"])
            h1_best_corr = float(best["Spearman"])
            h1_best_p = float(best["p-valor"])

    model_missing = [c for c in MODEL_FEATURES if c not in df_view.columns]
    model_df_n = 0
    if not model_missing:
        model_df_n = len(df_view[MODEL_FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna())
    h3_status = "Pronta" if not model_missing and model_df_n >= 500 else "Dados insuficientes"
    h3_indicator = f"{model_df_n:,} registros válidos para o modelo".replace(",", ".") if not model_missing else "Faltam variáveis do modelo"

    c1, c2 = st.columns(2)
    with c1:
        summary_card(
            "H1 — Transporte e áreas verdes",
            "A proximidade a metrô, trem, ônibus e parques está associada ao valor/m²?",
            f"Spearman mais forte: {h1_best_label} | ρ={h1_best_corr:.3f} | p={pvalue_text(h1_best_p)}" if pd.notna(h1_best_corr) else "Sem indicador disponível",
            short_evidence_status(pd.notna(h1_best_corr) and h1_best_corr < 0 and h1_best_p < 0.05),
            direction_text_for_distance(h1_best_corr),
        )
    with c2:
        summary_card(
            "H2 — Zoneamento",
            "Os diferentes zoneamentos apresentam distribuições distintas de valor/m²?",
            f"Teste Kruskal-Wallis | p={pvalue_text(h2_p)}",
            conclusion_from_p(h2_p),
            "p<0,05 indica que pelo menos dois grupos de zoneamento diferem estatisticamente em valor/m².",
        )
    c3, c4 = st.columns(2)
    with c3:
        summary_card(
            "H3 — Modelo preditivo",
            "As variáveis urbanas, estruturais e socioeconômicas conseguem prever o valor/m²?",
            h3_indicator,
            h3_status,
            "A validação principal fica na aba H3, com R², MAE, importância das variáveis e gráfico real vs. predito.",
        )
    with c4:
        summary_card(
            "H4 — Amenidades culturais",
            "A proximidade a museus, teatros, cinemas e espaços culturais está associada ao valor/m²?",
            f"Spearman ρ={h4_corr:.3f} | p={pvalue_text(h4_p)}" if pd.notna(h4_corr) else "Sem indicador disponível",
            short_evidence_status(pd.notna(h4_corr) and h4_corr < 0 and h4_p < 0.05),
            direction_text_for_distance(h4_corr),
        )

    st.markdown(
        "**Como ler os sinais:** em variáveis de distância, correlação negativa indica que imóveis mais próximos tendem a ter maior valor/m². "
        "p-valor abaixo de 0,05 indica evidência estatística, mas o tamanho do coeficiente continua sendo importante."
    )

    if isinstance(h1, pd.DataFrame) and not h1.empty:
        fig = px.bar(
            h1.sort_values("Spearman"),
            x="Spearman",
            y="Variável",
            orientation="h",
            text=h1.sort_values("Spearman")["Spearman"].round(3),
            title="Correlação de Spearman com o valor/m²",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribuição do valor/m²")
        fig = px.histogram(df_plot, x=TARGET, nbins=80, title="Histograma filtrado pelo teto visual")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if ZONE_COL in df_plot.columns:
            top_zone = df_plot.groupby(ZONE_COL)[TARGET].median().sort_values(ascending=False).head(12).reset_index()
            st.subheader("Top zoneamentos por mediana")
            fig = px.bar(top_zone, x=TARGET, y=ZONE_COL, orientation="h", title="Maiores medianas por zoneamento")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# H1
# -----------------------------------------------------------------------------
with tab_h1:
    st.header("Hipótese 1 — Transporte público e áreas verdes")
    st.markdown("Quanto menor a distância até metrô, trem, ônibus ou parques, maior deveria ser o valor por m².")

    h1 = stats.get("h1", pd.DataFrame())
    if isinstance(h1, pd.DataFrame) and not h1.empty:
        fig = px.bar(
            h1.sort_values("Spearman"),
            x="Spearman",
            y="Variável",
            orientation="h",
            text=h1.sort_values("Spearman")["Spearman"].round(3),
            title="Resumo da H1 — Correlação de Spearman com valor/m²",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        table = h1.copy()
        table["Pearson"] = table["Pearson"].round(4)
        table["Spearman"] = table["Spearman"].round(4)
        table["p-valor"] = table["p-valor"].map(pvalue_text)
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.warning("Não há colunas suficientes de distância para calcular a H1.")

    with st.expander("O que significa usar escala logarítmica da distância?"):
        st.markdown(
            "A escala logarítmica comprime distâncias muito grandes para o gráfico ficar legível. "
            "Ela não muda o dado original nem a estatística da tabela; muda apenas o eixo X do scatter plot. "
            "É útil porque a diferença entre 100m e 500m costuma ser mais importante visualmente do que a diferença entre 10km e 10,4km."
        )

    col1, col2 = st.columns([0.30, 0.70])
    with col1:
        selected_label = st.selectbox("Distância analisada", list(DISTANCE_COLUMNS.keys()))
        dist_col = DISTANCE_COLUMNS[selected_label]
        use_log = st.checkbox(
            "Usar escala logarítmica da distância no gráfico",
            value=True,
            help="Transforma o eixo X com log1p(distância), isto é, log(1 + distância). Serve apenas para melhorar a visualização.",
        )
        scatter_n = st.slider("Pontos no scatter H1", 1_000, 40_000, 8_000, step=1_000)

    with col2:
        if dist_col in df_plot.columns:
            plot_df = df_plot[[dist_col, TARGET]].dropna().copy()
            if len(plot_df) > scatter_n:
                plot_df = plot_df.sample(scatter_n, random_state=42)
            x = dist_col
            x_label = dist_col
            if use_log:
                plot_df[f"log1p_{dist_col}"] = np.log1p(plot_df[dist_col])
                x = f"log1p_{dist_col}"
                x_label = f"log(1 + {dist_col})"
            fig = px.scatter(
                plot_df,
                x=x,
                y=TARGET,
                opacity=0.35,
                trendline="ols",
                title=f"Valor/m² vs. distância até {selected_label.lower()}",
                labels={x: x_label},
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Coluna ausente: {dist_col}")

    if dist_col in df_plot.columns:
        bins = [0, 250, 500, 1000, 2000, 5000, np.inf]
        labels = ["0–250m", "250–500m", "500m–1km", "1–2km", "2–5km", ">5km"]
        faixa = df_plot[[dist_col, TARGET]].dropna().copy()
        faixa["Faixa"] = pd.cut(faixa[dist_col], bins=bins, labels=labels, include_lowest=True)
        resumo = faixa.groupby("Faixa", observed=False)[TARGET].agg(contagem="count", mediana="median", media="mean").reset_index()
        fig = px.bar(resumo, x="Faixa", y="mediana", text=resumo["mediana"].round(0), title=f"Mediana por faixa — {selected_label}")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# H2
# -----------------------------------------------------------------------------
with tab_h2:
    st.header("Hipótese 2 — Zoneamento urbano")
    st.markdown("Testa se o tipo de zoneamento altera significativamente o valor do m².")

    h2_p = stats.get("h2_p", np.nan)
    h2_stat = stats.get("h2_stat", np.nan)
    metric_cards([
        ("Resultado", conclusion_from_p(h2_p), None),
        ("Kruskal-Wallis H", br_number(h2_stat, 2), None),
        ("p-valor", pvalue_text(h2_p), None),
    ])

    render_zone_guide(df_view)

    with st.expander("Como interpretar o coeficiente por zoneamento?"):
        st.markdown(
            "O coeficiente vem de uma regressão linear exploratória. O modelo escolhe um zoneamento como categoria de referência. "
            "Cada coeficiente mostra quanto o valor/m² médio daquele zoneamento fica acima ou abaixo da referência, mantendo apenas essa comparação categórica. "
            "Coeficiente positivo indica associação com maior valor/m²; negativo indica associação com menor valor/m². "
            "Isso não prova causalidade, porque localização, renda, infraestrutura e centralidade também influenciam os preços."
        )

    resumo = stats.get("h2_resumo", pd.DataFrame())
    if isinstance(resumo, pd.DataFrame) and not resumo.empty:
        resumo = add_zone_group_columns(resumo)
        resumo["_ordem_grupo"] = resumo[ZONE_GROUP_COL].apply(zone_group_sort_key)
        resumo = resumo.sort_values(["_ordem_grupo", "mediana"], ascending=[True, False])

        max_z = min(40, len(resumo))
        top_n = st.slider("Zoneamentos no gráfico", 5, max_z, min(20, max_z)) if max_z >= 5 else max_z
        resumo_plot = resumo.head(top_n).copy()
        order = resumo_plot[ZONE_COL].astype(str).tolist()
        box_df = df_plot[df_plot[ZONE_COL].astype(str).isin(order)].copy()

        st.caption(
            "O boxplot abaixo está ordenado por grupo de zoneamento. Dentro de cada grupo, os códigos aparecem da maior para a menor mediana de valor/m²."
        )
        if ZONE_GROUP_COL in box_df.columns:
            color_order = sorted(box_df[ZONE_GROUP_COL].dropna().unique(), key=zone_group_sort_key)
        else:
            color_order = None
        fig = px.box(
            box_df,
            x=TARGET,
            y=ZONE_COL,
            color=ZONE_GROUP_COL if ZONE_GROUP_COL in box_df.columns else None,
            category_orders={ZONE_COL: list(reversed(order)), ZONE_GROUP_COL: color_order} if color_order else {ZONE_COL: list(reversed(order))},
            points=False,
            orientation="h",
            title="Distribuição do valor/m² por zoneamento — grupos próximos ficam juntos",
        )
        fig.update_layout(height=max(560, 28 * len(order)), legend_title_text="Grupo")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Resumo estatístico por zoneamento")
        show = resumo.drop(columns=["_ordem_grupo"], errors="ignore").copy()
        for c in ["media", "mediana", "desvio"]:
            show[c] = show[c].map(br_money)
        show["contagem"] = show["contagem"].map(lambda x: f"{int(x):,}".replace(",", "."))
        preferred_cols = unique_existing_columns([ZONE_GROUP_COL, ZONE_COL, "contagem", "media", "mediana", "desvio"], show)
        st.dataframe(show[preferred_cols], use_container_width=True, hide_index=True)

        st.subheader("Impacto financeiro estimado por regressão")
        if sm is None:
            st.warning("statsmodels não está instalado. Rode a célula de instalação novamente.")
        else:
            min_count = st.slider("Mínimo de registros por zona", 30, 1000, 100, step=10)
            valid = resumo.loc[resumo["contagem"] >= min_count, ZONE_COL]
            reg_df = df_view[df_view[ZONE_COL].isin(valid)][[ZONE_COL, TARGET]].dropna().copy()
            if len(reg_df) > 200_000:
                reg_df = reg_df.sample(200_000, random_state=42)
            if reg_df[ZONE_COL].nunique() >= 2:
                X = pd.get_dummies(reg_df[ZONE_COL], drop_first=True, dtype=int)
                X = sm.add_constant(X)
                y = reg_df[TARGET]
                model = sm.OLS(y, X).fit()
                coef = model.params.drop("const", errors="ignore").sort_values()
                coef_view = pd.concat([coef.head(8), coef.tail(8)]).reset_index()
                coef_view.columns = ["Zoneamento", "Coeficiente estimado"]
                coef_view = add_zone_group_columns(coef_view.rename(columns={"Zoneamento": ZONE_COL}))
                coef_view = coef_view.rename(columns={ZONE_COL: "Zoneamento"})
                fig = px.bar(
                    coef_view,
                    x="Coeficiente estimado",
                    y="Zoneamento",
                    color=ZONE_GROUP_COL if ZONE_GROUP_COL in coef_view.columns else None,
                    orientation="h",
                    title=f"Coeficientes por zoneamento | R²={model.rsquared:.3f}",
                )
                fig.update_layout(height=560, legend_title_text="Grupo")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Poucos zoneamentos elegíveis para regressão com o filtro atual.")
    else:
        st.warning("Coluna de zoneamento ausente ou insuficiente.")

# -----------------------------------------------------------------------------
# H3
# -----------------------------------------------------------------------------
with tab_h3:
    st.header("Hipótese 3 — Modelo preditivo")
    st.markdown("Usa características estruturais, distâncias urbanas e variáveis socioeconômicas para prever valor/m².")

    with st.expander("O que significam árvores e profundidade máxima?"):
        st.markdown(
            "A Random Forest é formada por várias árvores de decisão. **Árvores** é a quantidade de árvores usadas; mais árvores tendem a deixar o resultado mais estável, mas tornam o treino mais lento. "
            "**Profundidade máxima** limita o tamanho de cada árvore; profundidade maior permite capturar padrões mais complexos, mas aumenta o risco de o modelo decorar ruídos da amostra. "
            "Para apresentação, os valores padrão do dashboard já são uma escolha equilibrada."
        )

    with st.expander("Dicionário das variáveis usadas no modelo e no simulador"):
        render_feature_dictionary()

    missing = [c for c in MODEL_FEATURES if c not in df_view.columns]
    if missing:
        st.warning("Faltam colunas para treinar o modelo: " + ", ".join(missing))
        st.caption("Ative o spatial join do IPVS na barra lateral e confira o caminho do arquivo IPVS.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            upper_rows = min(150_000, len(df_view))
            if upper_rows <= 5_000:
                model_rows = upper_rows
                st.metric("Linhas máximas", f"{model_rows:,}".replace(",", "."))
            else:
                model_rows = st.slider("Linhas máximas para treino", 5_000, upper_rows, min(80_000, upper_rows), step=5_000)
        with c2:
            n_estimators = st.slider("Árvores da floresta", 50, 300, 100, step=50, help="Quantidade de árvores de decisão. Mais árvores: resultado mais estável e treino mais lento.")
        with c3:
            max_depth = st.slider("Profundidade máxima", 5, 30, 15, help="Limite de complexidade de cada árvore. Maior profundidade: modelo mais flexível, mas com mais risco de overfitting.")

        with st.spinner("Treinando Random Forest..."):
            rf, info = train_model(df_view, int(model_rows), int(n_estimators), int(max_depth), int(random_state))

        if rf is None:
            st.error(info.get("error", "Não foi possível treinar o modelo."))
        else:
            metric_cards([
                ("R²", f"{info['r2']:.4f}", "quanto maior, melhor"),
                ("MAE", br_money(info["mae"]), "erro médio por m²"),
                ("Treino/Teste", f"{info['n_train']:,}/{info['n_test']:,}".replace(",", "."), None),
            ])

            fig = px.bar(info["importance"], x="Importância", y="Variável", orientation="h", text=info["importance"]["Importância"].round(3), title="Importância das variáveis")
            fig.update_layout(height=440, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            pred = info["predictions"]
            if len(pred) > 6_000:
                pred = pred.sample(6_000, random_state=42)
            fig = px.scatter(pred, x="Real", y="Predito", opacity=0.35, title="Real vs. predito")
            min_axis = float(min(pred["Real"].min(), pred["Predito"].min()))
            max_axis = float(max(pred["Real"].max(), pred["Predito"].max()))
            fig.add_trace(go.Scatter(x=[min_axis, max_axis], y=[min_axis, max_axis], mode="lines", name="Predição perfeita"))
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Simulador de valor por m²")
            st.caption("Ajuste os valores abaixo para criar um cenário hipotético. A previsão é baseada no modelo treinado com a amostra filtrada.")
            ranges = info["ranges"]
            values = {}
            cols = st.columns(2)
            for i, feature in enumerate(MODEL_FEATURES):
                min_v = float(ranges.loc[feature, "min"])
                med_v = float(ranges.loc[feature, "50%"])
                max_v = float(ranges.loc[feature, "max"])
                if min_v == max_v:
                    max_v = min_v + 1.0
                with cols[i % 2]:
                    values[feature] = st.slider(
                        feature,
                        min_value=min_v,
                        max_value=max_v,
                        value=min(max(med_v, min_v), max_v),
                        help=FEATURE_EXPLANATIONS.get(feature, "Variável usada pelo modelo."),
                    )
            sim = pd.DataFrame([values])[MODEL_FEATURES]
            prediction = float(rf.predict(sim)[0])
            st.success(f"Predição estimada: {br_money(prediction)} por m²")

# -----------------------------------------------------------------------------
# H4
# -----------------------------------------------------------------------------
with tab_h4:
    st.header("Hipótese 4 — Amenidades culturais")
    st.markdown("Testa se a proximidade a museus, teatros, cinemas e espaços culturais eleva o valor por m².")

    if "DIST_CULTURA_M" not in df_view.columns:
        st.warning("A coluna DIST_CULTURA_M não está disponível. Ative o cálculo cultural e confira os caminhos das bases.")
    else:
        h4_p = stats.get("h4_p", np.nan)
        h4_corr = stats.get("h4_corr", np.nan)
        h4_status = short_evidence_status(pd.notna(h4_corr) and h4_corr < 0 and h4_p < 0.05)
        metric_cards([
            ("Resultado", h4_status, None),
            ("Spearman ρ", f"{h4_corr:.4f}" if pd.notna(h4_corr) else "—", None),
            ("p-valor", pvalue_text(h4_p), None),
        ])
        st.caption("Resultado favorável significa: correlação negativa + p-valor abaixo de 0,05. Como é uma distância, sinal negativo indica maior valor perto dos equipamentos culturais.")

        resumo = stats.get("h4_resumo", pd.DataFrame())
        if isinstance(resumo, pd.DataFrame) and not resumo.empty:
            fig = px.bar(resumo, x="Faixa", y="mediana", text=resumo["mediana"].round(0), title="Mediana do valor/m² por faixa de distância cultural")
            st.plotly_chart(fig, use_container_width=True)
            show = resumo.copy()
            show["mediana"] = show["mediana"].map(br_money)
            show["media"] = show["media"].map(br_money)
            if "Variação vs. até 500m" in show.columns:
                show["Variação vs. até 500m"] = show["Variação vs. até 500m"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
            st.dataframe(show, use_container_width=True, hide_index=True)

        scatter_n = st.slider("Pontos no scatter H4", 1_000, 40_000, 8_000, step=1_000)
        plot_df = df_plot[["DIST_CULTURA_M", TARGET]].dropna().copy()
        if len(plot_df) > scatter_n:
            plot_df = plot_df.sample(scatter_n, random_state=42)
        plot_df["log1p_DIST_CULTURA_M"] = np.log1p(plot_df["DIST_CULTURA_M"])
        fig = px.scatter(plot_df, x="log1p_DIST_CULTURA_M", y=TARGET, opacity=0.35, trendline="ols", title="Valor/m² vs. distância cultural")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Mapa
# -----------------------------------------------------------------------------
with tab_mapa:
    st.header("Mapa amostral")
    st.markdown(
        "O mapa usa pontos pequenos para manter a navegação leve e permitir enxergar melhor bairros, vias e concentração espacial dos valores."
    )

    if "lat" not in df_plot.columns or "lon" not in df_plot.columns:
        st.warning("Latitude/longitude não disponíveis.")
    else:
        max_limit = min(50_000, max(500, len(df_plot.dropna(subset=["lat", "lon", TARGET]))))
        default_map_n = min(20_000, max_limit)

        col_map_1, col_map_2 = st.columns([0.45, 0.55])

        with col_map_1:
            map_n = st.slider("Pontos no mapa", 500, max_limit, default_map_n, step=500)

        with col_map_2:
            color_mode = st.radio(
                "Modo de cor",
                [
                    "Percentil do valor/m²",
                    "Valor real do m²",
                ],
                horizontal=True,
                help=(
                    "Percentil distribui melhor as cores e facilita a leitura espacial. "
                    "Valor real usa o valor em R$/m², mas pode ser mais sensível a valores extremos."
                ),
            )

        map_df = df_plot.dropna(subset=["lat", "lon", TARGET]).copy()

        if len(map_df) > map_n:
            map_df = map_df.sample(map_n, random_state=42)

        if map_df.empty:
            st.info("Não há pontos válidos para o mapa.")
        else:
            custom_colorscale = [
                [0.00, "#243B8A"],  # azul escuro
                [0.25, "#2F80ED"],  # azul
                [0.50, "#7B2CBF"],  # roxo
                [0.75, "#F15BB5"],  # rosa
                [1.00, "#D00000"],  # vermelho
            ]

            if color_mode == "Percentil do valor/m²":
                map_df["COR_MAPA"] = map_df[TARGET].rank(pct=True) * 100
                color_values = map_df["COR_MAPA"]
                cmin = 0
                cmax = 100
                colorbar = {
                    "title": "Percentil<br>valor/m²",
                    "ticksuffix": "%",
                }
                map_title = "Lotes representados por pontos pequenos — cor por percentil do valor/m²"
                caption = (
                    "A cor representa o percentil do valor/m² dentro dos pontos exibidos no mapa. "
                    "Esse modo distribui melhor as cores e facilita a comparação espacial. "
                    "O valor real em R$/m² continua aparecendo ao passar o mouse sobre cada ponto."
                )
            else:
                map_df["COR_MAPA"] = map_df[TARGET]

                # Escala robusta: continua usando valor real, mas limita visualmente a escala
                # entre os percentis 1 e 99 para evitar que poucos outliers deixem o mapa inteiro
                # com uma única cor. O valor real não é alterado e segue aparecendo no hover.
                cmin = float(map_df[TARGET].quantile(0.01))
                cmax = float(map_df[TARGET].quantile(0.99))
                if cmin == cmax:
                    cmin = float(map_df[TARGET].min())
                    cmax = float(map_df[TARGET].max())

                color_values = map_df["COR_MAPA"]
                colorbar = {
                    "title": "Valor/m²<br>R$",
                }
                map_title = "Lotes representados por pontos pequenos — cor por valor real do m²"
                caption = (
                    "A cor representa o valor real do m². Para manter o contraste visual, a escala de cores "
                    "é limitada entre os percentis 1 e 99; valores muito extremos continuam no mapa e aparecem "
                    "normalmente ao passar o mouse."
                )

            hover_text = []
            for _, r in map_df.iterrows():
                parts = [
                    f"Valor/m²: {br_money(r.get(TARGET, np.nan))}",
                    f"Percentil no mapa: {br_number((map_df[TARGET].rank(pct=True) * 100).loc[r.name], 1)}",
                ]
                if AREA_COL in map_df.columns:
                    parts.append(f"Área: {br_number(r.get(AREA_COL, np.nan), 1)} m²")
                if ZONE_COL in map_df.columns:
                    parts.append(f"Zoneamento: {r.get(ZONE_COL, '—')}")
                if ZONE_GROUP_COL in map_df.columns:
                    parts.append(f"Grupo: {r.get(ZONE_GROUP_COL, '—')}")
                if "DIST_METRO_M" in map_df.columns:
                    parts.append(f"Dist. metrô: {br_number(r.get('DIST_METRO_M', np.nan), 0)} m")
                if "DIST_CULTURA_M" in map_df.columns:
                    parts.append(f"Dist. cultura: {br_number(r.get('DIST_CULTURA_M', np.nan), 0)} m")
                hover_text.append("<br>".join(parts))

            fig = go.Figure(
                go.Scattermapbox(
                    lat=map_df["lat"],
                    lon=map_df["lon"],
                    mode="markers",
                    marker={
                        "size": 4,
                        "opacity": 0.78,
                        "color": color_values,
                        "colorscale": custom_colorscale,
                        "cmin": cmin,
                        "cmax": cmax,
                        "showscale": True,
                        "colorbar": colorbar,
                    },
                    text=hover_text,
                    hoverinfo="text",
                )
            )

            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=10.5,
                mapbox_center={
                    "lat": float(map_df["lat"].median()),
                    "lon": float(map_df["lon"].median()),
                },
                height=680,
                margin={"r": 0, "t": 35, "l": 0, "b": 0},
                title=map_title,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(caption)

# -----------------------------------------------------------------------------
# Dados
# -----------------------------------------------------------------------------
with tab_dados:
    st.header("Dados usados no dashboard")
    st.write(f"Linhas filtradas: **{len(df_view):,}**".replace(",", "."))
    st.write("Colunas disponíveis:", list(df_view.columns))

    preview_cols = unique_existing_columns([
        TARGET,
        AREA_COL,
        ZONE_COL,
        "DIST_METRO_M",
        "DIST_TREM_M",
        "DIST_ONIBUS_M",
        "DIST_PARQUE_M",
        "cd_indice_",
        "qt_populac",
        "DIST_CULTURA_M",
        "lat",
        "lon",
    ], df_view)

    st.dataframe(df_view[preview_cols].head(1000), use_container_width=True)
    csv = df_view[preview_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar amostra filtrada em CSV",
        data=csv,
        file_name="amostra_dashboard_iptu_sp.csv",
        mime="text/csv",
    )
