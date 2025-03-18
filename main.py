# region Configurações
import streamlit as st
import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import funcoes
import duckdb
import multiprocessing

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
st.set_page_config(page_title="Estudo de Ampacidade", page_icon="📊", layout="wide")

st.sidebar.image(
    "C:\\Users\\caio.alves\\OneDrive - ARAXA\\Imagens\\Logo\\d63fa966-d82c-4bac-ae1e-d1bb60e8e2a6.webp",
    width=200,
)
st.sidebar.markdown("")

num_cores = multiprocessing.cpu_count()

estacoesselecionadas = st.session_state.get("estacoesselecionadas", [])

PontosEstacoes = pd.read_csv(
    "C:\\Users\\caio.alves\\OneDrive - ARAXA\\Python\\AXT\\Projetos\\INMET\\.streamlit\\BaseDeDados\\INMETPontos.csv",
    encoding="ISO-8859-1",
    sep=";",
    names=["Estacao", "Longitude", "Latitude"],
)
PontosEstacoes["Classificação do Ponto"] = "Fora do Buffer"
# endregion

# region Cache
if "df_pontos" not in st.session_state:
    st.session_state.df_pontos = None
if "df_total" not in st.session_state:
    st.session_state.df_total = None
if "df_diretriz" not in st.session_state:
    st.session_state.df_diretriz = None
if "df_pontos_vertices" not in st.session_state:
    st.session_state.df_pontos_vertices = pd.DataFrame(
        columns=["Nome do Vertice", "Latitude", "Longitude"], data=[["P1", 0.0, 0.0]]
    )
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "buffer": 0,
        "idnom": 0.0,
        "innom": 0.0,
        "vdnom": 0.0,
        "vnnom": 0.0,
        "idsobrec": 0.0,
        "insobrec": 0.0,
        "vdsobrec": 0.0,
        "vnsobrec": 0.0,
        "tensao": 0,
        "diamentrototal": 0.0,
        "diamentroaluminio": 0.0,
        "coefabsorsolar": 0.0,
        "condutor": "",
        "feixe": 0,
        "rdc20cc": 0.0,
        "epsilon": 0.0,
        "alturamediadalt": 0,
        "coefvarrestemp": 0.0,
    }
# endregion

st.sidebar.subheader("")
menu = st.sidebar.radio(
    "Escolha uma página", ["Dados de Entrada", "Ver Mapa e Gráficos"]
)

if menu == "Dados de Entrada":
    st.title("Dados de Entrada")
    st.text("Página dedicada às informações necessárias para o estudo de ampacidade")
    st.text("")

    buffer = st.number_input(
        "Buffer",
        format="%d",
        value=st.session_state.inputs.get("buffer", 0),
        key="buffer",
    )

    st.session_state.inputs["buffer"] = buffer

    st.text("")

    tab1, tab2, tab3 = st.tabs(
        ["Informações Gerais", "Correntes de Referência", "Vértices"]
    )

    with tab1:  # Informações Gerais
        col_a, col_b = st.columns([1, 1])

        tensao = col_a.number_input(
            "Tensão (kV)",
            format="%d",  # Alterado para inteiro
            value=st.session_state.inputs.get("tensao", 0),
            key="tensao_input",
        )
        st.session_state.inputs["tensao"] = tensao

        diamentrototal = col_a.number_input(
            "Diâmetro Total (m)",
            format="%.5f",
            value=st.session_state.inputs.get("diamentrototal", 0.0),
            key="diamentrototal_input",
        )
        st.session_state.inputs["diamentrototal"] = diamentrototal

        diamentroaluminio = col_a.number_input(
            "Diâmetro Alumínio (m)",
            format="%.5f",
            value=st.session_state.inputs.get("diamentroaluminio", 0.0),
            key="diamentroaluminio_input",
        )
        st.session_state.inputs["diamentroaluminio"] = diamentroaluminio

        coefabsorsolar = col_a.number_input(
            "Coef. Absor. Solar",
            format="%.1f",
            value=st.session_state.inputs.get("coefabsorsolar", 0.0),
            key="coefabsorsolar_input",
        )
        st.session_state.inputs["coefabsorsolar"] = coefabsorsolar

        alturamediadalt = col_a.number_input(
            "Altura Média da LT",
            format="%d",  # Alterado para inteiro
            value=st.session_state.inputs.get("alturamediadalt", 0),
            key="alturamediadalt_input",
        )
        st.session_state.inputs["alturamediadalt"] = alturamediadalt

        condutor = col_b.text_input(
            "Condutor",
            value=st.session_state.inputs.get("condutor", ""),
            key="condutor_input",
        )
        st.session_state.inputs["condutor"] = condutor

        feixe = col_b.number_input(
            "Feixe",
            format="%d",  # Alterado para inteiro
            value=st.session_state.inputs.get("feixe", 0),
            key="feixe_input",
        )
        st.session_state.inputs["feixe"] = feixe

        rdc20cc = col_b.number_input(
            "RDC20º CC(ohms/km)",
            format="%.3f",
            value=st.session_state.inputs.get("rdc20cc", 0.0),
            key="rdc20cc_input",
        )
        st.session_state.inputs["rdc20cc"] = rdc20cc

        epsilon = col_b.number_input(
            "Epsilon",
            format="%.2f",
            value=st.session_state.inputs.get("epsilon", 0.0),
            key="epsilon_input",
        )
        st.session_state.inputs["epsilon"] = epsilon

        coefvarrestemp = col_b.number_input(
            "Coef.Var.Res.Temp °C-¹",
            format="%.5f",
            value=st.session_state.inputs.get("coefvarrestemp", 0.0),
            key="coefvarrestemp_input",
        )
        st.session_state.inputs["coefvarrestemp"] = coefvarrestemp

    with tab2:  # Correntes de Referência
        col_a, col_b = st.columns([1, 1])

        idnom = col_a.number_input(
            "ID-Nom",
            format="%.1f",
            value=st.session_state.inputs.get("idnom", 0.0),
            key="idnom_input",
        )
        st.session_state.inputs["idnom"] = idnom

        innom = col_a.number_input(
            "IN-Nom",
            format="%.1f",
            value=st.session_state.inputs.get("innom", 0.0),
            key="innom_input",
        )
        st.session_state.inputs["innom"] = innom

        vdnom = col_a.number_input(
            "VD-Nom",
            format="%.1f",
            value=st.session_state.inputs.get("vdnom", 0.0),
            key="vdnom_input",
        )
        st.session_state.inputs["vdnom"] = vdnom

        vnnom = col_a.number_input(
            "VN-Nom",
            format="%.1f",
            value=st.session_state.inputs.get("vnnom", 0.0),
            key="vnnom_input",
        )
        st.session_state.inputs["vnnom"] = vnnom

        idsobrec = col_b.number_input(
            "ID-Sobrec",
            format="%.1f",
            value=st.session_state.inputs.get("idsobrec", 0.0),
            key="idsobrec_input",
        )
        st.session_state.inputs["idsobrec"] = idsobrec

        insobrec = col_b.number_input(
            "IN-Sobrec",
            format="%.1f",
            value=st.session_state.inputs.get("insobrec", 0.0),
            key="insobrec_input",
        )
        st.session_state.inputs["insobrec"] = insobrec

        vdsobrec = col_b.number_input(
            "VD-Sobrec",
            format="%.1f",
            value=st.session_state.inputs.get("vdsobrec", 0.0),
            key="vdsobrec_input",
        )
        st.session_state.inputs["vdsobrec"] = vdsobrec

        vnsobrec = col_b.number_input(
            "VN-Sobrec",
            format="%.1f",
            value=st.session_state.inputs.get("vnsobrec", 0.0),
            key="vnsobrec_input",
        )
        st.session_state.inputs["vnsobrec"] = vnsobrec

    with tab3:  # Vertices e importação da Database e criação de TCnom

        # region VerticesINMET
        COLUNAS_FIXAS = ["Nome do Vertice", "Latitude", "Longitude"]

        st.subheader("Pontos")
        st.text("A ordenação deles é fundamental para a Diretriz da LT no mapa.")

        df = st.session_state.df_pontos_vertices
        df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: f"{x:.6f}".replace(".", ","))
        PontosVertices = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key="pontos_vertices_editor",  # Chave única para o editor
        )
        PontosVertices["Latitude"] = (
            PontosVertices["Latitude"].replace(",", ".", regex=True).astype(float)
        )
        PontosVertices["Longitude"] = (
            PontosVertices["Longitude"].replace(",", ".", regex=True).astype(float)
        )
    if not PontosVertices.equals(st.session_state.df_pontos_vertices):
        st.session_state.df_pontos_vertices = PontosVertices
    aplicarpontosinseridos = st.button("Aplicar")
    if aplicarpontosinseridos:

        if PontosVertices.isnull().values.any() or not all(
            PontosVertices.iloc[:, 1:]
            .applymap(lambda x: isinstance(x, float))
            .all(axis=1)
        ):
            st.error(
                "Por favor, preencha todas as informações corretamente. As colunas 'Latitude' e 'Longitude' devem ser números."
            )
        else:
            st.success("Todas as informações foram preenchidas corretamente.")
            with st.spinner("Aguarde, estou consultando a Base do INMET..."):
                PontosDiretriz = funcoes.diretriz(PontosVertices, 0.1)

                for index, row in PontosDiretriz.iterrows():
                    nom = row["Estacao"]
                    lat = row["Latitude"]
                    long = row["Longitude"]
                    for index, row in PontosEstacoes.iterrows():
                        nom_est = row["Estacao"]
                        nom_lat = row["Latitude"]
                        nom_long = row["Longitude"]
                        if (
                            funcoes.haversine(
                                lat1=lat, lon1=long, lat2=nom_lat, lon2=nom_long
                            )
                            <= buffer
                        ):
                            estacoesselecionadas.append(nom_est)
                estacoesselecionadas = list(set(estacoesselecionadas))
                caminho_csv = "BaseDeDados\INMET.csv"

                if estacoesselecionadas:

                    query = f"""
                    SELECT * FROM read_csv_auto('{caminho_csv}', ALL_VARCHAR=TRUE)
                    WHERE Estacao IN ({', '.join(f"'{v}'" for v in estacoesselecionadas)})
                    """
                    # endregion

                    # region TCnom
                    df = duckdb.query(query).df()
                    st.text(f"Linhas processadas: {len(df)}")

                else:
                    st.error("Nenhuma estação selecionada")

                df["Tcnom"] = np.vectorize(funcoes.tcnom)(
                    st.session_state.inputs["alturamediadalt"],
                    st.session_state.inputs["diamentrototal"],
                    st.session_state.inputs["diamentroaluminio"],
                    st.session_state.inputs["epsilon"],
                    st.session_state.inputs["coefabsorsolar"],
                    st.session_state.inputs["coefvarrestemp"],
                    st.session_state.inputs["feixe"],
                    st.session_state.inputs["rdc20cc"],
                    df["TemperaturaAr"].apply(funcoes.parse_to_float),
                    df["RadiacaoGlobal"].apply(funcoes.parse_to_float),
                    df["VelocidadedoVento"].apply(funcoes.parse_to_float),
                    df["Data"].apply(
                        lambda x: int(str(x).replace("-", "/").split("/")[1])
                    ),
                    df["Hora"].apply(lambda x: int(str(x)[:2])),
                    st.session_state.inputs["idnom"],
                    st.session_state.inputs["innom"],
                    st.session_state.inputs["vdnom"],
                    st.session_state.inputs["vnnom"],
                    st.session_state.inputs["idsobrec"],
                    st.session_state.inputs["insobrec"],
                    st.session_state.inputs["vdsobrec"],
                    st.session_state.inputs["vnsobrec"],
                    True,
                    0.5,
                )
                df = df[df["Tcnom"] != 249.9]

                PontosCombinados = pd.concat(
                    [
                        PontosEstacoes,
                        PontosDiretriz,
                    ],
                    ignore_index=True,
                )
                st.session_state.df_diretriz = PontosCombinados
                st.session_state.df_total = df
                st.session_state.df_pontos = estacoesselecionadas
        st.success("Análise Disponível na página 'Ver Mapa e Gráficos'")

        # endregion


elif menu == "Ver Mapa e Gráficos":
    st.title("Ver Resultados")
    st.session_state.df_diretriz.loc[
        st.session_state.df_diretriz["Estacao"].isin(st.session_state.df_pontos),
        "Classificação do Ponto",
    ] = "Dentro do Buffer"

    tab1, tab2 = st.tabs(["Mapa", "Intervalos de Medição"])
    with tab1:
        fig = px.scatter_mapbox(
            st.session_state.df_diretriz,
            lat="Latitude",
            lon="Longitude",
            hover_name="Estacao",
            zoom=3,
            color="Classificação do Ponto",
            center={
                "lat": st.session_state.df_diretriz["Latitude"].mean(),
                "lon": st.session_state.df_diretriz["Longitude"].mean(),
            },
            height=600,
            size_max=2,
        )

        fig.update_traces(marker=dict(size=10))
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig)

    st.sidebar.subheader("Estações dentro do Buffer")
    st.sidebar.table(pd.DataFrame(st.session_state.df_pontos, columns=["Nomes"]))

    with tab2:
        estacao_selecionada = st.multiselect(
            "Escolha as Estações",
            sorted(st.session_state.df_pontos),
        )
        anos = sorted(
            st.session_state.df_total["Data"].apply(lambda x: int(str(x)[:4])).unique()
        )
        ano_min, ano_max = st.slider(
            "Escolha o Intervalo de Anos",
            min_value=min(anos),
            max_value=max(anos),
            value=(min(anos), max(anos)),
        )
        anos_selecionados = list(range(ano_min, ano_max + 1))

        for estacao in sorted(estacao_selecionada):
            df_estacao = st.session_state.df_total[
                (st.session_state.df_total["Estacao"] == estacao)
                & (
                    st.session_state.df_total["Data"].apply(lambda x: int(str(x)[:4]))
                    >= ano_min
                )
                & (
                    st.session_state.df_total["Data"].apply(lambda x: int(str(x)[:4]))
                    <= ano_max
                )
            ]

            bins = range(20, 251, 5)
            labels = [f"{i}-{i+5}" for i in bins[:-1]]

            df_estacao["Intervalo"] = pd.cut(
                df_estacao["Tcnom"],
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=False,
            )

            df_grouped = (
                df_estacao.groupby("Intervalo", observed=True)
                .size()
                .reset_index(name="Count")
            )

            # Criar o gráfico
            fig = px.bar(
                df_grouped,
                x="Intervalo",
                y="Count",
                title=f"Estação: {estacao}",
                labels={"Intervalo": "Intervalo Tcnom (mm)", "Count": "Quantidade"},
                category_orders={"Intervalo": labels},  # Garantir ordem correta
                height=600,
            )

            # Melhorar a legibilidade dos rótulos
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
