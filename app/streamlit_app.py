from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from update_curves import DATA_DIR, fetch_asset, update_assets

BASE_TICKERS_PATH = Path(__file__).resolve().parent / "base_tickers.txt"

ASSET_CATEGORIES = {
    "Beautiful Seven (US)": [
        "AAPL",
        "AMZN",
        "GOOG",
        "META",
        "MSFT",
        "NVDA",
        "TSLA",
    ],
    "Tout CAC 40": [
        "ACA.PA",
        "AC.PA",
        "AI.PA",
        "AIR.PA",
        "ALO.PA",
        "BN.PA",
        "BNP.PA",
        "CA.PA",
        "CAP.PA",
        "CS.PA",
        "DG.PA",
        "DSY.PA",
        "EDEN.PA",
        "EN.PA",
        "ENGI.PA",
        "ERF.PA",
        "GLE.PA",
        "HO.PA",
        "KER.PA",
        "LR.PA",
        "MC.PA",
        "MT.PA",
        "ML.PA",
        "OR.PA",
        "PUB.PA",
        "RI.PA",
        "RMS.PA",
        "RNO.PA",
        "RUI.PA",
        "SAF.PA",
        "SAN.PA",
        "SGO.PA",
        "STMPA.PA",
        "SU.PA",
        "TEP.PA",
        "TTE.PA",
        "URW.PA",
        "VIE.PA",
        "VIV.PA",
        "WLN.PA",
    ],
    "Luxe": [
        "MC.PA",
        "RMS.PA",
        "KER.PA",
        "RCO.PA",
        "STMPA.PA",
    ],
    "Industrie": [
        "AI.PA",
        "EN.PA",
        "SU.PA",
        "VIE.PA",
        "DG.PA",
        "RNO.PA",
        "STLAP.PA",
        "FRVIA.PA",
        "AIR.PA",
    ],
    "Defense": [
        "AIR.PA",
        "HO.PA",
        "SAF.PA",
        "EXENS.PA",
        "TE.PA",
    ],
}


def load_base_tickers():
    if not BASE_TICKERS_PATH.exists():
        return []
    with BASE_TICKERS_PATH.open("r", encoding="utf-8") as handle:
        return [line.strip().upper() for line in handle if line.strip()]


def format_base_ticker(ticker):
    return ticker.strip().upper()


def normalize_ticker(ticker, base_tickers):
    ticker = ticker.strip().upper()
    if not ticker:
        return ""
    if "." not in ticker and f"{ticker}.PA" in base_tickers:
        return f"{ticker}.PA"
    return ticker


def normalize_category_name(name):
    return "".join(
        char.lower() if char.isalnum() else "_" for char in name
    ).strip("_")


def build_category_assets(known_assets):
    known_set = set(known_assets)
    category_assets = {}
    for category, tickers in ASSET_CATEGORIES.items():
        category_assets[category] = [
            ticker for ticker in tickers if ticker in known_set
        ]
    return category_assets


def get_asset_path(ticker):
    safe_name = ticker.replace("/", "_")
    return DATA_DIR / f"{safe_name}.csv"


@st.cache_data(show_spinner=False)
def load_asset_frame(ticker, start_date=None, end_date=None):
    path = get_asset_path(ticker)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = fetch_asset(ticker, start=start_date, end=end_date)

    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError(f"Missing Date column for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    if "Close" not in df.columns and "adjclose" in df.columns:
        df = df.rename(columns={"adjclose": "Close"})
    if "Close" not in df.columns:
        raise ValueError(f"Missing Close column for {ticker}")

    return df[["Date", "Close"]]


def render_sidebar():
    with st.sidebar:
        st.markdown("## Deep Learning &\nFinance")
        st.caption("Prediction Quantitative Multi-Actifs")
        st.markdown("---")

        st.markdown("### Actifs selectionnes")
        base_tickers = load_base_tickers()
        base_ticker_set = set(base_tickers)
        known_assets = [format_base_ticker(ticker) for ticker in base_tickers]
        if DATA_DIR.exists():
            for path in DATA_DIR.glob("*.csv"):
                known_assets.append(normalize_ticker(path.stem, base_ticker_set))
        known_assets = sorted(set(filter(None, known_assets)))
        category_assets = build_category_assets(known_assets)
        available_categories = [
            name
            for name, assets in category_assets.items()
            if assets
        ]
        default_categories = [
            name
            for name in ["Beautiful Seven (US)", "Tout CAC 40"]
            if name in available_categories
        ]
        selected_categories = st.multiselect(
            "Categories d'actions",
            options=available_categories,
            default=default_categories,
            key="selected_asset_categories",
            placeholder="Choisissez une ou plusieurs categories",
        )

        if "assets_initialized" not in st.session_state:
            initial_selected = {"AIR.PA", "BNP.PA"}
            if DATA_DIR.exists():
                for path in DATA_DIR.glob("*.csv"):
                    normalized = normalize_ticker(path.stem, base_ticker_set)
                    if normalized:
                        initial_selected.add(normalized)
            st.session_state.assets_initialized = True
        else:
            initial_selected = set()

        for category_name in selected_categories:
            category_tickers = category_assets[category_name]
            if not category_tickers:
                continue
            with st.expander(
                f"{category_name} ({len(category_tickers)} actifs)",
                expanded=False,
            ):
                category_key = normalize_category_name(category_name)
                check_col, uncheck_col = st.columns(2)
                if check_col.button(
                    "Cocher toute la categorie",
                    key=f"check_category_{category_key}",
                    use_container_width=True,
                ):
                    for asset in category_tickers:
                        st.session_state[f"asset_check_{asset}"] = True
                if uncheck_col.button(
                    "Decocher toute la categorie",
                    key=f"uncheck_category_{category_key}",
                    use_container_width=True,
                ):
                    for asset in category_tickers:
                        st.session_state[f"asset_check_{asset}"] = False

        visible_assets = sorted(
            {
                asset
                for category_name in selected_categories
                for asset in category_assets[category_name]
            }
        )

        if visible_assets:
            select_col, clear_col = st.columns(2)
            if select_col.button(
                "Cocher la selection", use_container_width=True
            ):
                for asset in visible_assets:
                    st.session_state[f"asset_check_{asset}"] = True
            if clear_col.button(
                "Decocher la selection", use_container_width=True
            ):
                for asset in visible_assets:
                    st.session_state[f"asset_check_{asset}"] = False
        else:
            st.info("Choisissez au moins une categorie d'actions.")

        selected_assets = []
        for asset in visible_assets:
            key = f"asset_check_{asset}"
            if key not in st.session_state:
                st.session_state[key] = asset in initial_selected
            if st.checkbox(asset, key=key):
                selected_assets.append(asset)

        stale_keys = [
            key
            for key in st.session_state
            if key.startswith("asset_check_")
            and key[len("asset_check_") :] not in known_assets
        ]
        for key in stale_keys:
            st.session_state.pop(key, None)

        if not selected_assets:
            st.write("Aucun actif selectionne")

        st.markdown("### Date de debut")
        start_date = st.date_input("Date de debut", value=None)

        st.markdown("### Date de fin")
        end_date = st.date_input("Date de fin", value=None)

        st.markdown("### Parametres d'entrainement")
        window_size = st.number_input(
            "Taille de la fenetre",
            min_value=5,
            max_value=365,
            value=60,
            step=1,
        )
        epochs = st.number_input(
            "Nombre d'epochs",
            min_value=1,
            max_value=500,
            value=100,
            step=1,
        )
        model_type = st.selectbox(
            "Modele",
            options=["LSTM", "CNN", "CNN puis LSTM"],
            index=0,
        )
        if "generate_curves" not in st.session_state:
            st.session_state.generate_curves = False
        if st.button("Generer les courbes", use_container_width=True):
            st.session_state.generate_curves = True

        st.markdown("---")
        st.markdown("### Mise a jour des donnees")
        today = datetime.now().strftime("%Y-%m-%d")
        update_key = f"data_updated_{today}"
        force_update = st.checkbox("Ecraser les donnees existantes", value=False)
        if st.button("Mettre a jour toutes les donnees", use_container_width=True):
            if not selected_assets:
                st.info("Aucun actif selectionne.")
            else:
                progress_bar = st.progress(0)
                status_line = st.empty()

                def on_progress(index, total, ticker, status, error=None):
                    if total:
                        progress_bar.progress(index / total)
                    message = f"{status.upper()}: {ticker}"
                    if error:
                        message = f"{message} - {error}"
                    status_line.caption(message)

                with st.spinner("Mise a jour en cours..."):
                    result = update_assets(
                        selected_assets,
                        start=start_date,
                        end=end_date,
                        force=force_update,
                        progress=on_progress,
                    )
                load_asset_frame.clear()
                st.session_state[update_key] = True
                st.success("Mise a jour terminee.")
                st.write(
                    f"Mis a jour: {len(result['updated'])} | "
                    f"Ignores: {len(result['skipped'])} | "
                    f"Echecs: {len(result['failed'])}"
                )
                if result["skipped"]:
                    st.info(
                        f"Actifs ignores (recents): {', '.join(result['skipped'])}"
                    )
                if result["failed"]:
                    with st.expander("Details des erreurs"):
                        for item in result["failed"]:
                            st.write(f"{item['ticker']}: {item['error']}")
        if st.session_state.get(update_key, False):
            st.success(f"Mis a jour aujourd'hui ({today})")
        else:
            st.info("Pas de mise a jour aujourd'hui")

        st.markdown("### Importation des donnees")
        uploaded_file = st.file_uploader(
            "Importer un fichier CSV",
            type=["csv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            target_path = DATA_DIR / uploaded_file.name
            if target_path.exists() and not force_update:
                st.warning("Fichier deja present. Activez l'option d'ecrasement.")
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded_file.getbuffer())
                load_asset_frame.clear()
                st.success(f"Fichier enregistre: {target_path.name}")

        st.markdown("### Fichiers disponibles")
        if DATA_DIR.exists():
            files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
        else:
            files = []
        if files:
            st.write(", ".join(files))
        else:
            st.caption("Aucun fichier dans data/assets")

    return {
        "uploaded_file": uploaded_file,
        "assets": selected_assets,
        "start_date": start_date,
        "end_date": end_date,
        "window_size": window_size,
        "epochs": epochs,
        "model_type": model_type,
        "generate_curves": st.session_state.generate_curves,
    }


def render_predictions_tab(state):
    st.subheader("Analyse Multi-Actifs")
    st.caption("Donnees historiques des prix")
    if not state.get("generate_curves"):
        st.markdown(
            "Selectionnez des actifs et cliquez sur \"Generer les courbes\""
        )
        return
    if not state["assets"]:
        st.info("Aucun actif selectionne.")
        return

    frames = []
    errors = []
    with st.spinner("Chargement des donnees..."):
        for ticker in state["assets"]:
            try:
                df = load_asset_frame(
                    ticker, state["start_date"], state["end_date"]
                )
                df = df.set_index("Date").rename(columns={"Close": ticker})
                frames.append(df)
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")

    if errors:
        st.warning("Certaines donnees n'ont pas pu etre chargees.")
        st.write("\n".join(errors))

    if not frames:
        st.info("Aucune donnee disponible.")
        return

    combined = pd.concat(frames, axis=1).sort_index()
    st.line_chart(combined)
    st.dataframe(combined.tail(10), use_container_width=True)


def render_allocation_tab(state):
    st.subheader("Allocation Optimisee du Portefeuille")
    st.caption("Entrainez un modele pour voir l'allocation optimisee")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Parametres d'allocation")
        st.selectbox(
            "Horizon d'investissement",
            options=["", "3 mois", "6 mois", "1 an", "3 ans"],
            index=0,
        )
    with col2:
        st.markdown("### Poids optimaux")
        st.info("Aucune allocation disponible")


def main():
    st.set_page_config(
        page_title="Deep Learning Finance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    state = render_sidebar()

    tab_predictions, tab_allocation = st.tabs(
        ["Visualisation des Previsions", "Allocation du Portefeuille"]
    )

    with tab_predictions:
        render_predictions_tab(state)

    with tab_allocation:
        render_allocation_tab(state)


if __name__ == "__main__":
    main()
