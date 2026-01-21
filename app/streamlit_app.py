import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.markdown("## Deep Learning &\nFinance")
        st.caption("Prediction Quantitative Multi-Actifs")
        st.markdown("---")

        st.markdown("### Importation des donnees")
        uploaded_file = st.file_uploader(
            "Importer un fichier CSV",
            type=["csv"],
            label_visibility="collapsed",
        )

        st.markdown("### Choisissez le premier actif")
        asset = st.selectbox(
            "Choisir un actif",
            options=["", "AIR.PA", "BNP.PA", "CAP.PA", "OR.PA"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("### Date de debut")
        start_date = st.date_input("Date de debut", value=None)

        st.markdown("### Date de fin")
        end_date = st.date_input("Date de fin", value=None)

        st.markdown("### Parametres d'entrainement")
        window_size = st.number_input(
            "Taille de la fenetre",
            min_value=5,
            max_value=365,
            value=30,
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
        st.button("Generer les courbes", use_container_width=True)

    return {
        "uploaded_file": uploaded_file,
        "asset": asset,
        "start_date": start_date,
        "end_date": end_date,
        "window_size": window_size,
        "epochs": epochs,
        "model_type": model_type,
    }


def render_predictions_tab(state):
    st.subheader("Analyse Multi-Actifs")
    st.caption("Donnees historiques des prix")
    st.markdown(
        "Selectionnez des actifs et cliquez sur \"Generer les courbes\""
    )
    st.empty()


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
