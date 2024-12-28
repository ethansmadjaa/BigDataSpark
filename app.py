import gc

import streamlit as st

from analysis import analyze_data
from exploration import explore_data
from preprocessing import preprocess_data
from utils.constants import STOCK_CATEGORIES
from utils.risk_warnings import show_risk_warning
from utils.spark_utils import create_spark_session, cleanup_spark_cache
from utils.stock_utils import get_stock_info, format_market_cap, get_ytd_days


def main():
    # TODO: ajouter un cache pour les données stock (opti perf)
    # TODO: fix le bug avec le selectbox qui reset qd on change de tab

    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # note perso: refactor cette partie dans un fichier séparé
    with st.expander("ℹ️ About this project", expanded=True):
        st.markdown("""
        ### 📚 ECE Paris - M1 DATA & AI Project
        
        **Big Data Frameworks Course Project**
        
        Ce dashboard a été dev par:
        - **Ethan SMADJA**
        - **Tom URBAN**
        
        Dans le cadre du cours Big Data Frameworks en M1 DATA & AI à ECE Paris.
        
        ⚠️ **ATTENTION AUX RISQUES**
        
        Cet outil est uniquement éducatif:
        - Les perfs passées garantissent pas le futur 
        - Pas de décision basée uniquement sur l'analyse technique
        - Faites vos recherches et voyez des pros
        - Le trading c dangereux lol
        
        ---
        """)

    # css un peu crade mais ça marche
    st.markdown("""
            <style>
            .main { padding: 0rem 1rem; }
            .stTabs [data-baseweb="tab-list"] { gap: 2px; }
            .stTabs [data-baseweb="tab"] { padding: 10px 20px; }

            .stock-info {
                background-color: #262730;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid #464B5C;
            }
            .stock-info h4 {
                color: #FFFFFF; 
                margin: 0 0 10px 0;
                font-size: 1.1em;
            }
            .stock-info p {
                color: #FAFAFA;
                margin: 5px 0;
                font-size: 0.9em;
            }
            .stock-info .highlight {
                color: #00CC96;
                font-weight: bold;
            }
            .stock-info .label {
                color: #9BA1B9;
                margin-right: 5px;
            }
            </style>
        """, unsafe_allow_html=True)

    st.title("📈 Stock Analysis Dashboard")

    st.sidebar.title("Stock Selection")

    # choix du mode de selection des stocks
    selection_mode = st.sidebar.radio(
        "Comment choisir le stock ?",
        ["Category", "Custom Ticker"],
        key="main_selection_method"
    )

    if selection_mode == "Category":
        # selection par categorie 
        categ = st.sidebar.selectbox(
            "Choisis une catégorie",
            list(STOCK_CATEGORIES.keys()),
            key="main_category"
        )
        stock_opts = STOCK_CATEGORIES[categ]
        selected_stock = st.sidebar.selectbox(
            "Choisis ton stock",
            list(stock_opts.keys()),
            format_func=lambda x: f"{x} - {stock_opts[x]}",  # affiche symbole + nom
            key="main_stock"
        )
    else:
        # saisie manuelle du symbole
        selected_stock = st.sidebar.text_input(
            "Entre un symbole boursier",
            value="AAPL",  # valeur par defaut
            max_chars=5,
            key="main_ticker"
        ).upper()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Période")

    # periodes predefinies
    periods = {
        "1 Mois": 30,
        "3 Mois": 90,
        "6 Mois": 180,
        "YTD": get_ytd_days(),  # jours depuis 1er janvier
        "1 An": 365,
        "2 Ans": 730,
        "5 Ans": 1825
    }

    selected_period = st.sidebar.select_slider(
        "Choisis une période",
        options=list(periods.keys())
    )
    nb_jours = periods[selected_period]

    # init spark 
    spark = create_spark_session()

    # recup infos du stock
    stock_df = get_stock_info(selected_stock, spark)
    if stock_df:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Détails du stock")

        current_stock_info = stock_df.where(stock_df.ticker == selected_stock).collect()

        if current_stock_info:
            nom = current_stock_info[0]['name']
            prix = current_stock_info[0]['current_price']
            variation = current_stock_info[0]['price_change']
            vol = current_stock_info[0]['volume']
            market_cap = format_market_cap(current_stock_info[0]['market_cap'])
            secteur = current_stock_info[0]['sector']

            # affichage des infos avec couleur conditionnelle pour la variation
            variation_color = "#FF4B4B" if variation < 0 else "#00CC96"  # rouge si négatif, vert si positif
            display_html = f"""
                <div class="stock-info">
                <h4>{selected_stock} - {nom if nom else 'N/A'}</h4>
                <p><span class="label">Prix:</span> <span class="highlight">
                    {f'${prix:.2f}' if prix is not None else 'N/A'}</span></p>
                <p><span class="label">Variation (24h):</span> <span style="color: {variation_color}; font-weight: bold;">
                    {f'{variation:+.2f}%' if variation is not None else 'N/A'}</span></p>
                <p><span class="label">Volume:</span> 
                    {f'{int(vol):,}' if vol is not None else 'N/A'}</p>
                <p><span class="label">Capitalisation:</span> {market_cap if market_cap else 'N/A'}</p>
                <p><span class="label">Secteur:</span> {secteur if secteur else 'N/A'}</p>
                </div>
            """

            st.sidebar.markdown(display_html, unsafe_allow_html=True)
        else:
            st.sidebar.warning("Pas de données dispo pour ce stock")

    # warning risques
    show_risk_warning()

    # tabs pour les diff analyses
    tab1, tab2, tab3 = st.tabs(["Explorer", "Traiter", "Analyser"])

    with tab1:
        explore_data(spark, selected_stock, nb_jours)
    with tab2:
        preprocess_data(spark, selected_stock, nb_jours)
    with tab3:
        analyze_data(spark, selected_stock, nb_jours)

    # nettoyage memoire
    cleanup_spark_cache(spark)
    gc.collect()

    if 'spark' in locals():
        spark.stop()


if __name__ == "__main__":
    main()
