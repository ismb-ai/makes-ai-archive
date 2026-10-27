# main.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This file contains the main routine for our business analytics tool, wherein
# one finds: (i) a client and product-centric recommender system, (ii) an RFM
# and ABC analysis, (iii) a forecast and clustering tool for the time-evolution
# of product's and client's RFM, and (iv) a ROI/ROE analysis. The tool is imple-
# mented in the framework of streamlit's library in the format of a web app.

from recsys import *
from rfm import *
from trends import *
from roi import *
from webinterface import *

def main():
    # Display the introduction to the program.
    display_intro()

    # Upload a CSV file and store the returned file object.
    file = upload_csv_file()

    # If a file is uploaded successfully:
    if file is not None:
        # Process the data from the uploaded file, including computing RFM scores.
        df, rfm_scores = process_data(file)

        # Create product features based on the processed data and RFM scores.
        product_features = create_product_features(df, rfm_scores)

        # Initialize the interaction matrix as None.
        interaction_matrix = None

        # Select the analysis tab to determine the type of analysis to perform.
        selected_tab = select_analysis_tab()

        # Based on the selected tab, run the corresponding analysis function.
        if selected_tab == "Recomendações: Produtos para Clientes":
            run_recommendations_clients_to_products(df, interaction_matrix)
        elif selected_tab == "Recomendações: Clientes para Produtos":
            run_recommendations_products_to_clients(df, interaction_matrix)
        elif selected_tab == "Análise RFM":
            run_rfm_analysis(df)
        elif selected_tab == "Clientes Similares":
            run_client_similarity_analysis(df, interaction_matrix)
        elif selected_tab == "Tendências de RFM":
            run_rfm_trends_analysis(df)
        elif selected_tab == "Produtos Similares":
            run_product_analysis(df)
        elif selected_tab == "Análise ABC":
            run_abc_analysis(df)
        elif selected_tab == "Performance de Vendas":
            run_sales_performance_analysis(df, product_features)
        elif selected_tab == "Análise de Retorno de Investimento":
            run_roi_roe_analysis(df)
        elif selected_tab == "Análise de Retenção de Clientes":
            run_churn_analysis(df)
        elif selected_tab == "Detecção de Anomalias em Vendas":
            run_sales_anomaly_detection(df)
        elif selected_tab == "Relatório de Experiência de Produto":
            run_anomalous_products_and_recommendations(df)
        elif selected_tab == "Estratégia de Vendas para Clientes em Risco":
            run_client_risk_bayesian_analysis(df)
        elif selected_tab == "Deep RecSys":
            run_deeprecsys(df)

if __name__ == "__main__":
    main()
