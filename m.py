import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Title of the app
st.title("Market Basket Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select a Section", ("Upload Data", "Data Sample", "Apriori Results", "FP-Growth Results"))

# File uploader to load the transaction data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Load data only if the user has uploaded a file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Store the dataframe globally (for use in other sections)
    st.session_state.df = df
    # Group by InvoiceNo to get transactions
    transactions = df.groupby('InvoiceNo')['Item'].apply(list).values.tolist()

    # Convert to one-hot encoding
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # Store the encoded dataframe globally (for use in other sections)
    st.session_state.df_encoded = df_encoded

# Handle each sidebar option and display results accordingly
if option == "Upload Data":
    st.subheader("Upload Your Transaction Data")
    st.write("Please upload your CSV file to begin the analysis.")

elif option == "Data Sample":
    if 'df' in st.session_state:
        st.subheader("Data Sample:")
        st.write(st.session_state.df.head())
    else:
        st.write("Please upload a CSV file first.")

elif option == "Apriori Results":
    if 'df_encoded' in st.session_state:
        st.subheader("Apriori Algorithm Results")
        frequent_itemsets_apriori = apriori(st.session_state.df_encoded, min_support=0.01, use_colnames=True)
        rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1)

        if not rules_apriori.empty:
            st.write(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.write("No frequent itemsets found for Apriori.")
    else:
        st.write("Please upload a CSV file and process the data first.")

elif option == "FP-Growth Results":
    if 'df_encoded' in st.session_state:
        st.subheader("FP-Growth Algorithm Results")
        frequent_itemsets_fpgrowth = fpgrowth(st.session_state.df_encoded, min_support=0.01, use_colnames=True)
        rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1)

        if not rules_fpgrowth.empty:
            st.write(rules_fpgrowth[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.write("No frequent itemsets found for FP-Growth.")
    else:
        st.write("Please upload a CSV file and process the data first.")
