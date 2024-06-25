import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(matrix):
    norm_matrix = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))
    return norm_matrix

def weighted_normalized_matrix(norm_matrix, weights):
    return norm_matrix * weights

def calculate_compromise_scores(weighted_norm_matrix):
    sum_scores = np.sum(weighted_norm_matrix, axis=1)
    product_scores = np.prod(weighted_norm_matrix, axis=1)
    return sum_scores, product_scores

def combine_scores(sum_scores, product_scores):
    combined_scores = 0.5 * sum_scores + 0.5 * product_scores
    return combined_scores

def rank_alternatives(scores):
    return np.argsort(scores)[::-1]

def calculate_cocoso_weights(combined_scores):
    total_score = np.sum(combined_scores)
    weights = combined_scores / total_score
    return weights

st.title("COCOSO Method for MCDM")

st.header("Input Decision Matrix")
decision_matrix_input = st.text_area("Enter decision matrix (rows separated by new lines, columns by commas):", "7,9,9\n6,7,8\n8,8,7\n7,6,6")
weights_input = st.text_input("Enter weights (comma-separated):", "0.4,0.35,0.25")

if st.button("Calculate Ranking"):
    # Parse inputs
    decision_matrix = np.array([list(map(float, row.split(','))) for row in decision_matrix_input.split('\n')])
    weights = np.array(list(map(float, weights_input.split(','))))

    # Normalize decision matrix
    norm_matrix = normalize(decision_matrix)
    
    # Weighted normalized matrix
    weighted_norm_matrix = weighted_normalized_matrix(norm_matrix, weights)
    
    # Calculate Si and Pi
    sum_scores, product_scores = calculate_compromise_scores(weighted_norm_matrix)
    
    # Final combined scores
    combined_scores = combine_scores(sum_scores, product_scores)
    
    # Rankings
    rank_ka = rank_alternatives(sum_scores)
    rank_kb = rank_alternatives(product_scores)
    rank_kc = rank_alternatives(combined_scores)
    
    # COCOSO weights
    cocoso_weights = calculate_cocoso_weights(combined_scores)
    
    # Create DataFrames for better visualization
    df_decision_matrix = pd.DataFrame(decision_matrix, columns=[f"C{i+1}" for i in range(decision_matrix.shape[1])])
    df_norm_matrix = pd.DataFrame(norm_matrix, columns=[f"C{i+1}" for i in range(norm_matrix.shape[1])])
    df_weighted_norm_matrix = pd.DataFrame(weighted_norm_matrix, columns=[f"C{i+1}" for i in range(weighted_norm_matrix.shape[1])])
    
    # Display matrices and scores
    st.subheader("Initial Decision Matrix")
    st.dataframe(df_decision_matrix)
    
    st.subheader("Normalized Decision Matrix")
    st.dataframe(df_norm_matrix)
    
    st.subheader("Weighted Normalized Matrix")
    st.dataframe(df_weighted_norm_matrix)
    
    st.subheader("Comparability Sequence Measures")
    st.write("Si (Sum Scores):", sum_scores)
    st.write("Pi (Product Scores):", product_scores)
    
    st.subheader("Final Aggregation and Rankings")
    ranking_data = {
        'Alternative': [f"A{i+1}" for i in range(len(sum_scores))],
        'Ka (Sum Scores)': sum_scores,
        'Rank (Ka)': rank_ka + 1,
        'Kb (Product Scores)': product_scores,
        'Rank (Kb)': rank_kb + 1,
        'Kc (Combined Scores)': combined_scores,
        'Rank (Kc)': rank_kc + 1,
        'K (Final Weights)': cocoso_weights,
        'Global Rank (K)': rank_alternatives(cocoso_weights) + 1
    }
    df_ranking = pd.DataFrame(ranking_data)
    st.dataframe(df_ranking)
    
    st.subheader("COCOSO Weights")
    st.write("Global Weights of the alternatives:", cocoso_weights)
    
    # Plotting the relative and final weights
    fig, ax = plt.subplots()
    index = np.arange(len(cocoso_weights))
    bar_width = 0.2
    
    bar1 = ax.bar(index, sum_scores, bar_width, label='Ka (Sum Scores)')
    bar2 = ax.bar(index + bar_width, product_scores, bar_width, label='Kb (Product Scores)')
    bar3 = ax.bar(index + 2 * bar_width, combined_scores, bar_width, label='Kc (Combined Scores)')
    bar4 = ax.bar(index + 3 * bar_width, cocoso_weights, bar_width, label='K (Final Weights)')
    
    ax.set_xlabel('Solutions')
    ax.set_ylabel('Weights')
    ax.set_title('Relative and final weights of each solution')
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels([f"A{i+1}" for i in range(len(cocoso_weights))])
    ax.legend()
    
    st.pyplot(fig)
