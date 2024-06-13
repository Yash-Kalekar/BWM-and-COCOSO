import streamlit as st
import numpy as np

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

def rank_alternatives(combined_scores):
    ranking = np.argsort(combined_scores)[::-1]
    return ranking

st.title("COCOSO Method for MCDM")

st.header("Input Decision Matrix")
decision_matrix = st.text_area("Enter decision matrix (rows separated by new lines, columns by commas):", "7,9,9\n6,7,8\n8,8,7\n7,6,6")
weights = st.text_input("Enter weights (comma-separated):", "0.4,0.35,0.25")

if st.button("Calculate Ranking"):
    decision_matrix = np.array([list(map(float, row.split(','))) for row in decision_matrix.split('\n')])
    weights = np.array(list(map(float, weights.split(','))))

    norm_matrix = normalize(decision_matrix)
    weighted_norm_matrix = weighted_normalized_matrix(norm_matrix, weights)
    sum_scores, product_scores = calculate_compromise_scores(weighted_norm_matrix)
    combined_scores = combine_scores(sum_scores, product_scores)
    ranking = rank_alternatives(combined_scores)

    st.subheader("Ranking of Alternatives")
    st.write("Ranking (from best to worst):")
    st.write(ranking + 1)  # Adding 1 to convert from 0-based to 1-based indexing
