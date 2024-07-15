import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

def normalize(matrix, criteria_types):
    norm_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[1]):
        if criteria_types[i] == "benefit":
            norm_matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
        elif criteria_types[i] == "cost":
            norm_matrix[:, i] = (matrix[:, i].max() - matrix[:, i]) / (matrix[:, i].max() - matrix[:, i].min())
    return norm_matrix

def weighted_normalized_matrix(norm_matrix, weights):
    return norm_matrix * weights

def calculate_power_weighted_matrix(norm_matrix, weights):
    return np.power(norm_matrix, weights)

def calculate_compromise_scores(weighted_norm_matrix, power_weighted_matrix):
    sum_scores = np.sum(weighted_norm_matrix, axis=1)
    product_scores = np.sum(power_weighted_matrix, axis=1)
    return sum_scores, product_scores

def combine_scores(sum_scores, product_scores):
    combined_scores = ((0.5 * sum_scores) + ((1 - 0.5) * product_scores)) / ((0.5 * np.max(sum_scores)) + ((1 - 0.5) * np.max(product_scores)))
    return combined_scores

def rank_alternatives(scores):
    return np.argsort(scores)[::-1]

def calculate_cocoso_weights(ka, kb, kc):
    weights = ((ka * kb * kb) ** (1/3)) + ((1/3) * (ka + kb + kc))
    return weights

def calculate_rank(scores):
    sorted_scores = sorted(scores, reverse=True)
    ranks = [sorted_scores.index(x) + 1 for x in scores]
    return np.array(ranks)

st.title("COCOSO Method for MCDM")

st.header("Input Decision Matrix")
decision_matrix_input = st.text_area("Enter decision matrix (rows separated by new lines, columns by commas):", "7,9,9\n6,7,8\n8,8,7\n7,6,6")
weights_input = st.text_input("Enter weights (comma-separated):", "0.4,0.35,0.25")

if st.button("Calculate Ranking"):
    decision_matrix = np.array([list(map(float, row.split(','))) for row in decision_matrix_input.split('\n')])
    weights = np.array(list(map(float, weights_input.split(','))))

    criteria_types_benefit = ["benefit"] * decision_matrix.shape[1]
    criteria_types_cost = ["cost"] * decision_matrix.shape[1]

    norm_matrix_benefit = normalize(decision_matrix, criteria_types_benefit)
    norm_matrix_cost = normalize(decision_matrix, criteria_types_cost)
    
    weighted_norm_matrix_benefit = weighted_normalized_matrix(norm_matrix_benefit, weights)
    weighted_norm_matrix_cost = weighted_normalized_matrix(norm_matrix_cost, weights)
    
    power_weighted_matrix_benefit = calculate_power_weighted_matrix(norm_matrix_benefit, weights)
    power_weighted_matrix_cost = calculate_power_weighted_matrix(norm_matrix_cost, weights)
    
    sum_scores_benefit, product_scores_benefit = calculate_compromise_scores(weighted_norm_matrix_benefit, power_weighted_matrix_benefit)
    sum_scores_cost, product_scores_cost = calculate_compromise_scores(weighted_norm_matrix_cost, power_weighted_matrix_cost)
    
    combined_scores_benefit = combine_scores(sum_scores_benefit, product_scores_benefit)
    combined_scores_cost = combine_scores(sum_scores_cost, product_scores_cost)
    
    ka_benefit = (sum_scores_benefit + product_scores_benefit) / np.sum(sum_scores_benefit + product_scores_benefit)
    kb_benefit = sum_scores_benefit / np.min(sum_scores_benefit) + product_scores_benefit / np.min(product_scores_benefit)
    
    ka_cost = (sum_scores_cost + product_scores_cost) / np.sum(sum_scores_cost + product_scores_cost)
    kb_cost = sum_scores_cost / np.min(sum_scores_cost) + product_scores_cost / np.min(product_scores_cost)
    
    rank_ka_benefit = calculate_rank(ka_benefit)
    rank_kb_benefit = calculate_rank(kb_benefit)
    rank_kc_benefit = calculate_rank(combined_scores_benefit)
    
    rank_ka_cost = calculate_rank(ka_cost)
    rank_kb_cost = calculate_rank(kb_cost)
    rank_kc_cost = calculate_rank(combined_scores_cost)
    
    cocoso_weights_benefit = calculate_cocoso_weights(ka_benefit, kb_benefit, combined_scores_benefit)
    cocoso_weights_cost = calculate_cocoso_weights(ka_cost, kb_cost, combined_scores_cost)
    
    st.subheader("Initial Decision Matrix")
    df_decision_matrix = pd.DataFrame(decision_matrix, columns=[f"C{i+1}" for i in range(decision_matrix.shape[1])])
    st.dataframe(df_decision_matrix)
    
    st.subheader("Normalized Decision Matrix (Benefit Criteria)")
    df_norm_matrix_benefit = pd.DataFrame(norm_matrix_benefit, columns=[f"C{i+1}" for i in range(norm_matrix_benefit.shape[1])])
    st.dataframe(df_norm_matrix_benefit)
    
    st.subheader("Normalized Decision Matrix (Cost Criteria)")
    df_norm_matrix_cost = pd.DataFrame(norm_matrix_cost, columns=[f"C{i+1}" for i in range(norm_matrix_cost.shape[1])])
    st.dataframe(df_norm_matrix_cost)
    
    st.subheader("Sum of Weighted Normalized Matrix (Benefit Criteria)")
    df_weighted_norm_matrix_benefit = pd.DataFrame(weighted_norm_matrix_benefit, columns=[f"C{i+1}" for i in range(weighted_norm_matrix_benefit.shape[1])])
    st.dataframe(df_weighted_norm_matrix_benefit)
    
    st.subheader("Sum of Weighted Normalized Matrix (Cost Criteria)")
    df_weighted_norm_matrix_cost = pd.DataFrame(weighted_norm_matrix_cost, columns=[f"C{i+1}" for i in range(weighted_norm_matrix_cost.shape[1])])
    st.dataframe(df_weighted_norm_matrix_cost)
    
    st.subheader("Power of Weighted Normalized Matrix (Benefit Criteria)")
    df_power_weighted_matrix_benefit = pd.DataFrame(power_weighted_matrix_benefit, columns=[f"C{i+1}" for i in range(power_weighted_matrix_benefit.shape[1])])
    st.dataframe(df_power_weighted_matrix_benefit)
    
    st.subheader("Power of Weighted Normalized Matrix (Cost Criteria)")
    df_power_weighted_matrix_cost = pd.DataFrame(power_weighted_matrix_cost, columns=[f"C{i+1}" for i in range(power_weighted_matrix_cost.shape[1])])
    st.dataframe(df_power_weighted_matrix_cost)
    
    st.subheader("Comparability Sequence Measures (Benefit Criteria)")
    st.write("Si (Sum Scores):", sum_scores_benefit)
    st.write("Pi (Product Scores):", product_scores_benefit)
    
    st.subheader("Comparability Sequence Measures (Cost Criteria)")
    st.write("Si (Sum Scores):", sum_scores_cost)
    st.write("Pi (Product Scores):", product_scores_cost)
    
    st.subheader("Final Aggregation and Rankings (Benefit Criteria)")
    ranking_data_benefit = {
        'Alternative': [f"A{i+1}" for i in range(len(sum_scores_benefit))],
        'Ka': ka_benefit,
        'Rank (Ka)': rank_ka_benefit,
        'Kb': kb_benefit,
        'Rank (Kb)': rank_kb_benefit,
        'Kc (Combined Scores)': combined_scores_benefit,
        'Rank (Kc)': rank_kc_benefit,
        'K (Final Weights)': cocoso_weights_benefit,
        'Global Rank (K)': calculate_rank(cocoso_weights_benefit)
    }
    df_ranking_benefit = pd.DataFrame(ranking_data_benefit)
    st.dataframe(df_ranking_benefit)
    
    st.subheader("Final Aggregation and Rankings (Cost Criteria)")
    ranking_data_cost = {
        'Alternative': [f"A{i+1}" for i in range(len(sum_scores_cost))],
        'Ka': ka_cost,
        'Rank (Ka)': rank_ka_cost,
        'Kb': kb_cost,
        'Rank (Kb)': rank_kb_cost,
        'Kc (Combined Scores)': combined_scores_cost,
        'Rank (Kc)': rank_kc_cost,
        'K (Final Weights)': cocoso_weights_cost,
        'Global Rank (K)': calculate_rank(cocoso_weights_cost)
    }
    df_ranking_cost = pd.DataFrame(ranking_data_cost)
    st.dataframe(df_ranking_cost)
    
    st.subheader("COCOSO Weights (Benefit Criteria)")
    st.write("Global Weights of the alternatives (Benefit):", cocoso_weights_benefit)
    
    st.subheader("COCOSO Weights (Cost Criteria)")
    st.write("Global Weights of the alternatives (Cost):", cocoso_weights_cost)
    
    index = np.arange(len(cocoso_weights_benefit))
    labels = [f"A{i+1}" for i in range(len(cocoso_weights_benefit))]
    
    fig_benefit = px.line(x=index, y=[sum_scores_benefit, product_scores_benefit, combined_scores_benefit, cocoso_weights_benefit], 
                          labels={'x': 'Solutions', 'y': 'Weights'}, 
                          title='Benefit Criteria Weights',
                          markers=True)
    fig_benefit.update_layout(xaxis=dict(tickmode='array', tickvals=index, ticktext=labels))
    fig_benefit.data[0].name = 'Si (Sum Scores)'
    fig_benefit.data[1].name = 'Pi (Product Scores)'
    fig_benefit.data[2].name = 'Kc (Combined Scores)'
    fig_benefit.data[3].name = 'K (Final Weights)'
    
    fig_cost = px.line(x=index, y=[sum_scores_cost, product_scores_cost, combined_scores_cost, cocoso_weights_cost], 
                       labels={'x': 'Solutions', 'y': 'Weights'}, 
                       title='Cost Criteria Weights',
                       markers=True)
    fig_cost.update_layout(xaxis=dict(tickmode='array', tickvals=index, ticktext=labels))
    fig_cost.data[0].name = 'Si (Sum Scores)'
    fig_cost.data[1].name = 'Pi (Product Scores)'
    fig_cost.data[2].name = 'Kc (Combined Scores)'
    fig_cost.data[3].name = 'K (Final Weights)'
    
    st.plotly_chart(fig_benefit)
    st.plotly_chart(fig_cost)
