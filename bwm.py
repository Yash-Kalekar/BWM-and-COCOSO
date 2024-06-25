import streamlit as st
import pulp
import pandas as pd

def bwm(weights_best_to_others, weights_others_to_worst):
    n = len(weights_best_to_others)
    
    prob = pulp.LpProblem("BWM", pulp.LpMinimize)
    
    w = pulp.LpVariable.dicts("w", range(n), lowBound=0, cat='Continuous')
    xi = pulp.LpVariable("xi", lowBound=0, cat='Continuous')
    
    for i in range(n):
        prob += w[0] - weights_best_to_others[i] * w[i] <= xi
        prob += w[0] - weights_best_to_others[i] * w[i] >= -xi
        
        prob += w[i] - weights_others_to_worst[i] * w[n-1] <= xi
        prob += w[i] - weights_others_to_worst[i] * w[n-1] >= -xi

    prob += pulp.lpSum(w[i] for i in range(n)) == 1
    
    prob += xi
    
    prob.solve()
    
    weights = [pulp.value(w[i]) for i in range(n)]
    
    return weights, pulp.value(xi)

st.title("Best-Worst Method (BWM)")

st.write("""
This application calculates the weights and consistency ratio using the Best-Worst Method (BWM).
""")

st.sidebar.header("Input Weights")

weights_best_to_others = st.sidebar.text_area("Weights (Best-to-Others)", value="1,2,3,4,5")
weights_others_to_worst = st.sidebar.text_area("Weights (Others-to-Worst)", value="9,8,7,6,5")

weights_best_to_others = list(map(int, weights_best_to_others.split(',')))
weights_others_to_worst = list(map(int, weights_others_to_worst.split(',')))

if st.sidebar.button("Calculate"):
    weights, consistency_ratio = bwm(weights_best_to_others, weights_others_to_worst)
    
    results_df = pd.DataFrame({
        'Criterion': [f'C{i+1}' for i in range(len(weights))],
        'Weight': weights
    })
    
    st.subheader("Results(Local Weights)")
    st.table(results_df)
    st.write("Consistency Ratio:", consistency_ratio)
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='bwm_results.csv',
        mime='text/csv'
    )
