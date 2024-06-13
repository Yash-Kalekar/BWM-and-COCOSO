import streamlit as st
import pulp

def bwm(weights_best_to_others, weights_others_to_worst):
    # Number of criteria
    n = len(weights_best_to_others)
    
    # Define the problem
    prob = pulp.LpProblem("BWM", pulp.LpMinimize)
    
    # Variables
    w = pulp.LpVariable.dicts("w", range(n), lowBound=0, cat='Continuous')
    xi = pulp.LpVariable("xi", lowBound=0, cat='Continuous')
    
    # Constraints
    for i in range(n):
        # Constraints for Best-to-Others (BO)
        prob += w[0] - weights_best_to_others[i] * w[i] <= xi
        prob += w[0] - weights_best_to_others[i] * w[i] >= -xi
        
        # Constraints for Others-to-Worst (OW)
        prob += w[i] - weights_others_to_worst[i] * w[n-1] <= xi
        prob += w[i] - weights_others_to_worst[i] * w[n-1] >= -xi

    # Sum of weights should be 1
    prob += pulp.lpSum(w[i] for i in range(n)) == 1
    
    # Objective function
    prob += xi
    
    # Solve the problem
    prob.solve()
    
    # Extract the results
    weights = [pulp.value(w[i]) for i in range(n)]
    
    return weights, pulp.value(xi)

# Streamlit UI
st.title("Best-Worst Method (BWM)")

st.write("""
This application calculates the weights and consistency ratio for given Best-to-Others and Others-to-Worst criteria weights using the Best-Worst Method (BWM).
""")

st.sidebar.header("Input Weights")

weights_best_to_others = st.sidebar.text_area("Weights (Best-to-Others)", value="1, 7, 6, 8, 6, 5, 5, 7, 6, 4, 6, 5, 7, 9, 6, 5, 4, 7, 6, 5")
weights_others_to_worst = st.sidebar.text_area("Weights (Others-to-Worst)", value="9, 8, 7, 8, 7, 6, 6, 8, 7, 5, 7, 6, 8, 1, 7, 6, 5, 8, 7, 6")

# Convert input strings to lists of integers
weights_best_to_others = list(map(int, weights_best_to_others.split(',')))
weights_others_to_worst = list(map(int, weights_others_to_worst.split(',')))

if st.sidebar.button("Calculate"):
    weights, consistency_ratio = bwm(weights_best_to_others, weights_others_to_worst)
    
    st.subheader("Results")
    st.write("Calculated Weights:", weights)
    st.write("Consistency Ratio:", consistency_ratio)

