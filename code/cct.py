import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load the plant knowledge dataset and convert it to appropriate format.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        data_array: NumPy array of shape (N, M) containing binary responses
        informant_ids: List of informant IDs
    """
    try:
        df = pd.read_csv(file_path)
        # Take informant IDs and remove from dataframe
        informant_ids = df['Informant'].tolist()
        data_df = df.drop(columns=['Informant'])
        # Convert to numpy array 
        data_array = data_df.values.astype(int)
        return data_array, informant_ids
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def build_cct_model(data):
    """
    Build the Cultural Consensus Theory model using PyMC.
    
    Args:
        data: Binary response data of shape (N, M) where:
              N = number of informants
              M = number of items/questions
              
    Returns:
        model: PyMC model
    """
    N, M = data.shape  # N=informants, M=items/questions
    
    with pm.Model() as model:
        # Prior for competence (D): Beta(2,1) slightly favors higher competence
        # but still allows for a wide range of values. Bounded to be ≥ 0.5
        # as per CCT assumptions
        D_raw = pm.Beta('D_raw', alpha=2, beta=1, shape=N)
        D = pm.Deterministic('D', 0.5 + 0.5 * D_raw)
        
        # Prior for consensus answers (Z): uninformative Bernoulli(0.5)
        Z = pm.Bernoulli('Z', p=0.5, shape=M)
        
        # Calculate probabilities for each informant-item pair
        # Need to reshape D to broadcast correctly
        D_reshaped = D[:, None]  # Shape: (N, 1)
        
        # p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
        
        # Likelihood of observed data
        X = pm.Bernoulli('X', p=p, observed=data)
    
    return model

def run_inference(model, draws=2000, chains=4, tune=1000):
    """
    Run MCMC sampling for the model.
    
    Args:
        model: PyMC model
        draws: Number of samples per chain
        chains: Number of chains
        tune: Number of tuning steps
        
    Returns:
        trace: PyMC trace object
    """
    with model:
        # Run MCMC sampling
        trace = pm.sample(
            draws=draws,
            chains=chains,
            tune=tune,
            random_seed=42,
            return_inferencedata=True
        )
    
    return trace

def analyze_competence(trace, informant_ids=None):
    """
    Analyze the competence estimates from the trace.
    
    Args:
        trace: PyMC trace object
        informant_ids: List of informant IDs
        
    Returns:
        competence_df: DataFrame with competence estimates
    """
    # Get summary statistics for competence
    summary = az.summary(trace, var_names=['D'])
    
    # Create dataframe for competence estimates
    competence_df = pd.DataFrame({
        'Informant': informant_ids if informant_ids else range(len(summary)),
        'Mean_Competence': summary['mean'].values,
        'SD_Competence': summary['sd'].values,
        'HDI_3%': summary['hdi_3%'].values,
        'HDI_97%': summary['hdi_97%'].values
    })
    
    competence_df = competence_df.sort_values(by='Mean_Competence', ascending=False)
    
    return competence_df

def analyze_consensus(trace, data):
    """
    Analyze the consensus answers from the trace.
    
    Args:
        trace: PyMC trace object
        data: Original binary response data
        
    Returns:
        consensus_df: DataFrame with consensus estimates
    """
    # Get summary statistics for consensus answers
    summary = az.summary(trace, var_names=['Z'])
    
    # Calculate majority vote (naive aggregation)
    majority_vote = np.mean(data, axis=0) > 0.5
    majority_vote = majority_vote.astype(int)
    
    # Create dataframe for consensus answers
    consensus_df = pd.DataFrame({
        'Question': range(1, len(summary) + 1),
        'Consensus_Prob': summary['mean'].values,
        'Consensus_Answer': np.round(summary['mean'].values).astype(int),
        'Majority_Vote': majority_vote
    })
    
    # Add a column indicating if consensus differs from majority vote
    consensus_df['Differs_From_Majority'] = consensus_df['Consensus_Answer'] != consensus_df['Majority_Vote']
    
    return consensus_df

def main():
    """
    Main function to run the CCT analysis.
    """
    # Set paths - using os.path for better compatibility
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'data', 'plant_knowledge.csv')
    
    print(f"Looking for data at: {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        alt_path = os.path.join(current_dir, '..', 'data', 'plant_knowledge.csv')
        print(f"Trying alternative path: {alt_path}")
        if os.path.exists(alt_path):
            data_path = alt_path
            print(f"Found data at alternative path")
        else:
            print("Please provide the correct path to the data file:")
            user_path = input("> ")
            if os.path.exists(user_path):
                data_path = user_path
            else:
                print(f"ERROR: Data file not found at {user_path}")
                return
    
    # Load the data
    data, informant_ids = load_data(data_path)
    if data is None:
        return
        
    print(f"Loaded data with {data.shape[0]} informants and {data.shape[1]} questions.")
    
    # Build model
    model = build_cct_model(data)
    
    # Run inference
    print("Running MCMC sampling...")
    trace = run_inference(model)
    
    # Check convergence
    summary = az.summary(trace)
    print("\nConvergence diagnostics:")
    print(summary[['r_hat']])
    
    # Analyze competence
    competence_df = analyze_competence(trace, informant_ids)
    print("\nInformant competence estimates:")
    print(competence_df)
    
    # Analyze consensus
    consensus_df = analyze_consensus(trace, data)
    print("\nConsensus answer estimates:")
    print(consensus_df)
    
    # Create simple report
    most_competent = competence_df.iloc[0]
    least_competent = competence_df.iloc[-1]
    diff_count = consensus_df['Differs_From_Majority'].sum()
    
    print("\n=== CCT ANALYSIS REPORT ===")
    print(f"\nCompetence ranges from {competence_df['Mean_Competence'].min():.3f} to {competence_df['Mean_Competence'].max():.3f}")
    print(f"Most competent informant: {most_competent['Informant']} ({most_competent['Mean_Competence']:.3f})")
    print(f"Least competent informant: {least_competent['Informant']} ({least_competent['Mean_Competence']:.3f})")
    print(f"\nConsensus answers differ from majority vote in {diff_count} out of {len(consensus_df)} questions ({100*diff_count/len(consensus_df):.1f}%)")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()