import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# used Claude to help format this cct.py file 
# loading the data: 
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
        # take informant IDs and remove from dataframe
        informant_ids = df['Informant'].tolist()
        data_df = df.drop(columns=['Informant'])
        # convert to numpy array 
        data_array = data_df.values.astype(int)
        return data_array, informant_ids
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# defining priors + implementing the model: 
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
        # prior for competence (D): Beta(2,1) slightly favors higher competence
        # but still allows for a wide range of values
        # bounded to be ≥ 0.5
        D_raw = pm.Beta('D_raw', alpha=2, beta=1, shape=N)
        D = pm.Deterministic('D', 0.5 + 0.5 * D_raw)
        
        # prior for consensus answers (Z): uninformative Bernoulli(0.5)
        Z = pm.Bernoulli('Z', p=0.5, shape=M)
        
        # calculate probabilities for each informant-item pair
        # need to reshape D 
        D_reshaped = D[:, None]  # Shape: (N, 1)
        
        # p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
        
        # likelihood of observed data
        X = pm.Bernoulli('X', p=p, observed=data)
    
    return model

# performing inference: 
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
        # run MCMC sampling
        trace = pm.sample(
            draws=draws,
            chains=chains,
            tune=tune,
            random_seed=42,
            return_inferencedata=True
        )
    
    return trace

# analyzing results: 
def analyze_competence(trace, informant_ids=None):
    """
    Analyze the competence estimates from the trace.
    
    Args:
        trace: PyMC trace object
        informant_ids: List of informant IDs
        
    Returns:
        competence_df: DataFrame with competence estimates
    """
    # get summary statistics for competence
    summary = az.summary(trace, var_names=['D'])
    
    # make dataframe for competence estimates
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
    # get summary statistics for consensus answers
    summary = az.summary(trace, var_names=['Z'])
    
    # calculate majority vote (naive aggregation)
    majority_vote = np.mean(data, axis=0) > 0.5
    majority_vote = majority_vote.astype(int)
    
    # make dataframe for consensus answers
    consensus_df = pd.DataFrame({
        'Question': range(1, len(summary) + 1),
        'Consensus_Prob': summary['mean'].values,
        'Consensus_Answer': np.round(summary['mean'].values).astype(int),
        'Majority_Vote': majority_vote
    })
    
    # add a column indicating if consensus differs from majority vote
    consensus_df['Differs_From_Majority'] = consensus_df['Consensus_Answer'] != consensus_df['Majority_Vote']
    
    return consensus_df

# visualizing results: 
def visualize_competence(trace, informant_ids=None, save_path=None):
    """
    Visualize the posterior distributions of informant competence.
    
    Args:
        trace: PyMC trace object
        informant_ids: List of informant IDs
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import arviz as az
    import numpy as np
    
    # get competence variable
    competence_var = trace.posterior["D"]
    n_informants = competence_var.shape[-1]
    
    # create a figure with informant competence posteriors
    plt.figure(figsize=(12, 8))
    
    # plot posterior distributions for each informant
    ax = az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
    
    # customize the y-axis labels with provided informant IDs
    if informant_ids and hasattr(ax, 'get_yticklabels'):
        try:
            labels = ax.get_yticklabels()
            new_labels = []
            
            for i, label in enumerate(labels):
                if i < n_informants:
                    # extract the index from the label text more safely
                    label_text = label.get_text()
                    if '[' in label_text and ']' in label_text:
                        try:
                            idx = int(label_text.split('[')[-1].strip(']'))
                            if 0 <= idx < len(informant_ids):
                                label.set_text(f"D[{informant_ids[idx]}]")
                        except (ValueError, IndexError):
                            # if there's any error parsing, leave the label as is
                            pass
            
            ax.set_yticklabels(labels)
        except Exception as e:
            print(f"Warning: Could not update y-axis labels: {e}")
    
    plt.title("Posterior Distributions of Informant Competence", fontsize=14)
    plt.xlabel("Competence Level (D)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # create a rank-ordered plot of competence estimates with HDI
    summary = az.summary(trace, var_names=['D'])
    
    # sort by mean competence
    sorted_indices = np.argsort(summary['mean'].values)[::-1]
    sorted_means = summary['mean'].values[sorted_indices]
    sorted_hdi_low = summary['hdi_3%'].values[sorted_indices]
    sorted_hdi_high = summary['hdi_97%'].values[sorted_indices]
    
    # create labels
    if informant_ids:
        sorted_labels = [informant_ids[i] for i in sorted_indices]
    else:
        sorted_labels = [f"Informant {i}" for i in sorted_indices]
    
    plt.figure(figsize=(10, 8))
    plt.errorbar(
        sorted_means, 
        range(len(sorted_means)), 
        xerr=np.vstack([sorted_means - sorted_hdi_low, sorted_hdi_high - sorted_means]),
        fmt='o', 
        capsize=5,
        markersize=8
    )
    
    plt.yticks(range(len(sorted_labels)), sorted_labels)
    plt.xlabel('Competence Estimate (D)', fontsize=12)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Minimum competence (0.5)')
    plt.grid(True, alpha=0.3)
    plt.title('Rank-Ordered Informant Competence with 94% HDI', fontsize=14)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_ranked.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_consensus(trace, save_path=None):
    """
    Visualize the posterior probabilities of consensus answers.
    
    Args:
        trace: PyMC trace object
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import arviz as az
    import numpy as np
    
    try:
        # get summary statistics for Z
        summary = az.summary(trace, var_names=['Z'])
        
        # plot posterior distributions
        plt.figure(figsize=(12, 8))
        az.plot_posterior(trace, var_names=['Z'], hdi_prob=0.94)
        plt.title('Posterior Distributions of Consensus Answers', fontsize=14)
        plt.xlabel('Probability of Answer = 1', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        # create a visual representation of the consensus answers
        n_questions = len(summary)
        
        # prepare data
        question_nums = [f"Q{i+1}" for i in range(n_questions)]
        consensus_probs = summary['mean'].values
        
        # sort by the certainty (how far from 0.5)
        certainty = np.abs(consensus_probs - 0.5)
        sorted_indices = np.argsort(certainty)[::-1]
        
        sorted_questions = [question_nums[i] for i in sorted_indices]
        sorted_probs = consensus_probs[sorted_indices]
        
        # create a color map (1=green, 0=red)
        colors = ['#ff6666' if p < 0.5 else '#66b266' for p in sorted_probs]
        values = [1-p if p < 0.5 else p for p in sorted_probs]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(sorted_questions, values, color=colors, alpha=0.7)
        
        # add annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            p = sorted_probs[i]
            answer = '1' if p >= 0.5 else '0'
            confidence = 100 * (1-p if p < 0.5 else p)
            plt.text(
                width + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f"Answer: {answer} ({confidence:.1f}%)",
                va='center'
            )
        
        plt.xlabel('Consensus Certainty', fontsize=12)
        plt.ylabel('Question', fontsize=12)
        plt.title('Consensus Answers Ranked by Certainty', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        plt.xlim(0.5, 1.0)
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_certainty.png'), dpi=300, bbox_inches='tight')
            
        plt.show()
    except Exception as e:
        print(f"Warning: Error in consensus visualization: {e}")
        print("Continuing with analysis...")

# compare w/ naive aggregation: 
def compare_consensus_with_majority(consensus_df, save_path=None):
    """
    Visualize the comparison between CCT consensus and majority vote.
    
    Args:
        consensus_df: DataFrame with consensus and majority vote data
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    try:
        # create a figure comparing consensus with majority vote
        plt.figure(figsize=(10, 8))
        
        # get questions where answers differ
        differ_df = consensus_df[consensus_df['Differs_From_Majority']]
        
        # if there are no differences, show a message and return
        if len(differ_df) == 0:
            plt.text(0.5, 0.5, "No differences between CCT consensus and majority vote", 
                    ha='center', va='center', fontsize=14)
            plt.title("Consensus vs Majority Vote", fontsize=14)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.show()
            return
        
        # data for visualization
        questions = differ_df['Question'].values
        consensus_probs = differ_df['Consensus_Prob'].values
        majority_votes = differ_df['Majority_Vote'].values
        
        # calculate approximation of proportion of informants who voted with majority
        majority_proportions = []
        for i, q in enumerate(questions):
            maj_vote = majority_votes[i]
            # approximate for visualization
            majority_proportions.append(0.75 if maj_vote == 1 else 0.25)
        
        # plot data
        x = np.arange(len(questions))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # plot CCT estimates
        rects1 = ax.bar(x - width/2, consensus_probs, width, label='CCT Consensus', color='skyblue')
        
        # plot majority vote proportions
        rects2 = ax.bar(x + width/2, majority_proportions, width, label='Majority Vote', color='lightcoral')
        
        # add some text and labels
        ax.set_ylabel('Probability / Proportion', fontsize=12)
        ax.set_xlabel('Question Number', fontsize=12)
        ax.set_title('Questions Where CCT Consensus Differs from Majority Vote', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{q}' for q in questions])
        ax.set_ylim(0, 1)
        ax.legend()
        
        # add a horizontal line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # add annotations
        for i, q in enumerate(questions):
            # add consensus answer
            cons_ans = '1' if consensus_probs[i] >= 0.5 else '0'
            ax.text(x[i] - width/2, consensus_probs[i] + 0.05, 
                    f'{cons_ans}', ha='center', fontsize=9)
            
            # add majority vote
            maj_ans = int(majority_votes[i])
            ax.text(x[i] + width/2, majority_proportions[i] + 0.05, 
                    f'{maj_ans}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    except Exception as e:
        print(f"Warning: Error in comparison visualization: {e}")
        print("Continuing with analysis...")

def compare_consensus_with_majority(consensus_df, save_path=None):
    """
    Visualize the comparison between CCT consensus and majority vote.
    
    Args:
        consensus_df: DataFrame with consensus and majority vote data
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # create a figure comparing consensus with majority vote
    plt.figure(figsize=(10, 8))
    
    # get questions where answers differ
    differ_df = consensus_df[consensus_df['Differs_From_Majority']]
    
    # data for visualization
    questions = differ_df['Question'].values
    consensus_probs = differ_df['Consensus_Prob'].values
    majority_votes = differ_df['Majority_Vote'].values
    
    # calculate proportion of informants who voted with majority
    majority_proportions = []
    for i, q in enumerate(questions):
        maj_vote = majority_votes[i]
        # Need to calculate this from original data, 
        # but we'll approximate for this visualization
        majority_proportions.append(0.75 if maj_vote == 1 else 0.25)
    
    # plot data
    x = np.arange(len(questions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # plot CCT estimates
    rects1 = ax.bar(x - width/2, consensus_probs, width, label='CCT Consensus', color='skyblue')
    
    # plot majority vote proportions
    rects2 = ax.bar(x + width/2, majority_proportions, width, label='Majority Vote', color='lightcoral')
    
    # add some text and labels
    ax.set_ylabel('Probability / Proportion', fontsize=12)
    ax.set_xlabel('Question Number', fontsize=12)
    ax.set_title('Questions Where CCT Consensus Differs from Majority Vote', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{q}' for q in questions])
    ax.set_ylim(0, 1)
    ax.legend()
    
    # add a horizontal line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # add annotations
    for i, q in enumerate(questions):
        # add consensus answer
        cons_ans = '1' if consensus_probs[i] >= 0.5 else '0'
        ax.text(x[i] - width/2, consensus_probs[i] + 0.05, 
                f'{cons_ans}', ha='center', fontsize=9)
        
        # add majority vote
        maj_ans = int(majority_votes[i])
        ax.text(x[i] + width/2, majority_proportions[i] + 0.05, 
                f'{maj_ans}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def main():
    """
    Main function to run the CCT analysis.
    """
    try:
        # set paths - using os.path for better compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_path = os.path.join(parent_dir, 'data', 'plant_knowledge.csv')
        
        # create output directory for visualizations
        output_dir = os.path.join(current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Looking for data at: {data_path}")
        
        # check if file exists
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
        
        # load the data
        data, informant_ids = load_data(data_path)
        if data is None:
            return
            
        print(f"Loaded data with {data.shape[0]} informants and {data.shape[1]} questions.")
        
        # build model
        model = build_cct_model(data)
        
        # run inference
        print("Running MCMC sampling...")
        trace = run_inference(model)
        
        # check convergence
        summary = az.summary(trace)
        print("\nConvergence diagnostics:")
        print(summary[['r_hat']])
        
        # analyze competence
        competence_df = analyze_competence(trace, informant_ids)
        print("\nInformant competence estimates:")
        print(competence_df)
        
        # analyze consensus
        consensus_df = analyze_consensus(trace, data)
        print("\nConsensus answer estimates:")
        print(consensus_df)
        
        # create simple report
        most_competent = competence_df.iloc[0]
        least_competent = competence_df.iloc[-1]
        diff_count = consensus_df['Differs_From_Majority'].sum()
        
        print("\n=== CCT ANALYSIS REPORT ===")
        print(f"\nCompetence ranges from {competence_df['Mean_Competence'].min():.3f} to {competence_df['Mean_Competence'].max():.3f}")
        print(f"Most competent informant: {most_competent['Informant']} ({most_competent['Mean_Competence']:.3f})")
        print(f"Least competent informant: {least_competent['Informant']} ({least_competent['Mean_Competence']:.3f})")
        print(f"\nConsensus answers differ from majority vote in {diff_count} out of {len(consensus_df)} questions ({100*diff_count/len(consensus_df):.1f}%)")
        
        # create visualizations
        print("\nCreating visualizations...")
        
        try:
            # visualize competence
            comp_fig_path = os.path.join(output_dir, 'competence_posterior.png')
            visualize_competence(trace, informant_ids, save_path=comp_fig_path)
            print(f"Competence visualization saved to {comp_fig_path}")
        except Exception as e:
            print(f"Warning: Error in competence visualization: {e}")
            print("Continuing with analysis...")
        
        try:
            # visualize consensus answers
            cons_fig_path = os.path.join(output_dir, 'consensus_posterior.png')
            visualize_consensus(trace, save_path=cons_fig_path)
            print(f"Consensus visualization saved to {cons_fig_path}")
        except Exception as e:
            print(f"Warning: Error in consensus visualization: {e}")
            print("Continuing with analysis...")
        
        # compare with majority vote
        if diff_count > 0:
            try:
                comp_fig_path = os.path.join(output_dir, 'consensus_vs_majority.png')
                compare_consensus_with_majority(consensus_df, save_path=comp_fig_path)
                print(f"Comparison visualization saved to {comp_fig_path}")
            except Exception as e:
                print(f"Warning: Error in comparison visualization: {e}")
                print("Continuing with analysis...")
        
        # check for convergence issues
        r_hat_issues = (summary['r_hat'] > 1.05).sum()
        if r_hat_issues > 0:
            print(f"\nWARNING: {r_hat_issues} parameters have r_hat values > 1.05, suggesting potential convergence issues.")
            print("You may want to increase the number of tuning steps or draws.")
        else:
            print("\nAll r_hat values are below 1.05, suggesting good convergence.")
        
        # create a simplified alternative visualization that doesn't rely on arviz plotting
        try:
            # create simplified competence plot
            plt.figure(figsize=(10, 6))
            comp_means = summary.loc[[f'D[{i}]' for i in range(len(informant_ids))], 'mean'].values
            comp_stds = summary.loc[[f'D[{i}]' for i in range(len(informant_ids))], 'sd'].values
            
            # sort by competence
            sorted_idx = np.argsort(comp_means)[::-1]
            sorted_means = comp_means[sorted_idx]
            sorted_stds = comp_stds[sorted_idx]
            sorted_ids = [informant_ids[i] for i in sorted_idx]
            
            plt.errorbar(range(len(sorted_means)), sorted_means, yerr=sorted_stds, fmt='o')
            plt.xticks(range(len(sorted_means)), sorted_ids, rotation=45)
            plt.axhline(y=0.5, color='red', linestyle='--')
            plt.title('Informant Competence (Sorted)')
            plt.ylabel('Competence')
            plt.xlabel('Informant ID')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'competence_simple.png'), dpi=300)
            plt.show()
            
            # create simplified consensus plot
            plt.figure(figsize=(12, 6))
            z_means = summary.loc[[f'Z[{i}]' for i in range(consensus_df.shape[0])], 'mean'].values
            z_stds = summary.loc[[f'Z[{i}]' for i in range(consensus_df.shape[0])], 'sd'].values
            
            # sort by how certain we are (distance from 0.5)
            certainty = np.abs(z_means - 0.5)
            cert_idx = np.argsort(certainty)[::-1]
            sorted_z = z_means[cert_idx]
            sorted_std = z_stds[cert_idx]
            q_nums = [f'Q{i+1}' for i in cert_idx]
            
            plt.errorbar(range(len(sorted_z)), sorted_z, yerr=sorted_std, fmt='o')
            plt.xticks(range(len(sorted_z)), q_nums, rotation=90)
            plt.axhline(y=0.5, color='gray', linestyle='--')
            plt.title('Consensus Answers (Sorted by Certainty)')
            plt.ylabel('Probability of Answer = 1')
            plt.xlabel('Question')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'consensus_simple.png'), dpi=300)
            plt.show()
            
            print(f"Simplified visualizations saved to {output_dir}")
        except Exception as e:
            print(f"Warning: Error in simplified visualization: {e}")
            print("Continuing with analysis...")
        
        print("\nAnalysis complete.")
        
    except Exception as e:
        print(f"Error in main analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
