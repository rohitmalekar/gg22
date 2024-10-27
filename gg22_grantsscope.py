import streamlit as st
import pandas as pd

# Set the Streamlit page configuration to wide mode
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([0.25, 2, 0.25])  # Adjust the width ratios as needed

# Function to normalize a series using Min-Max normalization
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# List of metrics
metrics = [
    "star_count",
    "fork_count",
    "contributor_count_6_months",
    "new_contributor_count_6_months",
    "fulltime_developer_average_6_months",
    "active_developer_count_6_months",
    "commit_count_6_months",
    "merged_pull_request_count_6_months",
    "closed_issue_count_6_months"
]

# Title of the app
col2.title("GrantsScope for GG22") 
col2.markdown("### Powered by Open Source Observer & RegenData") 
col2.caption("Created by Rohit Malekar | [grantsscope.xyz](grantsscope.xyz)")

col2.markdown(""" Since June 2023, GrantsScope has been pioneering data-driven innovations to enhance Gitcoin Grants. We've launched personalized, AI-powered tools like RAG (Retrieval Augmented Generation) experiences and one-click grantee recommendations, making it easier for donors to explore impactful projects based on donation history and advanced clustering algorithms. """)

col2.markdown(""" In GG22, we're taking it a step further. Donors can now create their own composite impact metrics, powered by Open Source Observer, and discover how grantees align with their values and goals. If you find value in these insights, support GrantsScope and contribute to more innovations for the community. """)

col2.link_button("Support GrantsScope in GG22", "https://explorer.gitcoin.co/#/round/42161/608/77", type="primary")
scol1, scol2, scol3 = st.columns([1, 1.5, 1])  # Adjust the width ratios as needed
selected_metrics = scol2.multiselect("Select up to 3 metrics:", metrics, max_selections=3)

# Total weight variable
total_weight = 100
assigned_weight = 0

# Initialize weights dictionary
weights = {metric: 0 for metric in selected_metrics}

# Input for weights using sliders
if selected_metrics:
    scol2.write("Assign weights (total must equal 100):")
    remaining_weight = total_weight

    for metric in selected_metrics:
        # Ensure the remaining weight is not negative
        weight = scol2.slider(
            f"Weight for {metric}:",
            min_value=0,
            max_value=100,
            value=0,
            step=5  # Set the step to 5
        )
        weights[metric] = weight
        remaining_weight -= weight

    # Display the total assigned weight
    assigned_weight = sum(weights.values())
    scol2.write(f"Total assigned weight: {assigned_weight} / {total_weight}")


    # Check if the total weight is valid
    if assigned_weight > total_weight:
        scol2.error("Total weight exceeds 100. Please adjust the weights.")
    elif assigned_weight < total_weight:
        scol2.warning("Total weight is less than 100. Please adjust the weights.")
    else:   
        scol2.success("Weights are valid!")     
        # Add a refresh button that is enabled if weights are valid
        if scol2.button("Apply"):
            
            # Read the CSV files from the data folder
            oso_df = pd.read_csv('./gg22/data/GG22 GrantsScope OSO - oso.csv')
            regendata_df = pd.read_csv('./gg22/data/GG22 GrantsScope OSO - regendata.csv')

            # Remove the prefix "https://github.com/" from the project_github column
            regendata_df['oso_project_name'] = regendata_df['project_github'].str.replace("https://github.com/", "", regex=False)

            # Perform a left join between regendata and oso on the specified columns
            merged_df = regendata_df.merge(oso_df, left_on='oso_project_name', right_on='project_name', how='inner')

            # Normalize the selected metrics in merged_df
            for metric in selected_metrics:
                merged_df[f'normalized_{metric}'] = min_max_normalize(merged_df[metric])

            # Calculate the Development Activity Index
            merged_df['composite_score'] = round(
                sum(
                merged_df[f'normalized_{metric}'] * (weights[metric] / total_weight) for metric in selected_metrics
                ) *100
            )

            
            # Create a new DataFrame with the specified columns and rename them
            display_df = merged_df[[
                'project_title',
                'round_name',
                'explorer_link',
                'composite_score',
                'star_count',
                'fork_count',
                'contributor_count_6_months',
                'new_contributor_count_6_months',
                'fulltime_developer_average_6_months',
                'active_developer_count_6_months',
                'commit_count_6_months',
                'merged_pull_request_count_6_months',
                'closed_issue_count_6_months'
            ]].rename(columns={
                'project_title': 'Project',
                'round_name': 'Round',
                'explorer_link': 'Donation Link',
                'composite_score' : 'Composite Score (0 to 100)',
                'star_count': 'Star Count',
                'fork_count': 'Fork Count',
                'contributor_count_6_months': 'Contributor Count (6 Month)',
                'new_contributor_count_6_months': 'New Contributor Count (6 Month)',
                'fulltime_developer_average_6_months': 'Full Time Dev Average (6 Month)',
                'active_developer_count_6_months': 'Active Dev Count (6 Month)',
                'commit_count_6_months': 'Commit Count (6 Month)',
                'merged_pull_request_count_6_months': 'Merged PR Count (6 Month)',
                'closed_issue_count_6_months': 'Closed Issue Count (6 Month)',
            })

            # Sort the DataFrame by Composite Score in descending order
            display_df = display_df.sort_values(by='Composite Score (0 to 100)', ascending=False)

            # Display the modified DataFrame
            # Display the DataFrame with hyperlinks
            
            # st.dataframe(merged_df)
            st.markdown("#### Customized Impact Scores for GG22 Projects")
            st.markdown("Below is a tailored ranking of GG22 projects based on the metrics you selected and the weights you assigned. The composite score, calculated on a scale from 0 to 100, reflects how each project aligns with your preferences. Explore the detailed data for each project, and if you find one that resonates with your goals, consider supporting it through a direct donation.")
            st.caption("Note: Projects that applied after the guarnteed review date might not showcase in this analysis.")

            with st.expander("Understanding the Composite Score Calculation"):
                st.markdown('''                            
                    The composite score is a key metric that helps us evaluate and compare different projects based on various performance indicators. Here’s a simple breakdown of how we calculate this score:
                    1. **Normalization of Metrics:** We start by normalizing each selected metric. Normalization is a process that adjusts the values of these metrics to a common scale, typically between 0 and 1. This ensures that no single metric disproportionately influences the overall score due to its scale.
                    2. **Weight Assignment**: Each metric is assigned a weight by the user based on its importance. You can think of weights as the significance of each metric in the overall evaluation. 
                    3. **Weighted Sum Calculation:** The composite score is calculated by taking the weighted sum of the normalized metrics. This means we multiply each normalized metric by its corresponding weight and then sum these values together. The multiplication by 100 scales the score to a percentage format, making it easier to interpret. The formula looks like this:
                    > Composite Score=(∑(Normalized Metric×Weight))×100
                    4. **Rounding:** Finally, we round the composite score to the nearest whole number for simplicity. This gives us a final score that is easy to read and understand.
                    ''')

            st.dataframe(
                display_df,
                column_config={
                    "Donation Link": st.column_config.LinkColumn(
                        "Donation Link",
                        help="Click to donate",
                        display_text="Donate Here"
                    ),
                },
                hide_index=True,
            )
            
            

            # Split the DataFrame into different DataFrames based on the round names
            #rounds = [
            #    "GG22 OSS - dApps and Apps",
            #    "GG22 OSS - Developer Tooling and Libraries",
            #    "GG22 OSS - Hackathon Alumni",
            #    "GG22 OSS - Web3 Infrastructure"
            #]

            # Create a dictionary to hold the DataFrames
            #dfs = {Round: display_df[display_df['Round'] == Round] for Round in rounds}

            # Assign all other records as "Community Rounds"
            #dfs["Community Rounds"] = display_df[~display_df['Round'].isin(rounds)]

            # Create tabs for each DataFrame
            #tab_names = list(dfs.keys())
            #tabs = st.tabs(tab_names)

            # Display each DataFrame in its respective tab
            #for tab, Round in zip(tabs, tab_names):
            #    with tab:
            #        st.subheader(Round)
            #        st.dataframe(
            #            dfs[Round],
            #            column_config={
            #                "Donation Link": st.column_config.LinkColumn(
            #                   "Donation Link",
            #                    help="Click to donate",
            #                    display_text="Donate Here"
            #                ),
            #            },
            #            hide_index=True,
            #        )
            #
            

