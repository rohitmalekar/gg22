import streamlit as st
import pandas as pd

# Set the Streamlit page configuration to wide mode
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([0.25, 2, 0.25])  # Adjust the width ratios as needed

# Function to normalize a series using Min-Max normalization
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Preserve state for filter selections using st.session_state
#if 'selected_columns' not in st.session_state:
#    st.session_state.selected_columns = []


# List of metrics
metrics = [
    "star_count",
    "fork_count",
#    "contributor_count_6_months",
#    "new_contributor_count_6_months",
#    "fulltime_developer_average_6_months",
#    "active_developer_count_6_months",
    "commit_count_6_months",
    "merged_pull_request_count_6_months",
    "closed_issue_count_6_months"
]

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
import streamlit as st


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    #modify = st.checkbox("Add filters")

    #if not modify:
    #    return df

    # Use session_state to track filter selections
    #if 'selected_columns' not in st.session_state:
    #    st.session_state.selected_columns = []

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:

        # Use session state to store selected columns
        #selected_columns = st.multiselect(
        #    "Filter dataframe on", df.columns, default=st.session_state.selected_columns
        #)
        #st.session_state.selected_columns = selected_columns  # Store in session state

        columns_to_exclude = ['Donation Link']

        # Create a list of columns to include in the multiselect
        available_columns = [col for col in df.columns if col not in columns_to_exclude]

        selected_columns = st.multiselect("Filter dataframe on", available_columns)
        
        for column in selected_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = round(df[column].min())
                _max = round(df[column].max())
                step = round((_max - _min) / 100) if (_max - _min) >= 100 else 1
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

# Title of the app
col2.title("GrantsScope for GG22") 
col2.markdown("### Powered by Open Source Observer & RegenData") 
col2.caption("Created by Rohit Malekar | [grantsscope.xyz](grantsscope.xyz)")

col2.markdown(""" Since June 2023, GrantsScope has been pioneering data-driven innovations to enhance Gitcoin Grants. We've launched personalized, AI-powered tools like RAG (Retrieval Augmented Generation) experiences and one-click grantee recommendations, making it easier for donors to explore impactful projects based on donation history and advanced clustering algorithms. """)

col2.markdown(""" In GG22, we're taking it a step further. Donors can now create their own composite impact metrics, powered by Open Source Observer, and discover how grantees align with their goals. If you find value in these insights, support GrantsScope and contribute to more innovations for the community. """)

col2.link_button("Support GrantsScope in GG22", "https://explorer.gitcoin.co/#/round/42161/608/77", type="primary")
scol1, scol2, scol3 = st.columns([1, 1.5, 1])  # Adjust the width ratios as needed
scol2.markdown("**STEP 1:** Pick your metrics")
selected_metrics = scol2.multiselect("Select up to 3 metrics:", metrics, max_selections=3)

# Total weight variable
total_weight = 100
assigned_weight = 0

# Initialize weights dictionary
weights = {metric: 0 for metric in selected_metrics}

# Input for weights using sliders
if selected_metrics:
    scol2.markdown("**STEP 2:** Assign weights (total must equal 100):")
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
        # if scol2.button("Apply"):
            
        # Read the CSV files from the data folder
        oso_df = pd.read_csv('./data/GG22 GrantsScope OSO - oso.csv')
        regendata_df = pd.read_csv('./data/GG22 GrantsScope OSO - regendata.csv')

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

        # Normalize the contributor count
        merged_df['normalized_contributor_count_6_months'] = min_max_normalize(merged_df['contributor_count_6_months'])

        # Calculate composite score per contributor
        merged_df['composite_score_per_contributor'] = merged_df['composite_score'] / merged_df['normalized_contributor_count_6_months'].replace(0, 1)  # Replace 0 with 1 to avoid division by zero

        # Normalize composite_score_per_contributor to a scale of 0 to 100
        max_score_per_contributor = merged_df['composite_score_per_contributor'].max()
        if max_score_per_contributor > 0:  # Avoid division by zero
            merged_df['composite_score_per_contributor'] = round((merged_df['composite_score_per_contributor'] / max_score_per_contributor) * 100)
        else:
            merged_df['composite_score_per_contributor'] = 0  # Set to 0 if max is 0
        
        # Create a new DataFrame with the specified columns and rename them
        display_df = merged_df[[
            'project_title',
            'round_name',
            'explorer_link',
            'composite_score',
            'composite_score_per_contributor',
            'first_commit_date',
            'star_count',
            'fork_count',
            'developer_count',
            'contributor_count',
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
            'composite_score_per_contributor' : 'Impact Efficiency Per Contributor',
            'first_commit_date': 'First Commit Date',
            'star_count': 'Star Count',
            'fork_count': 'Fork Count',
            'developer_count':'Developer Count',
            'contributor_count':'Contributor Count',
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

        
        filtered_df = filter_dataframe(display_df)

        st.dataframe(
            filtered_df,
            column_config={
                "Donation Link": st.column_config.LinkColumn(
                    "Donation Link",
                    help="Click to donate",
                    display_text="Donate Here"
                ),
                "Composite Score (0 to 100)": st.column_config.ProgressColumn(
                    "Composite Score (0 to 100)",
                    help="Composite score representing the overall impact and performance",
                    format="%f",
                    min_value=0,
                    max_value=100
                ),
                "Impact Efficiency Per Contributor": st.column_config.ProgressColumn(
                    "Impact Efficiency Per Contributor",
                    help="The efficiency of impact per contributor, scaled from 0 to 100",
                    format="%f",
                    min_value=0,
                    max_value=100
                ),
            },
            hide_index=True
        )

        with st.expander("Understanding the Composite Score and Impact Efficiency Per Contributor"):
            st.markdown('''
                **Composite Score:**
                The composite score is a key metric that helps us evaluate and compare different projects based on various performance indicators. Here’s a simple breakdown of how we calculate this score:
                1. **Normalization of Metrics:** We start by normalizing each selected metric. Normalization is a process that adjusts the values of these metrics to a common scale, typically between 0 and 1. This ensures that no single metric disproportionately influences the overall score due to its scale.
                2. **Weight Assignment**: Each metric is assigned a weight by the user based on its importance. You can think of weights as the significance of each metric in the overall evaluation. 
                3. **Weighted Sum Calculation:** The composite score is calculated by taking the weighted sum of the normalized metrics. This means we multiply each normalized metric by its corresponding weight and then sum these values together. The multiplication by 100 scales the score to a percentage format, making it easier to interpret. The formula looks like this:
                > Composite Score=(∑(Normalized Metric×Weight))×100
                4. **Rounding:** Finally, we round the composite score to the nearest whole number for simplicity. This gives us a final score that is easy to read and understand.
                        
                **Impact Efficiency Per Contributor:**

                This metric offers a detailed look at how efficiently each contributor drives impact in a project. Instead of simply looking at the overall size of a project’s team, this metric normalizes the project’s contributions by the number of contributors. This allows us to highlight how effective smaller or larger teams are at creating value.

                1. We first calculate a composite score, which takes into account key performance indicators such as commits, pull requests, and issues resolved.
                2. Then, we divide this composite score by the number of contributors in the last 6 months, adjusted so that projects with very few contributors aren’t disproportionately penalized.
                3. The result is a normalized score, scaled between 0 and 100, showing how much impact each contributor is responsible for, on average.
                
                **Why does this matter?**

                - **Fair Comparison:** By focusing on per-contributor efficiency, we ensure that smaller teams with fewer contributors aren't overshadowed by larger teams simply due to their size.
                - **Productivity:** This metric emphasizes contributor productivity, showcasing teams where each individual is making significant contributions.
                In essence, Impact Efficiency Per Contributor helps donors and users identify which projects are maximizing their contributors' potential, allowing for a more meaningful comparison of teams regardless of their size.
                ''')

        st.error("Note that the composite scores are relative within this group of projects, not absolute measures. A low score doesn't necessarily mean a project is inactive; it might be in a stable phase or have a different development model. \
                A higher Impact Efficiency Per Contributor indicates that a project is achieving more activity with fewer contributors. However, it's important to consider this alongside other factors such as project complexity, stage of development, and specific project goals, as high activity per contributor isn't always indicative of overall project health or success.")
        

            
            

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
