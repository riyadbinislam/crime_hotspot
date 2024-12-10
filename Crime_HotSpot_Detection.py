import io
import numpy as np
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster
from streamlit_folium import st_folium, folium_static
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


st.markdown(
    """
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    data = pd.read_csv('scraped_arrests_with_zip_codes.csv')
    return data

def check_page_status(url):
    try:
        response = requests.get(url)
        return response.status_code
    except requests.exceptions.RequestException as e:
        return None

st.title("Crime Hotspot Detection - Winston-Salem, NC")

tab1, tab2, tab3, tab4 = st.tabs(["Data Description", "Exploratory Data Analysis", "Machine Learning", "Performance Metrics"])

with tab1:


    st.header("Data Description")
    data = load_data()
    st.write("Data Information:")
    buf = io.StringIO()
    data.info(buf=buf)
    info_str = buf.getvalue()

    st.text(info_str)

    st.write("Dataset Overview:")
    st.write(data.head(10))
    st.write(data.tail(10))

    st.write("Data Shape:")
    st.write(data.shape)

    st.write("Check for Duplication")
    st.write(data.nunique())

    st.write("Missing Values:")
    st.write(data.isnull().sum())

with tab2:
    st.header("Exploratory Data Analysis (EDA)")

    #################################################################
    # description
    st.subheader("Summary Statistics")
    st.write(data.describe())

    ##################################################################


    ##################################################################
    # data extraction and modification

    # Convert the 'Date' column into datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Extract 'YearMonth' for aggregation
    data['YearMonth'] = data['Date'].dt.to_period('M')

    # Count of arrests per month
    monthly_counts = data['YearMonth'].value_counts().sort_index()

    # Extract hour information from 'Time' column
    data['Hour'] = pd.to_datetime(data['Time'], errors='coerce').dt.hour

    data['DayOfWeek'] = data['Date'].dt.day_name()

    # Count of arrests by day of the week (reorder days)
    arrests_by_day = data['DayOfWeek'].value_counts().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )

    # Process data
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    data['DayType'] = data['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

    weekday_data = data[data['DayType'] == 'Weekday']
    weekend_data = data[data['DayType'] == 'Weekend']

    top_weekday_charges = weekday_data['Charge'].value_counts().head(15)
    top_weekend_charges = weekend_data['Charge'].value_counts().head(15)

    repeat_offenders = data['Arrestee'].value_counts()
    repeat_offenders = repeat_offenders[repeat_offenders > 1]  # Filter for repeat offenders
    top_repeat_offenders = repeat_offenders.head(15)

    # Prepare data for trends of top 10 crime types
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    top_charges = data['Charge'].value_counts().head(10).index
    filtered_data = data[data['Charge'].isin(top_charges)]
    crime_trends = filtered_data.groupby([filtered_data['Date'].dt.to_period('M'), 'Charge']).size().unstack().fillna(0)


    crime_mapping = {
        'Theft': ['b&e', 'larceny', 'stole', 'stolen', 'theft', 'robbery', 'breaking', 'entering', 'embezzlement', 'larc', 'unauthorized use', 'break', 'tampering', 'obtain', 'maintain', 'maintn', 'shoplifting', 'burglary', 'retur', 'return', 'ret', 'concealment', 'unlawful poss', 'unlawful use', 'extortion', 'steal', 'alter', 'tamper'],
        'Violent': ['asslt', 'threat', 'threats', 'affray', 'assault', 'stalking', 'injury', 'kidnapping', 'murder', 'child abuse', 'cyberstalking', 'aslt', 'inju', 'harassing', 'manslaughter', 'endanger', 'imprisonment', 'knife', 'abuse', 'threatening', 'battery', 'affra', 'violence', 'aslt', 'adw', 'death', 'injure', 'cruelty', 'kill', 'intent', 'awdwikisi', 'malicious', 'awol', 'armed', 'peeping', 'burn', 'burning', 'arson', 'intimidate', 'profane', 'rioting'],
        'Drug': ['methamhetamine', 'drugs', 'sched', 'control', 'substance', 'cocaine', 'drug', 'paraphernalia', 'marijuana', 'mdse', 'sch', 'marij', 'heroin', 'methamphetamine', 'methaq', 'marijuan', 'cs', 'cocain'],
        'Traffic': ['driving', 'dwi', 'licensee', 'regis', 'plate', 'suspended', 'revoked', 'dwlr', 'speeding', 'license', 'permit', 'drive', 'ndl', 'speed', 'hit', 'veh no ins', 'traff', 'drvg', 'registration', 'obstructing', 'imp', 'heed', 'following', 'drvt', 'signals', 'dwlf', 'stop', 'dui'],
        'Fraud': ['forgery', 'uttering', 'fraud', 'defraud', 'security', 'impersonate', 'misuse', 'check', 'alt', 'fict', 'illegal', 'hiring', 'solicit', 'fraudulent', 'notary'],
        'Weapons': ['firearms', 'handgun', 'firearm', 'weapon', 'weap', 'weapon', 'gun', 'disch', 'ccw', 'carry'],
        'Sex': ['indecent', 'sexual', 'incest', 'rape', 'prostitution', 'sex', 'masturbation', 'porn'],
        'Other': ['abc', 'del', 'alc', 'consume', 'order', 'fail', 'appear', 'compl', 'trespass', 'fcso', 'hold', 'drunk', 'disruptive', 'alcohol', 'vand', 'beverage', 'conspiracy', 'disorderly', 'noise', 'probation', 'summons', 'trespassing', 'beg', 'alms', 'resisting', 'liquor', 'delinq', 'urinate', 'defecate', 'interference', 'wine', 'beer', 'panhandling', 'brdcast', 'fugitive', 'bonfire', 'rubbish', 'insurance', 'littering', 'dwelling', 'restraint', 'sleep', 'gambling', 'accessory', 'barking', 'flee', 'prop', 'tresp', 'arson', 'law', 'liability', 'wn', 'lq', 'consume', 'produce', 'false', 'information', 'begging', 'regulate', 'rdo', 'strike', 'restr', 'disperse', 'aof', 'park', 'vio', 'misuse', 'crime', 'passing', 'stopped', 'escape', 'undisciplined', 'nol', 'felonious', 'fine', 'flee', 'loitering', 'prohibitions', 'prohibition', 'beverage', 'gamb', 'dispose', 'prob', 'deliver', 'commercial', 'interfere', 'conversion', 'cause', 'violation', 'guard', 'interfering', 'litter', 'violence', 'res', 'unsafe', 'ban', 'disturbance', 'hoax', 'tag', 'abandonment']
    }


    def categorize_crime(charge):
        if not isinstance(charge, str):
            charge = str(charge)  # Convert to string if it's not already
        for category, keywords in crime_mapping.items():
            if any(keyword.upper() in charge.upper() for keyword in keywords):
                return category
        return 'Uncategorized'


    data['CrimeType'] = data['Charge'].apply(categorize_crime)

    crime_type_distribution = data['CrimeType'].value_counts()

    st.subheader("Crime Type Distribution")

    crime_type_distribution_df = crime_type_distribution.reset_index()
    crime_type_distribution_df.columns = ['Crime Type', 'Count']

    st.dataframe(crime_type_distribution_df)

    # Define helper functions for season and time slot
    def get_season(date):
        if pd.isnull(date):
            return 'Unknown'
        if date.month in [12, 1, 2]:
            return 'Winter'
        elif date.month in [3, 4, 5]:
            return 'Spring'
        elif date.month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'


    def get_time_slot(hour):
        if pd.isnull(hour):
            return 'Unknown'
        if 0 <= hour < 6:
            return 'Night'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'


    # Ensure 'Date' and 'Time' columns are processed correctly
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Time'] = data['Time'].astype(str)
    data['DateTime'] = pd.to_datetime(data['Date'].dt.strftime('%Y-%m-%d') + ' ' + data['Time'], errors='coerce')

    # Add 'Season' and 'TimeSlot' columns
    data['Season'] = data['DateTime'].apply(get_season)
    data['TimeSlot'] = data['DateTime'].dt.hour.apply(get_time_slot)

    # Group by 'Season' and 'TimeSlot'
    seasonal_trends = data.groupby(['Season', 'TimeSlot']).size().reset_index(name='Count')

    # Display the trends
    st.subheader("Seasonal Crime Trends by Time Slot")
    st.dataframe(seasonal_trends)

    seasonal_pivot = seasonal_trends.pivot(index='Season', columns='TimeSlot', values='Count').fillna(0)
    seasons = seasonal_pivot.index
    time_slots = seasonal_pivot.columns
    x = np.arange(len(seasons))  # X-axis positions for seasons
    width = 0.2  # Width of each bar

    zip_code_analysis = data.groupby('ZIPCode').size().reset_index(name='Count')

    zip_code_analysis = zip_code_analysis.sort_values(by='Count', ascending=False)

    st.subheader("Crime Analysis by ZIP Code")
    st.dataframe(zip_code_analysis)

    categorized_seasonal_trends = data.groupby(['CrimeType', 'Season']).size().reset_index(name='Count')

    pivot_categorized_trends = categorized_seasonal_trends.pivot(index='CrimeType', columns='Season',
                                                                 values='Count').fillna(0)

    categorized_zipcode_trends = data.groupby(['CrimeType', 'ZIPCode']).size().reset_index(name='Count')

    pivot_categorized_zipcode = categorized_zipcode_trends.pivot(index='CrimeType', columns='ZIPCode',
                                                                 values='Count').fillna(0)

    st.subheader("Categorized Crime Trends by Zip Code")
    st.dataframe(pivot_categorized_zipcode)

    required_columns = ['Season', 'Latitude', 'Longitude']
    if not all(col in data.columns for col in required_columns):
        st.error(f"The dataset must contain the following columns: {', '.join(required_columns)}")
    else:
        data = data.dropna(subset=['Latitude', 'Longitude'])

        data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
        data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

        data = data.dropna(subset=['Latitude', 'Longitude'])

        seasons = data['Season'].dropna().unique()

    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

    data['CrimeType'] = data['Charge'].apply(categorize_crime)
    crime_types = data['CrimeType'].unique()

    ##################################################################

    ##################################################################

    st.subheader("Plotting Window")

    # data plotting section
    plot_options = [
        "",
        "Monthly Arrest Trends",
        "Top 15 Most Common Charges",
        "Top 15 Most Common Charges",
        "Top 15 Most Common Arrest Locations",
        "Arrests by Time of Day",
        "Arrests by Day of the Week",
        "Top 15 Crimes on Weekdays vs Weekends",
        "Top 15 Repeat Offenders",
        "Trends of Top 10 Crime Types Over Time",
        "Seasonal Crime Trends by Time Slot",
        "Crime Analysis by ZIP Code",
        "Crime Heatmap",
        "Categorized Crime Trends by Season",
        "Categorized Crime Trends by ZIP Code",
        "Crime Heatmap with Time Slider",
        "Seasonal Crime Density Maps",
        "Crime Maps for Each Crime Type"
    ]
    selected_plot = st.selectbox("Select a plot to display:", plot_options)

    if selected_plot == "Monthly Arrest Trends":
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_counts.plot(kind='bar', ax=ax, color='green', alpha=0.7)
        ax.set_title('Monthly Arrest Counts')
        ax.set_xlabel('Year-Month')
        ax.set_ylabel('Number of Arrests')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif selected_plot == "Top 15 Most Common Charges":
        fig, ax = plt.subplots(figsize=(10, 6))
        data['Charge'].value_counts().head(15).plot(kind='bar', ax=ax, color='coral', alpha=0.7)
        ax.set_title('Top 15 Most Common Charges')
        ax.set_xlabel('Charge')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        st.pyplot(fig)


    elif selected_plot == "Top 15 Most Common Arrest Locations":
        fig, ax = plt.subplots(figsize=(10, 6))
        data['Location'].value_counts().head(15).plot(kind='bar', ax=ax, color='lightgreen', alpha=0.7)
        ax.set_title('Top 15 Most Common Arrest Locations')
        ax.set_xlabel('Location')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif selected_plot == "Arrests by Time of Day":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data['Hour'], bins=24, range=(0, 24), edgecolor='black', color='purple', alpha=0.7)
        ax.set_title('Arrests by Time of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Arrests')
        ax.set_xticks(range(0, 24))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    elif selected_plot == "Arrests by Day of the Week":
        fig, ax = plt.subplots(figsize=(10, 6))
        arrests_by_day.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title('Arrests by Day of the Week')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Number of Arrests')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif selected_plot == "Top 15 Crimes on Weekdays vs Weekends":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Weekday crimes
        top_weekday_charges.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Top 15 Crimes on Weekdays')
        ax1.set_xlabel('Charge')
        ax1.set_ylabel('Number of Arrests')
        ax1.set_xticklabels(top_weekday_charges.index, rotation=90)

        # Weekend crimes
        top_weekend_charges.plot(kind='bar', ax=ax2, color='salmon', edgecolor='black')
        ax2.set_title('Top 15 Crimes on Weekends')
        ax2.set_xlabel('Charge')
        ax2.set_xticklabels(top_weekend_charges.index, rotation=90)

        plt.tight_layout()
        st.pyplot(fig)

    elif selected_plot == "Top 15 Repeat Offenders":
        fig, ax = plt.subplots(figsize=(10, 6))
        top_repeat_offenders.plot(kind='bar', ax=ax, color='purple', edgecolor='black', alpha=0.8)
        ax.set_title('Top 15 Repeat Offenders')
        ax.set_xlabel('Arrestee')
        ax.set_ylabel('Number of Arrests')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif selected_plot == "Trends of Top 10 Crime Types Over Time":
        fig, ax = plt.subplots(figsize=(12, 6))
        crime_trends.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Trends of Top 10 Crime Types Over Time')
        ax.set_xlabel('Date (Month)')
        ax.set_ylabel('Number of Arrests')
        ax.legend(title='Crime Type')
        ax.grid(axis='both', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    elif selected_plot == "Seasonal Crime Trends by Time Slot":
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each time slot as a separate bar group
        for i, slot in enumerate(time_slots):
            ax.bar(x + i * width, seasonal_pivot[slot], width, label=slot)

        # Add labels and title
        ax.set_xlabel('Season')
        ax.set_ylabel('Number of Crimes')
        ax.set_title('Seasonal Crime Trends by Time Slot')
        ax.set_xticks(x + width * (len(time_slots) - 1) / 2)
        ax.set_xticklabels(seasons)
        ax.legend(title='Time Slot')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

    elif selected_plot == "Crime Analysis by ZIP Code":
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(zip_code_analysis['ZIPCode'], zip_code_analysis['Count'], color='blue', alpha=0.7)

        ax.set_title('Crime Analysis by ZIP Code', fontsize=16)
        ax.set_xlabel('ZIP Code', fontsize=14)
        ax.set_ylabel('Number of Crimes', fontsize=14)
        ax.set_xticks(range(len(zip_code_analysis['ZIPCode'])))
        ax.set_xticklabels(zip_code_analysis['ZIPCode'], rotation=45, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

    elif selected_plot == "Crime Heatmap":
        center_lat = data['Latitude'].mean()
        center_lon = data['Longitude'].mean()
        heatmap_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        heat_data = data[['Latitude', 'Longitude']].dropna()

        HeatMap(heat_data.values, radius=10).add_to(heatmap_map)

        st_folium(heatmap_map, width=800, height=600)

    elif selected_plot == "Categorized Crime Trends by Season":
        fig, ax = plt.subplots(figsize=(14, 8))
        pivot_categorized_trends.plot(kind='bar', stacked=True, ax=ax)

        ax.set_title('Categorized Crime Trends by Season', fontsize=16)
        ax.set_xlabel('Crime Type', fontsize=14)
        ax.set_ylabel('Number of Crimes', fontsize=14)
        ax.legend(title='Season', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

    elif selected_plot == "Categorized Crime Trends by ZIP Code":
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            pivot_categorized_zipcode,
            cmap="YlGnBu",
            linewidths=0.5,
            annot=True,
            fmt=".0f",
            ax=ax
        )

        ax.set_title('Categorized Crime Trends by ZIP Code', fontsize=16)
        ax.set_xlabel('ZIP Code', fontsize=14)
        ax.set_ylabel('Crime Type', fontsize=14)

        st.pyplot(fig)

    elif selected_plot == "Crime Heatmap with Time Slider":
        if 'DateTime' not in data.columns:
            st.error("The dataset does not contain a 'DateTime' column. Please ensure it's correctly created.")
        elif 'Latitude' not in data.columns or 'Longitude' not in data.columns:
            st.error("The dataset does not contain 'Latitude' and 'Longitude' columns required for mapping.")
        else:
            data['YearMonth'] = data['DateTime'].dt.to_period('M').astype(str)

            time_heat_data = []
            unique_months = data['YearMonth'].sort_values().unique()

            for month in unique_months:
                monthly_data = data[data['YearMonth'] == month][['Latitude', 'Longitude']].dropna()

                time_heat_data.append(monthly_data.values.tolist())

            if not any(len(heat_data) > 0 for heat_data in time_heat_data):
                st.warning("No valid data points found for the heatmap. Please check the dataset.")
            else:
                time_slider_map = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()],
                                             zoom_start=12)

                HeatMapWithTime(data=time_heat_data, index=unique_months.tolist(), radius=10).add_to(time_slider_map)

                folium_static(time_slider_map)

    elif selected_plot == "Seasonal Crime Density Maps":
        for season in seasons:
            season_data = data[data['Season'] == season]

            season_map = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=12)

            HeatMap(season_data[['Latitude', 'Longitude']].values, radius=12).add_to(season_map)

            st.markdown(f"### {season} Crime Density Map")
            folium_static(season_map)

    elif selected_plot == "Crime Maps for Each Crime Type":
        for crime_type in crime_types:
            st.subheader(f"Map for Crime Type: {crime_type}")

            filtered_data = data[data['CrimeType'] == crime_type]

            st.write(f"Total crimes of type '{crime_type}': {len(filtered_data)}")

            crime_map = folium.Map(location=[filtered_data['Latitude'].mean(), filtered_data['Longitude'].mean()],
                                   zoom_start=12)

            marker_cluster = MarkerCluster().add_to(crime_map)

            for _, row in filtered_data.iterrows():
                folium.Marker(
                    location=(row['Latitude'], row['Longitude']),
                    popup=f"<b>Crime Type:</b> {row['CrimeType']}<br><b>Charge:</b> {row['Charge']}",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(marker_cluster)

            folium_static(crime_map)

    else:
        pass

with tab3:
    st.header("Crime Hotspot Detection and Clustering")

    with st.form(key="clustering_form"):
        clustering_algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ["DBSCAN", "HDBSCAN", "Hierarchical Clustering"]
        )

        features = st.multiselect(
            "Select Features for Clustering",
            ["Latitude", "Longitude", "ZIPCode", "CrimeType"],
            default=["Latitude", "Longitude"]
        )

        if clustering_algorithm == "DBSCAN":
            eps = st.slider("DBSCAN Epsilon (eps)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
            min_samples = st.slider("DBSCAN Minimum Samples", min_value=1, max_value=20, step=1, value=5)

        elif clustering_algorithm == "HDBSCAN":
            min_cluster_size = st.slider("HDBSCAN Minimum Cluster Size", min_value=2, max_value=50, step=1, value=5)

        elif clustering_algorithm == "Hierarchical Clustering":
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, step=1, value=5)

        submit_button = st.form_submit_button(label="Run Clustering")

    if submit_button:
        if len(features) < 2:
            st.warning("Please select at least two features for clustering.")
        else:
            # Filter and scale the data
            clustering_data = data[features].dropna()
            scaler = StandardScaler()
            clustering_data_scaled = scaler.fit_transform(clustering_data)

            # Apply the selected clustering algorithm
            if clustering_algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif clustering_algorithm == "HDBSCAN":
                model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            elif clustering_algorithm == "Hierarchical Clustering":
                from sklearn.cluster import AgglomerativeClustering
                model = AgglomerativeClustering(n_clusters=n_clusters)

            # Fit the model and assign cluster labels
            clustering_data["Cluster"] = model.fit_predict(clustering_data_scaled)

            # Store results in session state
            st.session_state["clustering_data"] = clustering_data
            st.session_state["clustering_data_scaled"] = clustering_data_scaled

            st.write(f"Number of Clusters Detected: {clustering_data['Cluster'].nunique()}")
            st.dataframe(clustering_data)

            # Add the cluster labels back to the original data
            data["Cluster"] = None
            data.loc[clustering_data.index, "Cluster"] = clustering_data["Cluster"]

            # Generate the Crime Hotspot Map
            st.subheader("Crime Hotspot Map")
            map_center = [data["Latitude"].mean(), data["Longitude"].mean()]
            folium_map = folium.Map(location=map_center, zoom_start=12)

            cluster_colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightred",
                              "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple", "white"]
            for _, row in data.dropna(subset=["Latitude", "Longitude"]).iterrows():
                cluster_id = row["Cluster"]
                if cluster_id != -1:
                    color = cluster_colors[cluster_id % len(cluster_colors)]
                    folium.CircleMarker(
                        location=(row["Latitude"], row["Longitude"]),
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        tooltip=f"CrimeType: {row['CrimeType']}<br>Cluster: {cluster_id}",
                    ).add_to(folium_map)

            st_folium(folium_map, width=800, height=600)





with tab4:
    st.header("Performance Metrics")

    if "clustering_data" in st.session_state and "clustering_data_scaled" in st.session_state:
        clustering_data = st.session_state["clustering_data"]
        clustering_data_scaled = st.session_state["clustering_data_scaled"]

        if "Cluster" in clustering_data.columns and clustering_data["Cluster"].nunique() > 1:
            db_score = davies_bouldin_score(clustering_data_scaled, clustering_data["Cluster"])
            st.write(f"Davies-Bouldin Index: {db_score:.2f}")
        else:
            st.warning("Not enough clusters for Davies-Bouldin Index.")

        if "Cluster" in clustering_data.columns and clustering_data["Cluster"].nunique() > 1:
            ch_score = calinski_harabasz_score(clustering_data_scaled, clustering_data["Cluster"])
            st.write(f"Calinski-Harabasz Index: {ch_score:.2f}")
        else:
            st.warning("Not enough clusters for Calinski-Harabasz Index.")

        silhouette = silhouette_score(clustering_data_scaled, clustering_data["Cluster"])
        st.write(f"Silhouette Score: {silhouette:.2f}")

        st.subheader("Cluster Summary")
        cluster_summary = clustering_data.groupby("Cluster").mean()
        st.dataframe(cluster_summary)

        if "TrueLabels" in data.columns:
            ari_score = adjusted_rand_score(data["TrueLabels"], clustering_data["Cluster"])
            st.write(f"Adjusted Rand Index (ARI): {ari_score:.2f}")
        else:
            st.warning("Ground truth labels are not available for ARI computation.")

    else:
        st.warning("Run clustering first to compute performance metrics.")

