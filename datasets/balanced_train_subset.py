import pandas as pd
from sklearn.model_selection import train_test_split

def generate_stratified_subset(csv_path, subset_size, num_classes, output_path):
    """
    Generates a stratified and balanced subset of the given size from a CSV file.

    Parameters:
        csv_path (str): Path to the input CSV file.
        subset_size (int): Desired size of the subset.
        output_path (str): Path to save the generated subset as a CSV file.

    Returns:
        pd.DataFrame: The generated subset DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Ensure the dataset has the required columns
    if 'genre' not in df.columns or 'id' not in df.columns:
        raise ValueError("The CSV must have 'genre' and 'id' columns.")

    # Check if subset size is greater than the total rows
    if subset_size > len(df):
        raise ValueError("Subset size cannot exceed the total number of rows in the dataset.")

    # Perform stratified sampling
    subset_df, _ = train_test_split(
        df,
        test_size=(len(df) - subset_size),
        stratify=df['genre'],
        random_state=42
    )
    target_count = subset_size // num_classes
    genre_counts = df['genre'].value_counts()

    balanced_test_data = pd.DataFrame()
    for genre in genre_counts.index:
        to_sample = df[df['genre'] == genre]
        if target_count - genre_counts[genre] > 0:
            balanced_genre_data = to_sample.sample(n=target_count - genre_counts[genre], random_state=42)
            balanced_test_data = pd.concat([balanced_test_data, balanced_genre_data])
        elif target_count - genre_counts[genre] < 0:
            balanced_genre_data = to_sample.sample(n=genre_counts[genre] - target_count, random_state=42)
            balanced_test_data = pd.concat([balanced_test_data, to_sample[~to_sample['id'].isin(balanced_genre_data['id'])]])

    subset_df.to_csv(output_path, index=False)
    print(f"Stratified subset saved to {output_path}")
    return balanced_test_data

csv_path = "datasets/annotations/train.csv"  
subset_size = 3000
num_classes = 9
output_path = "train_subset.csv"  

# Generate and save the subset
subset = generate_stratified_subset(csv_path, subset_size, num_classes, output_path)
print(subset['genre'].value_counts())
print(len(subset))
