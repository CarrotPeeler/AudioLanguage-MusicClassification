import pandas as pd
from sklearn.model_selection import train_test_split

TRACKS_WITH_LYRICS_CSV_PATH = "datasets/annotations/tracks_with_lyrics.csv"
TEST_SIZE = 1000
NUM_CLASSES = 9 # reggae missing

# Load the CSV file into a DataFrame
df = pd.read_csv(TRACKS_WITH_LYRICS_CSV_PATH)

subset_df = pd.read_csv("datasets/annotations/subset.csv")  # This is the subset that must be in the test split

# Ensure that the subset is included in the test data
# First, find rows in the main dataset that are also in the subset
test_data_from_subset = df[df['id'].isin(subset_df['id'])]

# Now, we need to split the remaining data, excluding the subset
remaining_data = df[~df['id'].isin(subset_df['id'])]

# Perform a stratified split on the remaining data
# We'll calculate the size of the remaining test data by subtracting the size of the subset
remaining_test_size = TEST_SIZE*3 - len(test_data_from_subset)

# Ensure we don't get a negative size by checking if the remaining data size is greater than the required test size
if remaining_test_size > 0:
    train_data, additional_test_data = train_test_split(
        remaining_data, 
        test_size=remaining_test_size, 
        stratify=remaining_data['genre'], 
        random_state=42
    )
else:
    # If there are not enough rows in the remaining data, just use the whole remaining data for test
    train_data = remaining_data
    additional_test_data = pd.DataFrame()  # No additional test data

# # Combine the test data (subset and remaining)
# final_test_data = pd.concat([test_data_from_subset, additional_test_data])

# Calculate the number of samples per genre in the final test data
genre_counts = test_data_from_subset['genre'].value_counts()

# Find the target number of samples per genre
target_count = TEST_SIZE // NUM_CLASSES

# Create a balanced final test data by sampling the minimum count of each genre
balanced_test_data = pd.DataFrame()

for genre in genre_counts.index:
    current_subset = test_data_from_subset[test_data_from_subset['genre'] == genre]
    to_sample = additional_test_data[additional_test_data['genre'] == genre]
    if target_count - genre_counts[genre] > 0:
        balanced_genre_data = to_sample.sample(n=target_count - genre_counts[genre], random_state=42)
        balanced_test_data = pd.concat([balanced_test_data, current_subset, balanced_genre_data])
    else:
        balanced_test_data = pd.concat([balanced_test_data, current_subset])
    

# Save the final train and test datasets to CSV
train_data.to_csv('train.csv', index=False)
balanced_test_data.to_csv('test.csv', index=False)

added = balanced_test_data[~balanced_test_data['id'].isin(subset_df['id'])]
added.to_csv('subset2.csv', index=False)

# Optionally, print a preview of the datasets to verify
print("Training data:")
print(len(train_data))
print("\nFinal Test data:")
print(len(balanced_test_data))
print("\nRemaining for training data:")
print(len(added))

print(balanced_test_data['genre'].value_counts())
