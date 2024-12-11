"""
PROMPT USED FOR DIRECT CLASSIFICATION: 
'What is the primary genre of this song? 
You can only choose electronic, rock, funk / soul, hip hop, jazz, blues, pop, latin, or classical.'
"""
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Define possible genres and their associated terms/synonyms/acronyms
GENRE_MAPPING = {
    "electronic": [
        "electronic", "edm", "drum and bass", "drum n bass", "drum & bass", 
        "disco", "techno", "house", "trance", "dubstep", "ambient", 
        "industrial", "synthwave", "electro", "electronic dance music", "dance"
    ],
    "rock": [
        "rock", "metal", "punk", "grunge", "hard rock", "alternative rock", 
        "progressive rock", "classic rock", "indie rock", "garage rock", "punk rock"
    ],
    "funk / soul": [
        "funk", "soul", "funk/soul", "funk / soul", "r&b", 
        "motown", "neo-soul", "groove", "funky"
    ],
    "hip hop": [
        "hip hop", "hip-hop", "hiphop", "rap", "trap", "boom bap", 
        "freestyle", "gangsta rap", "conscious rap"
    ],
    "jazz": [
        "jazz", "smooth jazz", "bebop", "big band", "swing", "fusion", 
        "free jazz", "latin jazz", "cool jazz", "traditional jazz"
    ],
    "blues": [
        "blues", "delta blues", "chicago blues", "electric blues", 
        "acoustic blues", "country blues", "modern blues"
    ],
    "pop": [
        "pop", "synthpop", "indie pop", "electropop", "k-pop", "j-pop", 
        "teen pop", "pop rock", "pop punk", "dance pop"
    ],
    "latin": [
        "latin", "bossanova", "bossa nova", "salsa", "reggaeton", 
        "bachata", "merengue", "cumbia", "latin pop", "tango", 
        "mariachi", "flamenco", "ranchera"
    ],
    "classical": [
        "classical", "symphony", "opera", "baroque", "romantic", 
        "orchestral", "chamber music", "concerto", "sonata", 
        "classical crossover", "early music"
    ]
}

def classify_genre(caption):
    """
    Classifies the genre based on the caption.
    Returns the genre name or 'unknown' if no match is found.
    """
    caption_lower = caption.lower()

    for genre, keywords in GENRE_MAPPING.items():
        for keyword in keywords:
           if keyword in caption_lower:
                return genre
    return "unknown"

def evaluate_classification(captions_csv, groundtruth_csv):
    """
    Evaluates the classification by comparing extracted genres with ground truth.
    Args:
        captions_csv: Path to the CSV file containing captions (with 'audio_path' and 'caption' columns).
        groundtruth_csv: Path to the CSV file containing ground truth genres (with 'id' and 'genre' columns).

    Returns:
        None
    """
    # Read captions and ground truth CSVs
    captions_df = pd.read_csv(captions_csv)
    groundtruth_df = pd.read_csv(groundtruth_csv)

    # Ensure the ground truth and captions align on IDs
    captions_df["audio_path"] = captions_df["audio_path"].str.replace('.npy$', '', regex=True)
    captions_df = captions_df.rename(columns={"audio_path": "id"})
    merged_df = pd.merge(captions_df, groundtruth_df, on="id", how="inner")

    if merged_df.empty:
        raise ValueError("No matching IDs found between captions and ground truth.")

    # Classify genres based on captions
    merged_df['predicted_genre'] = merged_df['caption'].apply(classify_genre)

    # Extract true and predicted genres
    true_genres = merged_df['genre']
    predicted_genres = merged_df['predicted_genre']

    # Classification report
    report = classification_report(true_genres, predicted_genres, zero_division=0)
    acc = accuracy_score(true_genres, predicted_genres)

    output_path = "classification_report_" + captions_csv.rpartition('/')[-1]
    with open(output_path, "w+") as f:
        f.write("Classification Report\n")
        f.write(f"Accuracy: {acc*100:0.02f}%")
        f.write("=====================\n")
        f.write(report)

    # Confusion matrix
    labels = list(GENRE_MAPPING.keys()) + ["unknown"]
    cm = confusion_matrix(true_genres, predicted_genres, labels=labels)

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.title("Confusion Matrix")
    # plt.show()
    plt.savefig(f"confmat_{captions_csv.rpartition('/')[-1].replace('.txt','')}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Path to captions CSV (with 'audio_path' and 'caption' columns)
    captions_csv = "results/classification_from_audio_qa/mi_all_100_classification.txt"

    # Path to ground truth CSV (with 'id' and 'genre' columns)
    groundtruth_csv = "datasets/annotations/test.csv"

    # Evaluate classification
    evaluate_classification(captions_csv, groundtruth_csv)