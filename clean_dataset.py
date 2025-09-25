import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def normalize_text(text):
    """Normalize text by removing extra spaces, converting to lowercase, etc."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove punctuation except for basic sentence structure
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def remove_duplicates(df, similarity_threshold=0.85):
    """Remove duplicate entries using TF-IDF and cosine similarity."""
    # Normalize all texts
    df['normalized_text'] = df['speech_turn'].apply(normalize_text)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['normalized_text'])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find duplicates
    duplicates = set()
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > similarity_threshold:
                duplicates.add(j)
    
    # Remove duplicates
    df = df.drop(list(duplicates))
    df = df.drop('normalized_text', axis=1)
    return df

def add_out_of_bounds_examples(df):
    """Add a large number of out-of-bounds examples to make them the majority."""
    out_of_bounds_examples = [
        # General conversation
        "How is the weather today?",
        "Nice car!",
        "What time is it?",
        "Can you tell me where the nearest gas station is?",
        "I love this neighborhood.",
        "The traffic is terrible today.",
        "Do you know any good restaurants around here?",
        "What's your favorite color?",
        "The sunset is beautiful.",
        "I need directions to the mall.",
        "What's your name?",
        "How are you doing today?",
        "Nice to meet you!",
        "Have a great day!",
        "Thank you for your help.",
        "I'm looking for my friend.",
        "Can you help me find my way?",
        "The park is lovely this time of year.",
        "I'm new to this area.",
        "What's the best way to get downtown?",
        
        # Police-related but non-force
        "I need to report a stolen bike.",
        "Can you help me with directions?",
        "I lost my wallet.",
        "There's a suspicious package.",
        "I need to file a noise complaint.",
        "Can you help me find my lost dog?",
        "I need to report a break-in.",
        "There's a broken street light.",
        "I need to get a police report.",
        "Can you help me with parking?",
        
        # Common phrases
        "Excuse me, officer.",
        "Thank you for your service.",
        "Have a good day.",
        "Stay safe out there.",
        "I appreciate your help.",
        "Is there anything I can do?",
        "Let me know if you need anything.",
        "I understand.",
        "I'll be careful.",
        "Take care.",
        
        # Questions about police work
        "How long have you been an officer?",
        "What's the most common call you get?",
        "Do you like your job?",
        "What's the hardest part of your job?",
        "How do you stay safe?",
        "What's your favorite part of the job?",
        "Do you work with a partner?",
        "What's your typical day like?",
        "How do you handle stress?",
        "What made you want to be an officer?",
        
        # Small talk
        "The weather is nice today.",
        "How's your day going?",
        "Did you see the game last night?",
        "What's your favorite food?",
        "Do you have any pets?",
        "What do you do for fun?",
        "Where are you from?",
        "How long have you lived here?",
        "What's your favorite season?",
        "Do you like music?",
        
        # Complaints about non-force issues
        "The traffic light is broken.",
        "There's a pothole in the road.",
        "The street is too noisy.",
        "Someone's dog is barking too much.",
        "The garbage wasn't picked up.",
        "The street needs cleaning.",
        "There's graffiti on the wall.",
        "The sidewalk is cracked.",
        "The street sign is missing.",
        "The street light is flickering.",
        
        # General questions
        "What's the speed limit here?",
        "Where can I park?",
        "Is this area safe?",
        "What time does the park close?",
        "Where's the nearest hospital?",
        "How do I get to the airport?",
        "What's the best route to downtown?",
        "Where's the post office?",
        "Is there a gas station nearby?",
        "Where's the nearest restaurant?",
        
        # Social interactions
        "Good morning, officer.",
        "Good afternoon.",
        "Good evening.",
        "Have a nice day.",
        "See you around.",
        "Take care.",
        "Stay safe.",
        "Be careful.",
        "Thanks again.",
        "You're welcome.",
        
        # Non-emergency situations
        "I found a lost wallet.",
        "There's a lost child.",
        "I need directions.",
        "Can you help me with my car?",
        "I need to report a lost item.",
        "There's a suspicious person.",
        "I need to file a report.",
        "Can you help me with paperwork?",
        "I need information.",
        "Can you answer some questions?"
    ]
    
    # Create DataFrame with out-of-bounds examples
    out_of_bounds_df = pd.DataFrame({
        'speech_turn': out_of_bounds_examples,
        'use_of_force_level': [1] * len(out_of_bounds_examples)  # Level 1 for non-force
    })
    
    # Combine with original dataset
    combined_df = pd.concat([df, out_of_bounds_df], ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure out-of-bounds examples are the majority (70%)
    target_out_of_bounds = int(len(combined_df) * 0.7)
    current_out_of_bounds = len(combined_df[combined_df['use_of_force_level'] == 1])
    
    if current_out_of_bounds < target_out_of_bounds:
        # Add more out-of-bounds examples if needed
        additional_examples = target_out_of_bounds - current_out_of_bounds
        extra_examples = pd.DataFrame({
            'speech_turn': out_of_bounds_examples[:additional_examples],
            'use_of_force_level': [1] * additional_examples
        })
        combined_df = pd.concat([combined_df, extra_examples], ignore_index=True)
    
    # Final shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df

def add_discussion_examples(df):
    """Add examples of discussing force without actual force."""
    discussion_examples = [
        "I heard about a shooting incident yesterday.",
        "The police used force to apprehend the suspect.",
        "There was a report of excessive force.",
        "The officer had to use his taser.",
        "They had to restrain him.",
        "The suspect was armed with a knife.",
        "He threatened to use force.",
        "The situation required force.",
        "They had to use pepper spray.",
        "The suspect was resisting arrest.",
        "The officer had to draw his weapon.",
        "They had to use physical force.",
        "The suspect was threatening violence.",
        "The situation escalated to force.",
        "They had to use non-lethal force.",
        "The officer had to use his baton.",
        "The suspect was armed.",
        "They had to use restraint.",
        "The situation required physical intervention.",
        "The officer had to use his taser."
    ]
    
    discussion_df = pd.DataFrame({
        'speech_turn': discussion_examples,
        'use_of_force_level': [1] * len(discussion_examples)  # Level 1 for discussion
    })
    
    return pd.concat([df, discussion_df], ignore_index=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv('Final_Enhanced_Use_of_Force_Dataset_Full.csv')
    
    print("Removing duplicates...")
    df = remove_duplicates(df)
    
    print("Adding out-of-bounds examples...")
    df = add_out_of_bounds_examples(df)
    
    print("Adding discussion examples...")
    df = add_discussion_examples(df)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal dataset statistics:")
    print(f"Total examples: {len(df)}")
    print("\nClass distribution:")
    print(df['use_of_force_level'].value_counts().sort_index())
    
    # Save the cleaned dataset
    output_file = 'Cleaned_Use_of_Force_Dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to {output_file}")

if __name__ == "__main__":
    main() 