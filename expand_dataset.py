import pandas as pd
import numpy as np
from transformers import pipeline
import random
import re

def load_current_dataset():
    df = pd.read_csv('Cleaned_Use_of_Force_Dataset.csv')
    return df

def analyze_patterns(df):
    patterns = {
        'level_0': [],  # Non-force related
        'level_1': [],  # Verbal commands
        'level_2': [],  # Soft empty hand
        'level_3': [],  # Hard empty hand
        'level_4': [],  # Less lethal
        'level_5': [],  # Lethal force
        'out_of_bounds': [],  # Force-related but not actual force
        'discussion': []  # Discussion of force
    }
    
    for _, row in df.iterrows():
        text = row['speech_turn'].lower()
        level = row['use_of_force_level']
        
        # Categorize based on content and level
        if level == 0:
            patterns['level_0'].append(text)
        elif level == 1:
            patterns['level_1'].append(text)
        elif level == 2:
            patterns['level_2'].append(text)
        elif level == 3:
            patterns['level_3'].append(text)
        elif level == 4:
            patterns['level_4'].append(text)
        elif level == 5:
            patterns['level_5'].append(text)
        
        # Check for out-of-bounds and discussion patterns
        if any(word in text for word in ['force', 'taser', 'gun', 'weapon', 'fight']):
            if 'discuss' in text or 'talk' in text or 'explain' in text:
                patterns['discussion'].append(text)
            elif not any(action in text for action in ['use', 'deploy', 'draw', 'point', 'fire']):
                patterns['out_of_bounds'].append(text)
    
    return patterns

def generate_variations(text, level):
    variations = []
    
    # Common variations for different force levels
    if level == 1:  # Verbal commands
        prefixes = [
            'please', 'i need you to', 'you need to', 'i want you to', 'you have to',
            "i'm asking you to", "i'm telling you to", "i'm ordering you to",
            'you must', 'i require you to', "i'm requesting you to", "i'm directing you to"
        ]
        suffixes = [
            'right now', 'immediately', 'as soon as possible', 'right away',
            'this instant', 'without delay', 'at once', 'now',
            'or there will be consequences', 'or i will take action'
        ]
        
        for prefix in prefixes:
            for suffix in suffixes:
                variations.append(f"{prefix} {text} {suffix}")
    
    elif level == 2:  # Soft empty hand
        variations.extend([
            f"i'm going to guide you {text}",
            f"let me help you {text}",
            f"i'll assist you to {text}",
            f"i'm going to gently guide you to {text}",
            f"i'll help you move to {text}",
            f"let me assist you to {text}",
            f"i'm going to help you to {text}",
            f"i'll gently guide you to {text}",
            f"let me help you move to {text}",
            f"i'm going to assist you to {text}"
        ])
    
    elif level == 3:  # Hard empty hand
        variations.extend([
            f"i'm forced to restrain you to {text}",
            f"i have to physically control you to {text}",
            f"i need to use physical force to {text}",
            f"i'm going to physically restrain you to {text}",
            f"i have to use physical control to {text}",
            f"i need to physically restrain you to {text}",
            f"i'm forced to use physical control to {text}",
            f"i have to physically restrain you to {text}",
            f"i need to use physical restraint to {text}",
            f"i'm going to use physical force to {text}"
        ])
    
    elif level == 4:  # Less lethal
        variations.extend([
            f"i'm deploying less lethal force to {text}",
            f"i need to use less lethal options to {text}",
            f"i'm forced to use less lethal measures to {text}",
            f"i'm going to deploy less lethal force to {text}",
            f"i have to use less lethal options to {text}",
            f"i need to deploy less lethal measures to {text}",
            f"i'm forced to use less lethal force to {text}",
            f"i have to deploy less lethal options to {text}",
            f"i need to use less lethal force to {text}",
            f"i'm going to use less lethal measures to {text}"
        ])
    
    elif level == 5:  # Lethal force
        variations.extend([
            f"i'm forced to use lethal force to {text}",
            f"i need to use deadly force to {text}",
            f"i have no choice but to use lethal force to {text}",
            f"i'm going to use lethal force to {text}",
            f"i have to use deadly force to {text}",
            f"i need to deploy lethal force to {text}",
            f"i'm forced to use deadly force to {text}",
            f"i have to deploy lethal force to {text}",
            f"i need to use lethal measures to {text}",
            f"i'm going to use deadly force to {text}"
        ])
    
    return variations

def generate_discussion_examples():
    discussion_templates = [
        "we were discussing the use of force policy",
        "let me explain our force continuum",
        "i want to talk about force options",
        "we need to discuss force protocols",
        "let's review force policies",
        "i'm explaining force procedures",
        "we're training on force techniques",
        "let me clarify our force guidelines",
        "i want to discuss force scenarios",
        "we're reviewing force documentation",
        "i'm explaining our force policy",
        "let's discuss force procedures",
        "we need to review force protocols",
        "i want to explain force guidelines",
        "we're discussing force training",
        "let me review force policies",
        "i'm clarifying force procedures",
        "we need to discuss force options",
        "let's review force documentation",
        "i want to explain force protocols"
    ]
    return discussion_templates

def generate_out_of_bounds_examples():
    out_of_bounds_templates = [
        "i see a weapon in the area",
        "there might be a threat nearby",
        "i noticed something concerning",
        "there's a potential risk here",
        "i see something suspicious",
        "there's a possible threat",
        "i noticed a concerning situation",
        "there might be danger here",
        "i see something that needs attention",
        "there's a potential problem",
        "i see a suspicious object",
        "there might be a weapon nearby",
        "i noticed a potential threat",
        "there's something concerning here",
        "i see a possible risk",
        "there might be a dangerous situation",
        "i noticed something suspicious",
        "there's a potential hazard",
        "i see a concerning object",
        "there might be a risk here"
    ]
    return out_of_bounds_templates

def generate_non_force_examples():
    non_force_templates = [
        "how can i help you today?",
        "what seems to be the problem?",
        "is everything alright?",
        "do you need assistance?",
        "what's going on here?",
        "can i help you with something?",
        "what brings you here today?",
        "is there something i can do for you?",
        "what's the situation?",
        "how may i assist you?",
        "what's happening here?",
        "do you need any help?",
        "what's the problem?",
        "can i be of assistance?",
        "what's going on?",
        "how can i be of help?",
        "what's the issue?",
        "do you need support?",
        "what's the matter?",
        "how may i help you?"
    ]
    return non_force_templates

def expand_dataset(target_size=10000):
    # Load current dataset
    df = load_current_dataset()
    current_size = len(df)
    
    # Define balanced proportions (approximately 15% each for levels 0-4, 10% for level 5)
    balanced_proportions = {
        'level_0': 0.15,  # Non-force related
        'level_1': 0.15,  # Verbal commands
        'level_2': 0.15,  # Soft empty hand
        'level_3': 0.15,  # Hard empty hand
        'level_4': 0.15,  # Less lethal
        'level_5': 0.10,  # Lethal force
        'out_of_bounds': 0.10,  # Force-related but not actual force
        'discussion': 0.10  # Discussion of force
    }
    
    # Generate new examples
    new_examples = []
    
    # Generate examples for each category
    for category, proportion in balanced_proportions.items():
        target_count = int(target_size * proportion)
        
        if category.startswith('level_'):
            level = int(category.split('_')[1])
            if level == 0:
                # Generate non-force examples
                templates = generate_non_force_examples()
                for _ in range(target_count):
                    new_examples.append({
                        'speech_turn': random.choice(templates),
                        'use_of_force_level': level
                    })
            else:
                # Generate variations for force levels
                for _ in range(target_count):
                    # Use a mix of templates and variations
                    if random.random() < 0.5:
                        # Use existing patterns if available
                        patterns = analyze_patterns(df)
                        if patterns[category]:
                            base_text = random.choice(patterns[category])
                            variations = generate_variations(base_text, level)
                            if variations:
                                new_examples.append({
                                    'speech_turn': random.choice(variations),
                                    'use_of_force_level': level
                                })
                    else:
                        # Use predefined templates
                        if level == 1:
                            templates = generate_variations("stop", level)
                        elif level == 2:
                            templates = generate_variations("move", level)
                        elif level == 3:
                            templates = generate_variations("stop resisting", level)
                        elif level == 4:
                            templates = generate_variations("stop", level)
                        elif level == 5:
                            templates = generate_variations("drop the weapon", level)
                        
                        if templates:
                            new_examples.append({
                                'speech_turn': random.choice(templates),
                                'use_of_force_level': level
                            })
        
        elif category == 'discussion':
            # Generate discussion examples
            templates = generate_discussion_examples()
            for _ in range(target_count):
                new_examples.append({
                    'speech_turn': random.choice(templates),
                    'use_of_force_level': 0
                })
        
        elif category == 'out_of_bounds':
            # Generate out-of-bounds examples
            templates = generate_out_of_bounds_examples()
            for _ in range(target_count):
                new_examples.append({
                    'speech_turn': random.choice(templates),
                    'use_of_force_level': 0
                })
    
    # Combine original and new examples
    expanded_df = pd.concat([
        df,
        pd.DataFrame(new_examples)
    ], ignore_index=True)
    
    # Shuffle the dataset
    expanded_df = expanded_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save expanded dataset
    expanded_df.to_csv('Expanded_Use_of_Force_Dataset.csv', index=False)
    
    # Print statistics
    print(f"Original dataset size: {current_size}")
    print(f"Expanded dataset size: {len(expanded_df)}")
    print("\nClass distribution:")
    print(expanded_df['use_of_force_level'].value_counts().sort_index())

if __name__ == "__main__":
    expand_dataset(10000) 