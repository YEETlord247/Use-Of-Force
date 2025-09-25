import pandas as pd
import random
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def generate_variations(template):
    """Generate variations of a template by adding modifiers and changing word order."""
    variations = [template]
    
    # Add intensity modifiers
    intensity_modifiers = ["immediately", "quickly", "now", "right now", "at once", "firmly"]
    variations.extend([f"{mod} {template}" for mod in intensity_modifiers])
    
    # Add context modifiers
    context_modifiers = ["for safety", "for protection", "for security", "to maintain order", 
                        "to ensure compliance", "for everyone's safety"]
    variations.extend([f"{template} {mod}" for mod in context_modifiers])
    
    # Add situational prefixes
    situation_prefixes = ["due to the situation", "given the circumstances", "based on your actions",
                         "for your safety", "considering the threat", "in this situation"]
    variations.extend([f"{prefix}, {template}" for prefix in situation_prefixes])
    
    return variations

def generate_test_examples():
    test_examples = []
    
    # Level 0 (Non-force) - Different scenarios with context
    level_0_templates = [
        "could you tell me what happened?",
        "may i see your identification please?",
        "where are you coming from today?",
        "how long have you been here?",
        "what brings you to this area?",
        "have you seen anything suspicious?",
        "when did you last see them?",
        "can you describe what you witnessed?",
        "what time did this occur?",
        "who else was present?",
        "how can I assist you today?",
        "do you need any help?",
        "are you familiar with this area?",
        "have you been drinking tonight?",
        "where were you headed?",
        "can you explain what happened?",
        "do you live nearby?",
        "what's your emergency?",
        "how long ago did this happen?",
        "can you provide more details?"
    ]
    
    # Level 1 (Verbal Commands) - Clear, authoritative commands
    level_1_templates = [
        "stop what you're doing",
        "put your hands where I can see them",
        "step out of the vehicle",
        "move away from the area",
        "stay where you are",
        "back away slowly",
        "keep your distance",
        "show me your hands",
        "remain still",
        "face away from me",
        "get on your knees",
        "turn around slowly",
        "exit the building",
        "drop what you're holding",
        "stop moving",
        "maintain your distance",
        "step back now",
        "keep your hands visible",
        "do not approach",
        "stay behind the line"
    ]
    
    # Level 2 (Soft Empty Hand) - Control techniques
    level_2_templates = [
        "i need to escort you to a safe location",
        "i'm going to guide you to the ground",
        "i'll help you to a seated position",
        "we need to move to a secure area",
        "i'm going to hold your arm",
        "let me guide you away from here",
        "i need to direct you to safety",
        "i'll assist you in moving back",
        "we're going to walk this way",
        "i need to guide you away",
        "i'm going to lead you to safety",
        "let me help you move",
        "i need to position you here",
        "i'll direct you to a safe spot",
        "we need to relocate you",
        "i'm going to assist your movement",
        "let me help you step back",
        "i need to guide you away",
        "i'll help you maintain balance",
        "we're moving to safety"
    ]
    
    # Level 3 (Hard Empty Hand) - Control and compliance
    level_3_templates = [
        "you're under arrest, stop resisting",
        "i'm taking you into custody",
        "stop fighting or force will be used",
        "physical restraint is necessary",
        "comply or force will be required",
        "resistance will lead to force",
        "you must be physically restrained",
        "force will be used if necessary",
        "physical control is required",
        "stop resisting arrest",
        "you're forcing me to use control techniques",
        "physical force may be necessary",
        "compliance is required now",
        "resist and force will be used",
        "you must be controlled",
        "physical restraint is coming",
        "force will be applied",
        "control techniques will be used",
        "you're requiring physical force",
        "restraint is necessary"
    ]
    
    # Level 4 (Less Lethal) - Less lethal force options
    level_4_templates = [
        "less lethal force will be deployed",
        "taser deployment imminent",
        "preparing to use less lethal options",
        "less lethal munitions will be used",
        "taser warning, stop resisting",
        "less lethal force is authorized",
        "preparing less lethal deployment",
        "less lethal options are ready",
        "taser will be deployed",
        "less lethal force is coming",
        "preparing to deploy less lethal",
        "less lethal munitions authorized",
        "taser deployment warning",
        "less lethal force warning",
        "preparing less lethal options",
        "less lethal deployment imminent",
        "taser will be used",
        "less lethal force prepared",
        "deploying less lethal options",
        "less lethal warning given"
    ]
    
    # Level 5 (Lethal) - Lethal force warnings
    level_5_templates = [
        "lethal force is authorized",
        "deadly force will be used",
        "drop the weapon or we'll shoot",
        "lethal force is imminent",
        "deadly force is justified",
        "weapons free, lethal force authorized",
        "prepare for lethal engagement",
        "lethal force is necessary",
        "deadly force warning given",
        "lethal options are authorized",
        "deadly force is imminent",
        "lethal force will be used",
        "weapons authorized for use",
        "lethal force warning issued",
        "deadly force is authorized",
        "lethal engagement imminent",
        "deadly force warning",
        "lethal force prepared",
        "weapons are authorized",
        "lethal force warning"
    ]
    
    # Generate examples with variations for each level
    for level, templates in enumerate([level_0_templates, level_1_templates, level_2_templates, 
                                    level_3_templates, level_4_templates, level_5_templates]):
        level_examples = []
        for template in templates:
            variations = generate_variations(template)
            level_examples.extend(variations)
        
        # Randomly select 100 examples for each level
        selected_examples = random.sample(level_examples, min(100, len(level_examples)))
        for example in selected_examples:
            test_examples.append({
                'speech_turn': example,
                'use_of_force_level': level
            })
    
    # Shuffle the test set
    random.shuffle(test_examples)
    
    # Create DataFrame and save
    test_df = pd.DataFrame(test_examples)
    test_df.to_csv('test_set.csv', index=False)
    
    # Print statistics
    print(f"Test set size: {len(test_df)}")
    print("\nClass distribution:")
    print(test_df['use_of_force_level'].value_counts().sort_index())

if __name__ == "__main__":
    generate_test_examples() 