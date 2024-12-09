import requests
import random
import os

def download_common_voice_sentences():
    """Download Common Voice sentences from the Mozilla repository"""
    # List of sentence sources in the Common Voice repository
    sources = [
        "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/en/wiki.en.txt",
        "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/en/europarl-v7.en.txt"
    ]
    
    all_sentences = set()  # Using a set to avoid duplicates
    
    for url in sources:
        try:
            print(f"Downloading sentences from {url.split('/')[-1]}...")
            response = requests.get(url)
            response.raise_for_status()
            sentences = response.text.splitlines()
            
            # Filter out empty lines, strip whitespace, and add to set
            valid_sentences = {s.strip() for s in sentences if s.strip()}
            all_sentences.update(valid_sentences)
            print(f"Added {len(valid_sentences)} sentences from this source")
            
        except requests.RequestException as e:
            print(f"Warning: Error downloading from {url}: {e}")
            continue
    
    if not all_sentences:
        print("Error: Could not download any sentences from any source")
        return False
    
    # Convert to list and filter sentences
    sentences = list(all_sentences)
    # Filter out sentences that are too long or too short
    sentences = [s for s in sentences if 10 <= len(s.split()) <= 20]
    
    # Calculate how many sentences we need for 30 minutes
    # Assuming 10 seconds per recording
    needed_sentences = int((30 * 60) / 10)  # 30 minutes * 60 seconds / 10 seconds per recording
    
    if len(sentences) < needed_sentences:
        print(f"Warning: Only found {len(sentences)} suitable sentences, needed {needed_sentences}")
        selected_sentences = sentences
    else:
        # Select random sentences
        selected_sentences = random.sample(sentences, needed_sentences)
    
    # Save to prompts.txt
    with open('prompts.txt', 'w', encoding='utf-8') as f:
        for sentence in selected_sentences:
            f.write(f"{sentence}\n")
    
    print(f"\nSuccessfully prepared {len(selected_sentences)} sentences")
    print(f"This will result in approximately {len(selected_sentences) * 10 / 60:.1f} minutes of audio")
    print("\nSentence length statistics:")
    lengths = [len(s.split()) for s in selected_sentences]
    print(f"Average words per sentence: {sum(lengths)/len(lengths):.1f}")
    print(f"Min words: {min(lengths)}, Max words: {max(lengths)}")
    
    return True

if __name__ == "__main__":
    download_common_voice_sentences() 