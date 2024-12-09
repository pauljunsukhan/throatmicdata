import requests
import random
import os
import argparse

def download_common_voice_sentences(num_sentences=180):
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
    
    if len(sentences) < num_sentences:
        print(f"Warning: Only found {len(sentences)} suitable sentences, needed {num_sentences}")
        selected_sentences = sentences
    else:
        # Select random sentences
        selected_sentences = random.sample(sentences, num_sentences)
    
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

def main():
    parser = argparse.ArgumentParser(description="Download and prepare Common Voice sentences")
    parser.add_argument("--sentences", type=int, default=180,
                      help="Number of sentences to download (default: 180)")
    parser.add_argument("--append", action="store_true",
                      help="Append to existing prompts.txt instead of overwriting")
    
    args = parser.parse_args()
    
    if args.append and os.path.exists('prompts.txt'):
        # Read existing prompts
        with open('prompts.txt', 'r', encoding='utf-8') as f:
            existing_prompts = set(line.strip() for line in f)
        print(f"Found {len(existing_prompts)} existing prompts")
        
        # Download new sentences
        temp_file = 'prompts_temp.txt'
        os.rename('prompts.txt', temp_file)
        success = download_common_voice_sentences(args.sentences)
        
        if success:
            # Combine old and new prompts
            with open('prompts.txt', 'r', encoding='utf-8') as f:
                new_prompts = set(line.strip() for line in f)
            
            # Merge prompts
            all_prompts = existing_prompts.union(new_prompts)
            
            # Write back all prompts
            with open('prompts.txt', 'w', encoding='utf-8') as f:
                for prompt in all_prompts:
                    f.write(f"{prompt}\n")
            
            print(f"\nTotal unique prompts after merging: {len(all_prompts)}")
            
            # Clean up
            os.remove(temp_file)
        else:
            # Restore original file if download failed
            os.rename(temp_file, 'prompts.txt')
            print("Restored original prompts.txt due to download failure")
    else:
        download_common_voice_sentences(args.sentences)

if __name__ == "__main__":
    main() 