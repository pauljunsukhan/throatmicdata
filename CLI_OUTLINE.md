# Throat Mic Recording Tool


record_audio.py
├── Argument parsing (--list-devices, --analyze, etc.)
├── Logging setup
├── Main program flow
└── Error handling

recorder.py (ThroatMicRecorder)
├── Device selection menu
├── Main menu
├── Recording interface
│   ├── Prompt display
│   ├── Recording process
│   ├── Level meter
│   └── Post-recording options
├── Progress tracking
└── Sentence management menu



## Initial Setup
1. Audio Device Selection
   - Shows numbered list of available devices
   - Example devices:
     1. Paul's iPhone MkII Microphone
     2. External Microphone
     3. MacBook Air Microphone
     etc.
   - Performs 1-second test of selected device

## Main Menu
Options:
1. Continue recording prompts
2. View progress
3. Toggle level meter
4. Manage sentences
5. Exit

## View Progress Menu
Shows statistics:
- Total sentences: XXX
- Recorded sentences: XXX/XXX (XX%)
- Remaining sentences: XXX
- Total recording time: XX.X minutes
- Trashed sentences: XXX

## Manage Sentences Menu
1. Download more sentences
   - Prompts for number to download
   - Shows download progress
   - Reports number of sentences added

2. Add custom sentences
   - Enter sentences one per line
   - Empty line to finish
   - Shows number of valid sentences added

3. View sentence stats
   - Total sentences
   - Available sentences
   - Used sentences
   - Trashed sentences

4. View trashed sentences
   - Shows numbered list of trashed sentences
   - Empty message if no trashed sentences

5. Analyze dataset
   Shows:
   ```
   Dataset Analysis Results:
   
   Quality Analysis:
   Total Recordings: XXX
   Total Duration: XX.X minutes
   Recordings with Issues: XX (XX.X%)
   
   Duration Statistics:
     Mean: XX.Xs
     Median: XX.Xs
     Range: XX.Xs - XX.Xs
     Std Dev: XX.Xs
   
   Duration Distribution:
     <6s:    XX recordings (XX.X%)
     6-8s:   XX recordings (XX.X%)
     8-10s:  XX recordings (XX.X%)
     10-12s: XX recordings (XX.X%)
     12-15s: XX recordings (XX.X%)
     >15s:   XX recordings (XX.X%)
   
   Coverage Analysis:
   Vocabulary:
     Total Words: XXXX
     Unique Words: XXXX
     Vocabulary Density: XX.XX
   
   Sentence Structure:
     Average Length: XX.X words
     Length Range: XX - XX words
     Unique Patterns: XXX
   
   Part of Speech Distribution:
     ADJ: XX.X%
     ADP: XX.X%
     ADV: XX.X%
     ...
   
   Sentence Complexity:
   Average Complexity Score: XX.X
   
   Complexity Distribution:
     Simple: XXX sentences
     Moderate: XXX sentences
     Complex: XXX sentences
     Very Complex: XXX sentences
   
   Syntactic Features:
     Average Tree Depth: XX.X
     Subordinate Clauses: XX.X%
     Relative Clauses: XX.X%
     Adverbial Clauses: XX.X%
   
   Vocabulary Features:
     Average Rare Words: XX.X
     Average Technical Terms: XX.X
     Average Abstract Concepts: XX.X
     Average Word Length: XX.X
   
   Logical Features:
     Causal Relations: XX.X%
     Comparisons: XX.X%
     Conditionals: XX.X%
   ```
   Press Space to generate detailed report, or Enter to continue

   Generates a comprehensive report file that includes everything from the basic analysis plus:
    - Timestamp of analysis
    - Top 20 most frequent words with counts
    - Individual file quality issues including:
        * Files with clipping
        * Files with poor SNR
        * Files outside duration limits
        * Files with excessive silence
    - Saves as: `analysis_report_YYYYMMDD_HHMMSS.txt` in project directory

    The detailed report is useful for:
    - Tracking dataset quality over time
    - Identifying specific recordings that need attention
    - Detailed vocabulary analysis
    - Archival purposes

6. Back to main menu

## Recording Interface
1. Prompt Display
   ```
   ==================================================
   Prompt XXX/YYY:
   
   [Sentence to be recorded]
   
   ==================================================
   ```

2. Pre-Recording Options
   - Press Enter → Start recording
   - 't' → Trash sentence
   - 'q' → Quit to menu

3. Recording Process
   - Countdown: "Starting in: 3...2...1..."
   - Progress bar shows recording progress
   - Shows estimated and actual recording duration
   - Visual level meter (when enabled)
     ```
     Level: ████████░░░░░░░░░░░░  # Normal level (green)
     Level: ████████████░░░░░░░░  # High level (yellow)
     Level: █████████████████░░░  # CLIPPING! (red)
     ```
   - Post-recording warnings:
     ```
     ⚠️  Warning: Clipping was detected in this recording!
     ```
     (Only shown if clipping occurred during recording)

4. Post-Recording Menu
   Options:
   1. Play recording
   2. Re-record
   3. Save and continue to next
   4. Save and quit to menu
   5. Discard and quit to menu
   6. Trash sentence and continue to next
   (Press Enter defaults to option 3)
   (Press Space defaults to option 6)

## Progress Tracking
- Shows current prompt number (e.g., "Prompt 733/909")
- Displays completion percentage
- Tracks total recording time
- Shows estimated speaking time per sentence

# Command Line Options

## Basic Analysis Display (--analyze)
```bash
python -m record_audio --analyze
```
Shows interactive analysis with core metrics (as shown in Manage Sentences Menu -> Analyze dataset).

## Detailed Report (--analyze --output report.txt or Space after analysis)
```bash
python -m record_audio --analyze --output report.txt
```
Generates a comprehensive report file that includes everything from the basic analysis plus:
- Timestamp of analysis
- Top 20 most frequent words with counts
- Individual file quality issues including:
  * Files with clipping
  * Files with poor SNR
  * Files outside duration limits
  * Files with excessive silence
- Saves as: `analysis_report_YYYYMMDD_HHMMSS.txt` in project directory

The detailed report is useful for:
- Tracking dataset quality over time
- Identifying specific recordings that need attention
- Detailed vocabulary analysis
- Archival purposes
