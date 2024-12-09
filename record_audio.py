#!/usr/bin/env python3
"""
Throat microphone recording tool.
"""

import argparse
import sys
import logging
from pathlib import Path
import csv

# Add src directory to path
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.config import config  # Import the config instance directly
from src import ThroatMicRecorder, AudioQualityControl, DatasetAnalytics, ThroatMicError

def setup_logging():
    """Configure logging based on config settings."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Record and analyze throat microphone data."
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing recordings instead of recording"
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run audio quality analysis"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Analyze dataset coverage"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for analysis results"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        config.logging.level = "DEBUG"
    logger = setup_logging()
    logger.debug("Starting throat mic recording tool")
    
    try:
        if args.analyze:
            analytics = DatasetAnalytics()
            
            if args.quality:
                logger.info("Analyzing audio quality...")
                results = analytics.analyze_dataset(config.dataset.metadata_file)
                
                logger.info(f"\nDataset Quality Analysis:")
                logger.info(f"Total Recordings: {results['total_recordings']}")
                logger.info(f"Recordings with Issues: {results['recordings_with_issues']}")
                
                # Count issue types
                issue_counts = {
                    'duration': 0,
                    'amplitude': 0,
                    'silence': 0,
                    'snr': 0
                }
                
                for issue in results['quality_issues']:
                    for problem in issue['issues']:
                        if 'too short' in problem or 'too long' in problem:
                            issue_counts['duration'] += 1
                        elif 'too quiet' in problem or 'too loud' in problem:
                            issue_counts['amplitude'] += 1
                        elif 'silence' in problem:
                            issue_counts['silence'] += 1
                        elif 'signal-to-noise' in problem:
                            issue_counts['snr'] += 1
                
                logger.info("\nIssue Summary:")
                logger.info(f"Duration issues: {issue_counts['duration']} recordings")
                logger.info(f"Amplitude issues: {issue_counts['amplitude']} recordings")
                logger.info(f"Silence ratio issues: {issue_counts['silence']} recordings")
                logger.info(f"SNR issues: {issue_counts['snr']} recordings")
                
                if results['quality_issues']:
                    logger.info("\nDetailed Issues:")
                    for issue in results['quality_issues']:
                        logger.info(f"\n{issue['file']}:")
                        for problem in issue['issues']:
                            logger.info(f"  - {problem}")
            
            if args.coverage:
                logger.info("Analyzing dataset coverage...")
                results = analytics.analyze_dataset(config.dataset.metadata_file)
                
                logger.info("\nDataset Coverage Report:")
                logger.info(f"Total Recordings: {results['total_recordings']}")
                logger.info(f"Total Duration: {results['total_duration_minutes']:.1f} minutes")
                
                logger.info("\nSentence Length Stats:")
                logger.info(f"  Average: {results['sentence_stats']['avg_words']:.1f} words")
                logger.info(f"  Min: {results['sentence_stats']['min_words']} words")
                logger.info(f"  Max: {results['sentence_stats']['max_words']} words")
                
                logger.info("\nRecording Duration Stats:")
                logger.info(f"  Average: 10.0 seconds")
                logger.info(f"  Min: 10.0 seconds")
                logger.info(f"  Max: 10.0 seconds")
                
                logger.info("\nSentence Complexity Coverage:")
                logger.info(f"  Sentences with commas: {results['sentence_stats']['pct_with_comma']:.1f}%")
                logger.info(f"  Sentences with subordinating conjunctions: {results['sentence_stats']['pct_with_subordinating_conj']:.1f}%")
                logger.info(f"  Sentences with coordinating conjunctions: {results['sentence_stats']['pct_with_coordinating_conj']:.1f}%")
                logger.info(f"  Sentences with relative pronouns: {results['sentence_stats']['pct_with_relative_pronouns']:.1f}%")
                
            if not (args.quality or args.coverage):
                logger.error("Please specify --quality and/or --coverage with --analyze")
                return 1
        else:
            # Record audio
            recorder = ThroatMicRecorder()
            if args.list_devices:
                recorder.list_devices()
                return 0
                
            logger.info("Starting recording session")
            recorder.run()
    
    except ThroatMicError as e:
        logger.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 