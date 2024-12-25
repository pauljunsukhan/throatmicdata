#!/usr/bin/env python3
"""
Throat microphone recording tool.
"""

import argparse
import sys
import logging
from pathlib import Path
import csv
from typing import Optional
from logging.handlers import RotatingFileHandler

# Add src directory to path
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src import (
    ThroatMicRecorder,
    AudioQualityControl,
    DatasetAnalytics,
    ThroatMicError,
    Config,
    config
)

def setup_logging() -> logging.Logger:
    """
    Configure logging based on config settings.
    Uses a rotating file handler to limit log file size to ~5000 lines.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            RotatingFileHandler(
                config.logging.file,
                maxBytes=1024*1024,  # 1MB ~ 5000 lines of typical log entries
                backupCount=1
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main() -> int:
    """
    Main entry point for the throat microphone recording tool.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
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
            analytics = DatasetAnalytics(config)  # Use imported config
            results = None
            
            # Validate output path if specified
            if args.output:
                output_path = Path(args.output)
                if output_path.exists():
                    logger.warning(f"Output file {args.output} already exists, will be overwritten")
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Cannot create output directory: {e}")
                    return 1
            
            # Analyze dataset once if needed
            if args.quality or args.coverage:
                logger.info("Analyzing dataset...")
                results = analytics.analyze_dataset(config.dataset.metadata_file)
                if 'error' in results:
                    logger.error(f"Analysis failed: {results['error']}")
                    return 1
            
            if args.quality:
                logger.info(f"\nDataset Quality Analysis:")
                logger.info(f"Total Recordings: {results['total_recordings']}")
                logger.info(f"Total Duration: {results['total_duration_minutes']:.1f} minutes")
                logger.info(f"Recordings with Issues: {results['recordings_with_issues']}")
                
                # Updated quality reporting using new metrics
                if results['quality_issues']:
                    logger.info("\nQuality Issues Summary:")
                    issue_types = {}
                    for issue in results['quality_issues']:
                        for problem in issue['issues']:
                            issue_type = problem.split(':')[0]
                            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                    
                    for issue_type, count in issue_types.items():
                        logger.info(f"{issue_type}: {count} recordings")

                # Add key input handling
                print("\nPress Space to generate a detailed report, or Enter to continue...")
                import sys, tty, termios
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                
                # Only generate report if Space was pressed
                if ch == ' ':
                    if args.output:
                        logger.info(f"\nGenerating detailed report...")
                        try:
                            report = analytics.generate_report(config.dataset.metadata_file, args.output)
                            logger.info(f"Report saved to {args.output}")
                        except Exception as e:
                            logger.error(f"Failed to generate report: {e}")
                            return 1
            
            if args.coverage:
                logger.info("\nDataset Coverage Report:")
                logger.info(f"Total Recordings: {results['total_recordings']}")
                logger.info(f"Total Duration: {results['total_duration_minutes']:.1f} minutes")
                
                # New complexity metrics
                complexity = results['sentence_complexity']
                logger.info("\nSentence Complexity:")
                logger.info(f"Average Complexity Score: {complexity['avg_complexity']:.1f}")
                logger.info("\nComplexity Distribution:")
                for level, count in complexity['complexity_distribution'].items():
                    logger.info(f"  {level.title()}: {count} sentences")
                
                logger.info("\nSyntactic Features:")
                syntactic = complexity['syntactic_features']
                logger.info(f"  Average Tree Depth: {syntactic['avg_tree_depth']:.1f}")
                logger.info(f"  Subordinate Clauses: {syntactic['pct_subordinate']:.1f}%")
                logger.info(f"  Relative Clauses: {syntactic['pct_relative']:.1f}%")
                logger.info(f"  Adverbial Clauses: {syntactic['pct_adverbial']:.1f}%")
                
                logger.info("\nVocabulary Features:")
                vocab = complexity['vocabulary_features']
                logger.info(f"  Average Rare Words: {vocab['avg_rare_words']:.1f}")
                logger.info(f"  Average Technical Terms: {vocab['avg_technical_terms']:.1f}")
                logger.info(f"  Average Abstract Concepts: {vocab['avg_abstract_concepts']:.1f}")
                logger.info(f"  Average Word Length: {vocab['avg_word_length']:.1f}")
                
                logger.info("\nLogical Features:")
                logical = complexity['logical_features']
                logger.info(f"  Causal Relations: {logical['pct_causal']:.1f}%")
                logger.info(f"  Comparisons: {logical['pct_comparison']:.1f}%")
                logger.info(f"  Conditionals: {logical['pct_conditional']:.1f}%")
                
                # Duration distribution
                if 'duration_stats' in results:
                    logger.info("\nDuration Distribution:")
                    for range_label, count in results['duration_stats']['distribution'].items():
                        logger.info(f"  {range_label}: {count} recordings")
            
            if args.output:
                logger.info(f"\nGenerating detailed report...")
                try:
                    report = analytics.generate_report(config.dataset.metadata_file, args.output)
                    logger.info(f"Report saved to {args.output}")
                except Exception as e:
                    logger.error(f"Failed to generate report: {e}")
                    return 1
                
            if not (args.quality or args.coverage):
                logger.error("Please specify --quality and/or --coverage with --analyze")
                return 1
        else:
            # Record audio
            recorder = ThroatMicRecorder(config)
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