#!/usr/bin/env python3
"""
Throat microphone recording tool.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src directory to path
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src import ThroatMicRecorder, AudioQualityControl, DatasetAnalytics, ThroatMicError, config

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
            # Run analysis
            if args.quality:
                logger.info("Analyzing audio quality...")
                analyzer = AudioQualityControl()
                results = analyzer.analyze_all_recordings()
                logger.info(f"Analyzed {results['total_recordings']} recordings")
                logger.info(f"Average quality score: {results['average_quality_score']:.1f}/100")
                if results['common_issues']:
                    logger.info("Most common issues:")
                    for issue, count in results['common_issues']:
                        logger.info(f"  {issue}: {count} recordings")
            
            if args.coverage:
                logger.info("Analyzing dataset coverage...")
                analytics = DatasetAnalytics()
                report = analytics.generate_report()
                logger.info("\n" + report)
                
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