#!/usr/bin/env python3
import argparse
from dataset_analytics import AudioQualityControl, DatasetAnalytics, analyze_recording_quality
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze throat mic dataset quality and coverage")
    parser.add_argument("--quality", action="store_true",
                      help="Run audio quality analysis")
    parser.add_argument("--coverage", action="store_true",
                      help="Analyze dataset coverage (vocabulary, phonetics)")
    parser.add_argument("--report", action="store_true",
                      help="Generate a full report")
    parser.add_argument("--output", type=str,
                      help="Output file for results (default: prints to console)")
    
    args = parser.parse_args()
    
    if not any([args.quality, args.coverage, args.report]):
        parser.print_help()
        return
    
    try:
        if args.quality:
            print("\nAnalyzing audio quality...")
            results = analyze_recording_quality("throatmic_data", 
                                             args.output and f"{args.output}_quality.json")
            print(f"\nAnalyzed {results['total_recordings']} recordings")
            print(f"Average quality score: {results['average_quality_score']:.1f}/100")
            print(f"Recordings with issues: {results['recordings_with_issues']}")
            if results['common_issues']:
                print("\nMost common issues:")
                for issue, count in results['common_issues']:
                    print(f"  {issue}: {count} recordings")
        
        if args.coverage or args.report:
            print("\nAnalyzing dataset coverage...")
            analytics = DatasetAnalytics()
            report = analytics.generate_report("metadata.csv", 
                                            args.output and f"{args.output}_coverage.txt")
            print("\n" + report)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 