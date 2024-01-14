import argparse
from src.multiframe.mf_analyzer import MFAnalyzer

def _get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cametra_path', action='store',
                        default=None, help='path to OD camera file, .tsv')
    parser.add_argument('--output_dir', action='store',required=False,
                        default=None, help='path to output directory')
    return parser.parse_args()

def run(args):
    mfa = MFAnalyzer(args.cametra_path, output_dir=args.output_dir)
    mfa.save_tracklets_with_physical_anomalies()
    mfa.save_tracklets_with_derivatives_anomalies()
    

if __name__ == "__main__":
    args = _get_parameters()
    run(args)