"""Analyze stored independent initialization trajectories; no ODE reruns here."""
import argparse
from .metrics import summarize_trajectories, summarize_endpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='results/raw/trajectories.csv')
    parser.add_argument('--output', default='results/summary.csv')
    parser.add_argument('--endpoints', default='results/endpoints.csv')
    args = parser.parse_args()
    summary = summarize_trajectories(args.raw, args.output)
    endpoints = summarize_endpoints(args.raw, args.endpoints)
    print(f'Saved {len(summary)} curve summaries to {args.output}')
    print(f'Saved {len(endpoints)} endpoint summaries to {args.endpoints}')
    print(endpoints[['beta','student_width','mean_final_risk','rank_floor',
                     'mean_final_gap','mcse_final_gap']].to_string(index=False))


if __name__ == '__main__':
    main()
