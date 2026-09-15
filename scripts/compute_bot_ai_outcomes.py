import argparse
import re
from pathlib import Path

import pandas as pd


AI_KEYWORDS = [
    'ai',
    'claude',
    'sora',
    'gemini',
    'mythos',
    'copilot',
    'codex',
    'grok',
    'llama',
    'chatgpt',
    'openai',
    'midjourney',
    'elevenlabs',
    'cursor',
    'perplexity',
    'deepseek',
    'llm',
    'gpt',
    'agi',
]

AI_KEYPHRASES = [
    'artificial intelligence',
    'vibe cod',
    'eleven labs',
    'mid journey',
]


def build_ai_title_pattern():
    terms = AI_KEYWORDS + AI_KEYPHRASES
    return re.compile(r'\b(?:' + '|'.join(map(re.escape, terms)) + r')\b', re.IGNORECASE)


def compute_outcomes(input_dir):
    ai_title_pattern = build_ai_title_pattern()
    results = []

    for csv_path in sorted(input_dir.rglob('*.csv')):
        data = pd.read_csv(csv_path)
        if 'title' not in data.columns:
            raise ValueError(f'{csv_path} does not contain a title column')

        titles = data['title'].fillna('').astype(str)
        ai_count = int(titles.str.contains(ai_title_pattern, regex=True).sum())
        total_videos = int(len(data))

        results.append(
            {
                'file': str(csv_path.relative_to(input_dir)),
                'ai_count': ai_count,
                'total_videos': total_videos,
                'proportion_ai': ai_count / total_videos if total_videos else float('nan'),
            }
        )

    return pd.DataFrame(
        results,
        columns=['file', 'ai_count', 'total_videos', 'proportion_ai'],
    )


def main():
    parser = argparse.ArgumentParser(
        description='Count AI-related video titles in each bot-simulation CSV.'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('./data/bot_simulations'),
        help='Folder containing CSV files (searched recursively).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./data/bot_ai_outcomes.csv'),
        help='Output CSV path.',
    )
    args = parser.parse_args()

    results = compute_outcomes(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f'Wrote {len(results)} rows to {args.output}')


if __name__ == '__main__':
    main()