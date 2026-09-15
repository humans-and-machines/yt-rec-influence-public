import csv
from datetime import datetime
import json
import os
import re
import zipfile
import pandas as pd
from tqdm import tqdm


def main():
    # read in csv from S2 so we know who is treatment/control
    s2_fn = './data/survey2.csv'
    df_s2 = pd.read_csv(s2_fn)
    ids_t = df_s2[df_s2['DELETE_TREATMENT'] == 'Yes']['PROLIFIC_PID'].to_list()
    ids_c = df_s2[df_s2['DELETE_CONTROL'] == 'Yes']['PROLIFIC_PID'].to_list()

    # init dict of watch histories
    watch_histories = {}

    # check all uploads
    uploads_dirs = [
        './data/pre_histories_a/',
        # './data/pre_histories_b/',
        # './data/post_histories_c/'
    ]
    output_fn = './data/pre_history_a_results.csv'

    for uploads_dir in uploads_dirs:
        fns = os.listdir(uploads_dir)
        for fn in tqdm(fns):
            fp = os.path.join(uploads_dir, fn)
            if not fp.endswith('.csv'):
                continue
            
            # extract participant id from filename
            # pattern = r"(R_[A-Za-z0-9]+)"
            pattern = r"([A-Za-z0-9]+)"
            participant_id = re.search(pattern, fn).group(1)
            # read in data
            data = pd.read_csv(fp)
            # save data
            watch_histories[participant_id] = data

    # Single word tokens — matched as whole words (handles AI's, AI-, #AI, etc.)
    ai_keywords = [
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
        'agi'
    ]

    # Multi-word phrases — matched as substrings (word-boundary anchored)
    ai_keyphrases = [
        'artificial intelligence',
        'vibe cod',  # catches "vibe code", "vibe coding", "vibe coder"
        'eleven labs',
        'mid journey',
    ]

    playlist_ids = [
        '9fogzZ_6kR0',
        'HbiyTZae61A',
        '6H17LkJB7U4',
        'Tx0uoiuLAZg',
        'rfIhRjFOpuc',
        '0kIhwa1e_5M',
        'EEw-7zVs8wE',
        'Khf6o0t-9zg',
        'Wd8RzZ3xn7w',
        'h89iKqL1bRg',
        'uoU_1KORocQ',
        'q6yq1i7vd9E',
        'vz3HKkVrJE4',
        'iWl7OMxZSUU',
        'A3S8yzJ4oBM',
        '7rEzhCwPhqc',
        'jimldQT1pQw',
        'mEahc3UbSq4',
        'QF-Vu93C4m0',
        '5sQeivm12x0',
        'Fyzr6jbSI_I'
    ]

    # look at titles
    results = []
    for participant_id, df in watch_histories.items():
        # get indicator for treatment or control
        is_t = participant_id in ids_t
        is_c = participant_id in ids_c

        # get playlist vids
        playlist_pattern = '|'.join(playlist_ids)
        df_playlist = df[df['link'].str.contains(playlist_pattern, case=False, na=False)]

        # get ai vids
        keyword_pattern = r'\b(' + '|'.join(map(re.escape, ai_keywords)) + r')\b'
        # Phrase match: whole-word anchored on both ends
        keyphrase_pattern = r'\b(' + '|'.join(map(re.escape, ai_keyphrases)) + r')\b'
        df_ai = df[
            df['title'].str.contains(keyphrase_pattern, case=False, na=False) |
            df['title'].str.contains(keyword_pattern, case=False, na=False)
            ]

        assignement = 'Treatment' if is_t else 'Control' if is_c else 'N/A'

        
        df_ai_excluding_playlist = df_ai[
            ~df_ai['link'].str.contains(playlist_pattern, case=False, na=False)
        ]
        ai_count_excluding_playlist = df_ai_excluding_playlist.shape[0]
        total_adjusted_videos = int(df.shape[0] - df_playlist.shape[0])
        if total_adjusted_videos > 0:
            share_ai_adjusted_videos = ai_count_excluding_playlist / total_adjusted_videos
        else:
            share_ai_adjusted_videos = float('nan')

        watch_time = pd.to_numeric(df.get('watch_time'), errors='coerce')
        total_watch_time = watch_time.sum()
        playlist_watch_time = pd.to_numeric(
            df_playlist.get('watch_time'), errors='coerce'
        ).sum()
        total_adjusted_watch_time = total_watch_time - playlist_watch_time
        all_ai_watch_time = pd.to_numeric(
            df_ai.get('watch_time'), errors='coerce'
        ).sum()
        ai_watch_time = pd.to_numeric(
            df_ai_excluding_playlist.get('watch_time'), errors='coerce'
        ).sum()
        if total_adjusted_watch_time > 0:
            share_ai_adjusted_watch_time = ai_watch_time / total_adjusted_watch_time
        else:
            share_ai_adjusted_watch_time = float('nan')

        results.append(
            {
                'participant_id': participant_id,
                'assignment': assignement,
                'total_videos': int(df.shape[0]),
                'total_adjusted_videos': total_adjusted_videos,
                'playlist_videos': int(df_playlist.shape[0]),
                'all_ai_videos': int(df_ai.shape[0]),
                'ai_adjusted_videos': int(ai_count_excluding_playlist),
                'share_ai_adjusted_videos': share_ai_adjusted_videos,
                'total_watch_time': total_watch_time,
                'total_adjusted_watch_time': total_adjusted_watch_time,
                'playlist_watch_time': playlist_watch_time,
                'all_ai_watch_time': all_ai_watch_time,
                'ai_adjusted_watch_time': ai_watch_time,
                'share_ai_adjusted_watch_time': share_ai_adjusted_watch_time,
            }
        )

    output_columns = [
        'participant_id',
        'assignment',
        'total_videos',
        'total_adjusted_videos',
        'playlist_videos',
        'all_ai_videos',
        'ai_adjusted_videos',
        'share_ai_adjusted_videos',
        'total_watch_time',
        'total_adjusted_watch_time',
        'playlist_watch_time',
        'all_ai_watch_time',
        'ai_adjusted_watch_time',
        'share_ai_adjusted_watch_time',
    ]
    df_results = pd.DataFrame(results, columns=output_columns)
    if not df_results.empty:
        df_results = df_results.sort_values(by='participant_id')
    df_results.to_csv(output_fn, index=False)
    print(f'Wrote {df_results.shape[0]} rows to {output_fn}')
    print('done')


if __name__ == '__main__':
    main()
