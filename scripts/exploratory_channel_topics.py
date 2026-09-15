"""
Exploratory Analysis - Channel Topics
Date: July 20, 2026
Purpose: look at similarity in channel topics for T/C users in periods A,B,C


Channel overlap / diversity exploration across experimental periods A, B, C.

A and B are sequential pre-intervention periods (12 days each).
C is the post-intervention period (12 days).

Two levels of analysis, for both:
  - unique channel counts / watch diversity within a period
  - channel-set overlap (Jaccard / overlap coefficient) BETWEEN periods

Two units of analysis:
  - within-subject: does an individual user's metric change across periods?
  - within-group: does the treatment/control group average change across periods,
    and does the pre->post CHANGE differ between treatment and control?

NOTE ON ASSUMPTIONS:
This script assumes you have a separate file mapping user_id -> group
('treatment' / 'control'). Update GROUP_ASSIGNMENT_PATH and, if your columns
are named differently, the `group_df` column-rename line below. If you don't
have such a file yet, build one first (e.g. from your survey/randomization
records) with columns: user_id, group.
"""

from collections import Counter
import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
period_to_history_dirs = {
    'A': '../data/pre_histories_a/',
    'B': '../data/pre_histories_b/',
    'C': '../data/post_histories_c/',
}
PERIOD_ORDER = ['A', 'B', 'C']

# CSV with columns: user_id, group  (group values e.g. "treatment" / "control")
GROUP_ASSIGNMENT_PATH = '../data/survey2.csv'

USE_CATEGORY_IDS = False
if USE_CATEGORY_IDS:
    OUTPUT_DIR = '/Users/[ANONYMIZED]/yt_rec_channel_overlap_ytapichannelids'
else:
    OUTPUT_DIR = '/Users/[ANONYMIZED]/yt_rec_channel_overlap'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
GROUP_PALETTE = {'treatment': '#ff7f0e', 'control': '#1f77b4'}

SHOW_PLOT = False

def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"saved {path}")

PLAYLIST_VIDEO_IDS = [
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

# ---------------------------------------------------------------------------
# 1A. GROUP ASSIGNMENT (treatment / control) -- load in group df
# ---------------------------------------------------------------------------
group_df = pd.read_csv(GROUP_ASSIGNMENT_PATH)
group_df['user_id'] = group_df['PROLIFIC_PID'].astype(str)
group_df['group'] = np.where(group_df['DELETE_CONTROL'] == 'Yes', 'control', 'treatment')


# ---------------------------------------------------------------------------
# 2. LOAD RAW WATCH HISTORIES -> long dataframe (user_id, period, channel_id)
# ---------------------------------------------------------------------------
records = []
for period_key, period_dir in period_to_history_dirs.items():
    for fn in os.listdir(period_dir):
        if not fn.endswith('.csv'):
            continue
        user_id = Path(fn).stem
        user_df = pd.read_csv(os.path.join(period_dir, fn), dtype=str)
        # init column for video id
        user_df['video_id'] = (
            user_df['link']
            .astype(str)
            .str.split('watch?v=', regex=False).str[1]
            .str.split('&', regex=False).str[0]
        )
        # Important: we filter out all intervention videos from the user df if the user is treated and in period c
        group = group_df[group_df['user_id'] == user_id]['group'].tolist()[0]
        if group == 'treatment' and period_key == 'C':
            # remove all rows where vide id is in the playlist
            user_df = user_df[~user_df['video_id'].isin(PLAYLIST_VIDEO_IDS)]

        # NOTE: two different ways we can use category or channel ids
        if USE_CATEGORY_IDS:
            # OPTION 2 -- this uses the YT API scraped category id's
            user_channel_ids = user_df['categoryId'].dropna().astype(int).astype(str).to_list()
        else:
            # OPTION 1 -- this uses the actual unique channel id
            user_channels = user_df['channel_link'].dropna().to_list()
            # TODO: we might skip some videos without this pattern, decide on this
            user_channel_ids = [x.split('.com/channel/')[1] for x in user_channels if '.com/channel' in x]
        for ch in user_channel_ids:
            records.append({'user_id': user_id, 'period': period_key, 'channel_id': ch})

watch_long = pd.DataFrame(records)
watch_long['period'] = pd.Categorical(watch_long['period'], categories=PERIOD_ORDER, ordered=True)
watch_long['user_id'] = watch_long['user_id'].astype(str)

print(f"Loaded {len(watch_long)} channel-watch rows across "
      f"{watch_long['user_id'].nunique()} users and {watch_long['period'].nunique()} periods.")

# ---------------------------------------------------------------------------
# 1B. GROUP ASSIGNMENT (treatment / control) -- actually merge
# ---------------------------------------------------------------------------

watch_long = watch_long.merge(group_df[['user_id', 'group']], on='user_id', how='left')
missing_group = watch_long.loc[watch_long['group'].isna(), 'user_id'].unique()
if len(missing_group):
    print(f"WARNING: {len(missing_group)} users have no group assignment and will be "
          f"dropped from group-level comparisons: {list(missing_group)[:10]}")

# ---------------------------------------------------------------------------
# 3. PER-USER, PER-PERIOD SUMMARY  (within-subject unit of analysis)
# ---------------------------------------------------------------------------
def summarize(df):
    ch = df['channel_id']
    n_watches = len(ch)
    n_unique = ch.nunique()
    return pd.Series({
        'n_watches': n_watches,
        'n_unique_channels': n_unique,
        'diversity_ratio': n_unique / n_watches if n_watches else np.nan,
    })

user_period_summary = (
    watch_long
    .groupby(['user_id', 'group', 'period'], observed=True)
    .apply(summarize, include_groups=False)
    .reset_index()
)

# the SET of channels watched per user per period, needed for overlap calcs
user_period_sets = (
    watch_long.groupby(['user_id', 'period'], observed=True)['channel_id']
    .apply(set)
    .reset_index()
    .rename(columns={'channel_id': 'channel_set'})
)

print("\n--- Per-user per-period summary (head) ---")
print(user_period_summary.head())

# ===========================================================================
# QUESTION 1: within-subject — does an individual's # unique channels change
#             across A -> B -> C, and does that differ by group?
# ===========================================================================

# --- Viz 1a: spaghetti plot, one line per user, faceted by group ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, grp in zip(axes, ['control', 'treatment']):
    sub = user_period_summary[user_period_summary['group'] == grp]
    for uid, u_df in sub.groupby('user_id'):
        u_df = u_df.sort_values('period')
        ax.plot(u_df['period'], u_df['n_unique_channels'], color=GROUP_PALETTE[grp],
                alpha=0.25, marker='o', markersize=3, linewidth=1)
    grp_mean = sub.groupby('period', observed=True)['n_unique_channels'].mean().reindex(PERIOD_ORDER)
    print(f'Plot 1A: grp_mean: ({grp}): {grp_mean}')
    ax.plot(grp_mean.index, grp_mean.values, color='black', marker='o',
             linewidth=3, label='group mean')
    ax.set_title(grp.capitalize())
    ax.set_xlabel('Period')
    ax.legend()
axes[0].set_ylabel('# unique channels watched')
# axes[0].set_yscale('log')
fig.suptitle('Within-subject trajectories: unique channels per period')
fig.tight_layout()
savefig(fig, '1a_spaghetti_unique_channels.png')
plt.show()

# --- Viz 1b: same idea for diversity ratio (unique / total watches) -------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, grp in zip(axes, ['control', 'treatment']):
    sub = user_period_summary[user_period_summary['group'] == grp]
    for uid, u_df in sub.groupby('user_id'):
        u_df = u_df.sort_values('period')
        ax.plot(u_df['period'], u_df['diversity_ratio'], color=GROUP_PALETTE[grp],
                alpha=0.25, marker='o', markersize=3, linewidth=1)
    grp_mean = sub.groupby('period', observed=True)['diversity_ratio'].mean().reindex(PERIOD_ORDER)
    print(f'Plot 1B: grp_mean: ({grp}): {grp_mean}')
    ax.plot(grp_mean.index, grp_mean.values, color='black', marker='o',
             linewidth=3, label='group mean')
    ax.set_title(grp.capitalize())
    ax.set_xlabel('Period')
    ax.legend()
axes[0].set_ylabel('diversity ratio (unique / total watches)')
fig.suptitle('Within-subject trajectories: channel diversity ratio per period')
fig.tight_layout()
savefig(fig, '1b_spaghetti_diversity_ratio.png')
plt.show()

# --- Stat: paired comparisons within each group across period pairs -------
print("\n--- Within-subject paired tests (Wilcoxon signed-rank), n_unique_channels ---")
for grp in ['control', 'treatment']:
    wide = (
        user_period_summary[user_period_summary['group'] == grp]
        .pivot(index='user_id', columns='period', values='n_unique_channels')
    )
    for p1, p2 in itertools.combinations(PERIOD_ORDER, 2):
        paired = wide[[p1, p2]].dropna()
        if len(paired) < 3:
            continue
        stat, p = stats.wilcoxon(paired[p1], paired[p2])
        print(f"  [{grp}] {p1} vs {p2}: n={len(paired)}, "
              f"mean diff={paired[p2].mean()-paired[p1].mean():+.2f}, "
              f"Wilcoxon p={p:.4f}")

# ===========================================================================
# QUESTION 2: within-group — does the GROUP AVERAGE # unique channels change
#             across periods, and does the treatment group change differently
#             from control (the actual intervention-effect question)?
# ===========================================================================

# --- Viz 2a: boxplot of unique channels by period x group -----------------
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=user_period_summary, x='period', y='n_unique_channels',
            hue='group', order=PERIOD_ORDER, palette=GROUP_PALETTE, ax=ax)
sns.stripplot(data=user_period_summary, x='period', y='n_unique_channels',
              hue='group', order=PERIOD_ORDER, dodge=True, alpha=0.4,
              palette=GROUP_PALETTE, ax=ax, legend=False, size=4)
ax.set_title('Unique channels watched per period, by group')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title='group')
fig.tight_layout()
savefig(fig, '2a_boxplot_unique_channels_by_group.png')
plt.show()

# --- Viz 2b: pre (avg of A,B) vs post (C) group-level change, per user -----
pivot_counts = user_period_summary.pivot(index='user_id', columns='period',
                                          values='n_unique_channels')
pivot_counts = pivot_counts.merge(group_df[['user_id', 'group']], on='user_id', how='left') \
    if 'user_id' not in pivot_counts.index.names else pivot_counts
change_df = user_period_summary.pivot_table(index=['user_id', 'group'], columns='period',
                                             values='n_unique_channels').reset_index()
change_df['pre_mean'] = change_df[['A', 'B']].mean(axis=1)
change_df['change_post_minus_pre'] = change_df['C'] - change_df['pre_mean']

fig, ax = plt.subplots(figsize=(7, 6))
sns.stripplot(data=change_df, x='group', y='change_post_minus_pre', hue='group',
              order=['control', 'treatment'], palette=GROUP_PALETTE, alpha=0.6,
              size=6, jitter=0.15, legend=False, ax=ax)
sns.pointplot(data=change_df, x='group', y='change_post_minus_pre', order=['control', 'treatment'],
              color='black', markers='D', linestyle='none', errorbar='ci', ax=ax)
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_ylabel('C − mean(A,B): change in # unique channels')
ax.set_title('Pre→post change in unique channels, by group')
fig.tight_layout()
savefig(fig, '2b_prepost_change_by_group.png')
plt.show()

# --- Stat: is the pre->post change different between treatment & control? -
c_change = change_df.loc[change_df['group'] == 'control', 'change_post_minus_pre'].dropna()
t_change = change_df.loc[change_df['group'] == 'treatment', 'change_post_minus_pre'].dropna()
if len(c_change) > 1 and len(t_change) > 1:
    u_stat, p_val = stats.mannwhitneyu(t_change, c_change, alternative='two-sided')
    print(f"\nGroup-level pre->post change comparison (Mann-Whitney U): "
          f"treatment mean={t_change.mean():+.2f}, control mean={c_change.mean():+.2f}, p={p_val:.4f}")

# ===========================================================================
# QUESTION 3: channel-SET overlap between periods (Jaccard / overlap coef)
#             — within-subject, then summarized within-group
# ===========================================================================

def jaccard(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)

def overlap_coef(a, b):
    if not a or not b:
        return np.nan
    return len(a & b) / min(len(a), len(b))

sets_wide = user_period_sets.pivot(index='user_id', columns='period', values='channel_set')

pair_records = []
for user_id, row in sets_wide.iterrows():
    for p1, p2 in itertools.combinations(PERIOD_ORDER, 2):
        s1, s2 = row.get(p1), row.get(p2)
        if not isinstance(s1, set) or not isinstance(s2, set):
            continue
        pair_records.append({
            'user_id': user_id,
            'period_pair': f'{p1}-{p2}',
            'jaccard': jaccard(s1, s2),
            'overlap_coef': overlap_coef(s1, s2),
        })

overlap_df = pd.DataFrame(pair_records).merge(group_df[['user_id', 'group']],
                                               on='user_id', how='left')
PAIR_ORDER = ['A-B', 'B-C', 'A-C']
overlap_df['period_pair'] = pd.Categorical(overlap_df['period_pair'],
                                            categories=PAIR_ORDER, ordered=True)

print("\n--- Per-user-pair overlap summary (head) ---")
print(overlap_df.head())

# --- Viz 3a: boxplot of Jaccard overlap by period-pair x group -------------
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=overlap_df, x='period_pair', y='jaccard', hue='group',
            order=PAIR_ORDER, palette=GROUP_PALETTE, ax=ax)
sns.stripplot(data=overlap_df, x='period_pair', y='jaccard', hue='group',
              order=PAIR_ORDER, dodge=True, alpha=0.4, palette=GROUP_PALETTE,
              ax=ax, legend=False, size=4)
ax.set_title('Channel-Set Jaccard Overlap Between Periods')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title='Group')
# Change axis names
ax.set_xlabel('Period Pair')
ax.set_ylabel('Jaccard Similarity')
fig.tight_layout()
savefig(fig, '3a_boxplot_jaccard_by_group.png')
plt.show()

# --- Viz 3b: heatmap, mean Jaccard per group x period-pair -----------------
heat = overlap_df.groupby(['group', 'period_pair'], observed=True)['jaccard'].mean().unstack()
heat = heat.reindex(columns=PAIR_ORDER)
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(heat, annot=True, fmt='.2f', cmap='viridis', ax=ax, cbar_kws={'label': 'mean Jaccard'})
ax.set_title('Mean channel-set overlap by group & period pair')
fig.tight_layout()
savefig(fig, '3b_heatmap_mean_jaccard.png')
plt.show()

# --- Stat: does overlap DROP more across the intervention (B-C) than
#           it does in the baseline (A-B), and does that drop differ by group?
print("\n--- Does overlap change more across the intervention than pre-period baseline? ---")
for grp in ['control', 'treatment']:
    wide_overlap = (
        overlap_df[overlap_df['group'] == grp]
        .pivot(index='user_id', columns='period_pair', values='jaccard')
    )
    paired = wide_overlap[['A-B', 'B-C']].dropna()
    if len(paired) < 3:
        continue
    stat, p = stats.wilcoxon(paired['A-B'], paired['B-C'])
    print(f"  [{grp}] A-B vs B-C Jaccard: n={len(paired)}, "
          f"mean A-B={paired['A-B'].mean():.3f}, mean B-C={paired['B-C'].mean():.3f}, "
          f"Wilcoxon p={p:.4f}")

# between-group: is the intervention-spanning overlap (B-C) itself different
# between treatment and control?
bc = overlap_df[overlap_df['period_pair'] == 'B-C']
c_bc = bc.loc[bc['group'] == 'control', 'jaccard'].dropna()
t_bc = bc.loc[bc['group'] == 'treatment', 'jaccard'].dropna()
if len(c_bc) > 1 and len(t_bc) > 1:
    u_stat, p_val = stats.mannwhitneyu(t_bc, c_bc, alternative='two-sided')
    print(f"\nB-C Jaccard, treatment vs control (Mann-Whitney U): "
          f"treatment mean={t_bc.mean():.3f}, control mean={c_bc.mean():.3f}, p={p_val:.4f}")


###
# combined: pairwise jaccard mean/var AND most-in-common channels, per group/period
###

# user_period_sets already has one row per (user_id, period) with a 'channel_set' column;
# merge in group so we can filter by it
sets_with_group = user_period_sets.merge(group_df[['user_id', 'group']], on='user_id', how='left')

between_user_records = []
channel_commonality_dfs = {}

for period in PERIOD_ORDER:
    for group in ['control', 'treatment']:
        subset = sets_with_group[
            (sets_with_group['period'] == period) & (sets_with_group['group'] == group)
        ]
        user_sets = subset['channel_set'].to_list()
        n_users = len(user_sets)

        group_period_jaccards = []
        channel_X = Counter()  # X: times channel appears in an intersection across all pairs
        channel_Y = Counter()  # Y: times channel appears in a union across all pairs

        for set_a, set_b in itertools.combinations(user_sets, 2):
            intersection = set_a & set_b
            union = set_a | set_b

            combo_jaccard = len(intersection) / len(union) if union else np.nan
            if not np.isnan(combo_jaccard):
                group_period_jaccards.append(combo_jaccard)

            for ch in union:
                channel_Y[ch] += 1
            for ch in intersection:
                channel_X[ch] += 1

        # --- mean/var summary ---
        group_mean = np.mean(group_period_jaccards) if group_period_jaccards else np.nan
        group_var = np.var(group_period_jaccards, ddof=1) if len(group_period_jaccards) > 1 else np.nan
        print(f'{group}, {period}, n_users={n_users}, '
              f'n_pairs={len(group_period_jaccards)}, mean={group_mean:.4f}, var={group_var:.4f}')
        between_user_records.append({
            'group': group, 'period': period, 'n_users': n_users,
            'n_pairs': len(group_period_jaccards), 'mean_jaccard': group_mean, 'var_jaccard': group_var,
        })

        # --- most-in-common channels ---
        records = []
        for ch, y in channel_Y.items():
            x = channel_X.get(ch, 0)
            records.append({
                'group': group, 'period': period, 'channel_id': ch,
                'X_intersection_count': x, 'Y_union_count': y,
                'X_over_Y': x / y if y else np.nan,
            })
        cell_df = pd.DataFrame(records).sort_values(
            ['X_intersection_count', 'X_over_Y'], ascending=False
        ).reset_index(drop=True)
        channel_commonality_dfs[(group, period)] = cell_df

        print(f'--- Top 10 most-in-common channels: {group}, {period} ---')
        print(cell_df.head(10)[['channel_id', 'X_intersection_count', 'Y_union_count', 'X_over_Y']]
              .to_string(index=False))

between_user_jaccard_df = pd.DataFrame(between_user_records)
print(between_user_jaccard_df)
channel_commonality_df = pd.concat(
    [df.assign(group=g, period=p) for (g, p), df in channel_commonality_dfs.items()],
    ignore_index=True
)


print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}")

print('done')