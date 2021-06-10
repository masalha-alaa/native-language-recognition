"""
Fetch data from reddit.com
"""

import praw
from prawcore import exceptions
from collections import defaultdict
from datetime import datetime
from pickle import dump
import re
import time
import argparse
from paths import *


def extract_country(flair):
    """
    Extract country from user flair.
    """
    if flair is not None and flair.strip():
        country = flair.split(': ')[-1]
        return country if re.match('[A-Za-z]+', country) else None
    return None


def save_checkpoint(path, obj, filename=None):
    """
    Save checkpoint of object to path.
    """
    if filename is None:
        path = path / f'checkpoint {datetime.now().strftime(DATE_STR_LONG)} data{PKL_LST_EXT}'
    else:
        path = path / filename
    with open(path, mode='wb') as f:
        dump(obj, f, -1)

    return datetime.now()


if __name__ == '__main__':
    SAVE_FREQ_HRS = 2

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--txt", required=False, action='store_false',
                    help="Save as txt files (default: %(default)s).")
    ap.add_argument("-d", "--id", required=True, help="Reddit client ID")
    ap.add_argument("-s", "--secret", required=True, help="Reddit client secret")
    ap.add_argument("-a", "--agent", required=True, help="User agent")
    args = vars(ap.parse_args())
    save_txt = args['txt']
    client_id = args['id']
    client_secret = args['secret']
    user_agent = args['agent']

    ts_begin = datetime.now()
    ts_checkpoint = ts_begin
    print_log(f'Program started\n{ts_begin}')

    # output path
    main_output = ROOT_DIR / f'dataset {ts_begin.strftime(DATE_STR_SHORT)}'
    raw_output = main_output / RAW_DIR_NAME
    main_output.mkdir(exist_ok=True)
    raw_output.mkdir(exist_ok=True)

    log_file = main_output / f'log {ts_begin.strftime(DATE_STR_SHORT)}{LOG_EXT}'
    print_log(f'Program started\n{ts_begin}', log_file, to_console=False)

    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)

    # subreddits = ['AskEurope', 'Europe']
    subreddits = ['EuropeanCulture', 'EuropeanFederalists', 'Eurosceptics']

    # collect data from subreddits
    posts = defaultdict(list)
    for i,subreddit in enumerate(subreddits):
        print_log(f"Parsing subreddit '{subreddit}'...", log_file)
        for submission in reddit.subreddit(subreddit).top('all', limit=None):
            if (datetime.now() - ts_checkpoint).total_seconds() / 3600 > SAVE_FREQ_HRS:
                ts_checkpoint = save_checkpoint(main_output, posts)
                print_log(f'Checkpoint saved.\n{ts_checkpoint}', log_file)
            country = extract_country(submission.author_flair_text)
            if country:
                if submission.title not in ['[removed]', '[deleted]']:
                    posts[country].append(submission.title)
                if submission.selftext and submission.selftext not in ['[removed]', '[deleted]']:
                    posts[country].append(submission.selftext)

            # parse comments and sub comments using bfs
            network_error = True
            attempts = 10
            while network_error and attempts:
                try:
                    submission.comments.replace_more(limit=None)
                    network_error = False
                except (exceptions.ServerError, exceptions.RequestException) as e:
                    print_log(f'Network Error:\n{str(e)}Retrying...', log_file)
                    time.sleep(5)
                    attempts -= 1
                    continue
                comment_queue = submission.comments[:]
                while comment_queue:
                    comment = comment_queue.pop(0)
                    country = extract_country(comment.author_flair_text)
                    if country and comment.body not in ['[removed]', '[deleted]']:
                        posts[country].append(comment.body)
                        comment_queue.extend(comment.replies)

        # save checkpoint if this is not the last subreddit
        if i < len(subreddits) - 1:
            ts_checkpoint = save_checkpoint(main_output, posts)
            print_log(f'\nCheckpoint saved.\n{ts_checkpoint}\n', log_file)

    # save the whole collection
    save_checkpoint(main_output, posts, f'data{PKL_DIC_EXT}')

    # save to files, deal with errors
    print_log('\nSaving to files...', log_file)
    if save_txt:
        saving_report = ''
        failed = []
        for k,v in posts.items():
            saving_report += f'{len(v)} items in {k}\n'
            try:
                with open(raw_output / f"{k.replace('/', '-')}.txt", encoding='utf-8', mode='a') as f:
                    # TODO: Can this be replaced with f.writelines ?
                    for line in v:
                        f.write(f'{line}\n')
            except OSError:
                failed.append(k)
        print_log(saving_report, log_file)

        if failed:
            failed_report = '\nCould not create files for:'
            for f in failed:
                failed_report += f'\n{f}'
            print_log(failed_report, log_file)

    print_log(f'\nCollected {len(posts)} countries, {sum([len(v) for v in posts.values()])} posts / comments.\n'
              f'{datetime.now()}\n'
              f'TOTAL TIME: {datetime.now() - ts_begin}', log_file)

    with open(main_output / 'README.md', mode='w') as f:
        f.write('Fetched subreddits:\n\n')
        for subreddit in subreddits:
            f.write(f'{subreddit}\n')
