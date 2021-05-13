import praw
from prawcore import exceptions
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pickle import dump
from pynput import keyboard
import re
import time


# output path
ts_begin = datetime.now()
output = Path(f'dataset {ts_begin.strftime("%Y-%m-%d %H-%M-%S")}')
output.mkdir(exist_ok=True)


print('Program started')
print(ts_begin)

reddit = praw.Reddit(client_id='2QN7gPlIYqgIfg',
                     client_secret='bOPKO537BgfDqIaABzTmSkqIG7o',
                     user_agent='windows:com.Alaa:v1.0.0 (by /u/alaa_137)')

subreddits = ['AskEurope', 'Europe']  # Tried 'EuropeanFederalists', useless.


key_pressed = False


def on_press(key):
    global key_pressed
    if key == keyboard.Key.esc:
        print('Stopped listening to keyboard')
        return False  # stop listener

    try:
        k = key.char  # single-char keys
    except AttributeError:
        k = key.name  # other keys

    if k == 'p':
        key_pressed = True  # store it in global-like variable
        return False  # stop listener


if __name__ == '__main__':

    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    listener.join()  # remove if main thread is polling self.keys

    # collect data from subreddits
    posts = defaultdict(list)
    for i,subreddit in enumerate(subreddits):
        if key_pressed:
            break
        print(f"Parsing subreddit '{subreddit}'...")
        for submission in reddit.subreddit(subreddit).top('all', limit=None):
            if key_pressed:
                break
            if submission.author_flair_text is not None and submission.author_flair_text.strip():
                country = submission.author_flair_text.split(': ')[-1]
                if re.match('^\(?[A-Za-z\s/]+\)?', country):
                    if submission.title != '[removed]':
                        posts[country].append(submission.title)
                    if submission.selftext and submission.selftext != '[removed]':
                        posts[country].append(submission.selftext)

                    # parse comments and sub comments using bfs
                    try:
                        submission.comments.replace_more(limit=None)
                        comment_queue = submission.comments[:]
                        while comment_queue:
                            comment = comment_queue.pop(0)
                            if comment.body != '[removed]':
                                posts[country].append(comment.body)
                                comment_queue.extend(comment.replies)
                    except exceptions.ServerError:
                        print('ServerError')
                        time.sleep(2)

        # save checkpoint if this is not the last subreddit
        if i < len(subreddits) - 1:
            with open(output / f'checkpoint_{i}_data_dict.pkl', mode='wb') as f:
                dump(posts, f, -1)
                print(f'Checkpoint {i} saved.')
                print(datetime.now())

    if key_pressed:
        print('Cancelled by key press')
        print(datetime.now())

    # save the whole collection
    with open(output / f'data_dict.pkl', mode='wb') as f:
        dump(posts, f, -1)

    # save to files, deal with errors
    failed = []
    for k,v in posts.items():
        print(f'{len(v)} items in {k}')
        try:
            with open(output / f"{k.replace('/', '-')}.txt", mode='w') as f:
                for line in v:
                    f.write(f'{line}\n')
        except OSError:
            failed.append(k)

    if failed:
        print('\nCould not create files:')
        for f in failed:
            print(f)

    print('')
    print(f'Collected {len(posts)} countries, {sum([len(v) for v in posts.values()])} posts / comments.')
    print(datetime.now())
    print(datetime.now() - ts_begin)
