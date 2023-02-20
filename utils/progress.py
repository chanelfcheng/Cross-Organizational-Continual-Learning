import sys
from datetime import datetime

def progress_bar(i, max_iter, rounds, task_number, loss):
    if not (i + 1) % 10 or (i + 1) == max_iter:
        progress = min(float((i + 1) / max_iter), 1)
        progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
        print('\r[ {} ] Task {} | round {}: |{}| loss: {:.2e}'.format(
            datetime.now().strftime("%m-%d | %H:%M"),
            task_number + 1 if isinstance(task_number, int) else task_number,
            rounds,
            progress_bar,
            loss
        ), file=sys.stderr, end='', flush=True)