import sys, time, threading


start_time = time.time()
progress_lock = threading.Lock()


def progress_bar(iteration: int, total: int, message = 'Compiling...', length=20):
    global start_time

    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (total - iteration) / iteration if iteration > 0 else 0
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

    percent = int(100 * iteration/total)
    filled_length = length * iteration // total
    bar = '#' * filled_length + '-' * (length - filled_length)+ f" {percent}% {iteration}/{total}"
    #sys.stdout.write('\r\033[K')
    #sys.stdout.flush()
    sys.stdout.write(f'\r[{bar}] | Elapsed: {elapsed_str} | ETA: {remaining_str} | {message}')
    sys.stdout.write(' ' * 20)
    sys.stdout.flush()


def nested_progress_bar(outer_iter: int, inner_iter: int, outer_total: int, inner_total: int, message='Compiling...', length=20):
    global start_time
    
    elapsed_time = time.time() - start_time
    progress_ratio = inner_total*(outer_total+1) / (outer_iter*inner_total + inner_iter)
    remaining_time = elapsed_time*(progress_ratio-1)
    #progress_ratio = (outer_iter - 1 + inner_iter / inner_total) / outer_total if outer_total > 0 else 0
    #remaining_time = (elapsed_time / progress_ratio) - elapsed_time if progress_ratio > 0 else 0
    
    elapsed_str = f"{int(elapsed_time//86400):02}:{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}"
    remaining_str = f"{int(remaining_time//86400):02}:{time.strftime("%H:%M:%S", time.gmtime(remaining_time))}" if progress_ratio > 0 else "--:--:--:--"

    outer_percent = int(100 * outer_iter / outer_total)
    inner_percent = int(100 * inner_iter / inner_total)
    outer_filled = length * outer_iter // outer_total
    inner_filled = length * inner_iter // inner_total

    outer_bar = '#' * outer_filled + '-' * (length - outer_filled) + f" {outer_percent}% {outer_iter}/{outer_total}"
    inner_bar = '#' * inner_filled + '-' * (length - inner_filled) + f" {inner_percent}% {inner_iter}/{inner_total}"

    with progress_lock:
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
        sys.stdout.write(f'\r[{outer_bar}] [{inner_bar}] | Elapsed: {elapsed_str} | ETA: {remaining_str} | {message}')
        sys.stdout.flush()
