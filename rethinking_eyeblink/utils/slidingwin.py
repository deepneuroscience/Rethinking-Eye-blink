# Rethinking Eye-blink project

## Author(s): Dr. Youngjun Cho *(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com

# sliding window
def overlap_windows_rate(data, overlap_rate, window_size):
    window_list = []
    start = 0
    end = window_size
    remain_length = len(data)

    overlap = overlap_rate * window_size
    while remain_length >= window_size:
        window_list.append(data[int(round(start + 0.01)):int(round(end + 0.01))])
        start += overlap
        end += overlap
        remain_length -= overlap

    return window_list


def overlap_windows_sec(data, framerate, overlap_sec, window_size):
    window_list = []
    start = 0
    end = window_size
    remain_length = len(data)

    overlap = framerate * overlap_sec
    while remain_length >= window_size:
        window_list.append(data[int(round(start + 0.01)):int(round(end + 0.01))])
        start += overlap
        end += overlap
        remain_length -= overlap

    return window_list

