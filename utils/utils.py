# -*- coding: utf-8 -*-


# ---------- Print Tools ----------
def print_to_logfile(logfile, content, init=False, end='\n'):
    if init:
        with open(logfile, 'w') as f:
            f.write(content + end)
    else:
        with open(logfile, 'a') as f:
            f.write(content + end)


def print_to_console(content, colored=False):
    if not colored:
        print(content, flush=True)
    else:
        raise AssertionError('Not implemented yet')


