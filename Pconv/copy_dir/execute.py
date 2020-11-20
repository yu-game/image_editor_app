import os


def execute_pconv(config):
    ex = 'python ../Pconv/copy_dir/Pconv_te.py '+config
    print(ex)
    os.system('pwd')
    os.system(ex)

