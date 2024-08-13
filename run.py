import os
import subprocess


def find_dir_name(dir, name):
    for dir_name in os.listdir(dir):
        if name in dir_name and dir_name.endswith('.pth'):
            return os.path.join(dir, dir_name)

save_root = 'other_result/width25'

t1 = (
    ['--config-file', 'configs/t1/t1_train.yaml',
     "--num-gpus", "2",
    'OUTPUT_DIR', f'{save_root}/t1',   
    "OWOD.OBJECT_NESS_INFERENCE", "True"],
)
t2 = (
    ['--config-file', 'configs/t2/t2_train.yaml',
     "--num-gpus", "2",
    'OUTPUT_DIR', f'{save_root}/t2'],
    (f'{save_root}/t1', 'bast_Current class'),
)
t2_ft = ([
    '--config-file', 'configs/t2/t2_ft.yaml',
    "--num-gpus", "2",
    'OUTPUT_DIR', f'{save_root}/t2/ft'],
    (f'{save_root}/t2', 'bast_Current class')
)

t3 = ([
    '--config-file', 'configs/t3/t3_train.yaml',
    "--num-gpus", "2",
    'OUTPUT_DIR', f'{save_root}/t3'], 
    (f'{save_root}/t2/ft', 'bast_Current class')
)
t3_ft = ([
    '--config-file', 'configs/t3/t3_ft.yaml',
    "--num-gpus", "2",
    'OUTPUT_DIR', f'{save_root}/t3/ft'],
    (f'{save_root}/t3', 'bast_Current class')
)

t4 = ([
    '--config-file', '/home/xx/repeat/configs/t4/t4_train.yaml',
    "--num-gpus", "2",
    'OUTPUT_DIR', f'{save_root}/t4'],
    (f'{save_root}/t3/ft', 'bast_Prev class')
)

t4_ft = ([
    '--config-file', '/home/xx/repeat/configs/t4/t4_ft.yaml',
    "--num-gpus", "2",
    'OUTPUT_DIR', f'{save_root}/t4/ft'],
    (f'{save_root}/t4', 'bast_Current class')
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
for task_i, task in enumerate([t3, t3_ft, t4, t4_ft]):
    # if task_i != 0:
    command = ['python', 'tools/train_net.py'] + task[0] + ['MODEL.WEIGHTS', find_dir_name(task[1][0], task[1][1])]
    # else:
    #     command = ['python', 'tools/train_net.py'] + task[0]
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        error = result.stderr
        print(f"error: {error}")
    else:
        print('success')