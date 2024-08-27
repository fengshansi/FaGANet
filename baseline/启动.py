from pathlib import Path
import subprocess
from os import listdir
from os.path import isfile, join
import sys
import time

real_dir = Path(__file__).resolve().parent


# 获取当前 Conda 环境中 Python 解释器的路径
python_executable = sys.executable

basic_param_list = [
    # "python",
    python_executable,
    real_dir / "main_baseline_raw.py",
]


# 循环添加额外参数并调用
for randomseed in [0, 1, 2, 3, 4]:

    extend_param_list = [
        "--TORCH_SEED",
        f"{randomseed}",
    ]
    current_param_list = basic_param_list + extend_param_list
    print(f"当前param list{current_param_list}", flush=True)
    env = {"CUDA_VISIBLE_DEVICES": f"{randomseed}"}

    output_file = real_dir / f"{randomseed}.txt"
    with open(output_file, "w") as f:
        p = subprocess.Popen(
            current_param_list,
            env=env,
            stdout=f,
        )

        # 必须要等待进程结束 不然会爆炸 很多个进程一起执行
        p.wait()
        # 返回结果如果不是0 则报错
        if p.returncode != 0:
            print("error")
            exit()
