from pathlib import Path
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count

# 获取当前 Conda 环境中 Python 解释器的路径
python_executable = sys.executable

real_dir = Path(__file__).resolve().parent

basic_param_list = [
    python_executable,
    real_dir / "main_baseline_raw.py",
]


# 用于执行单个任务的函数
def run_task(randomseed):
    extend_param_list = [
        "--TORCH_SEED",
        f"{randomseed}",
    ]
    current_param_list = basic_param_list + extend_param_list
    print(f"当前param list: {current_param_list}", flush=True)
    env = {"CUDA_VISIBLE_DEVICES": f"{randomseed+1}"}

    output_file = real_dir / f"{randomseed}.txt"
    with open(output_file, "w") as f:
        p = subprocess.Popen(
            current_param_list,
            env=env,
            stdout=f,
        )
        p.wait()

        if p.returncode != 0:
            print("error")
            exit()


# 设置最大进程数
max_processes = min(cpu_count(), 5)  # 这里的5可以替换为你想要的最大进程数
print(f"使用 {max_processes} 个进程进行任务执行")

# 创建进程池
with Pool(processes=max_processes) as pool:
    pool.map(run_task, [0, 1, 2, 3, 4])
