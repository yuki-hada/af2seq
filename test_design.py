"""
最小テスト: GradientDesign を XLA キャッシュ + メモリ制限付きで実行

使い方:
    ~/.venvs/af2seq-arm64/bin/python test_design.py
"""

import os
import resource

# --- メモリ上限: カーネルパニック防止 (20GB でプロセスを止める) ---
MEM_LIMIT_GB = 20
try:
    limit = MEM_LIMIT_GB * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    print(f"[Guard] メモリ上限 {MEM_LIMIT_GB}GB 設定済み (kernel panic 防止)")
except Exception as e:
    print(f"[Guard] メモリ上限設定スキップ: {e}")

# --- JAX / XLA 設定 ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
XLA_CACHE = os.path.expanduser("~/.cache/jax_xla")
os.makedirs(XLA_CACHE, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", XLA_CACHE)
print(f"[XLA]  cache={XLA_CACHE}")
print(f"[JAX]  version={jax.__version__}  backend={jax.default_backend()}")

# --- af2seq ---
from af2seq import GradientDesign

PDB       = os.path.join(os.path.dirname(__file__), "test_monomer.pdb")
DATA_DIR  = os.path.expanduser("~/Library/Caches/colabfold")
OUTPUT    = os.path.join(os.path.dirname(__file__), "test_results")
os.makedirs(OUTPUT, exist_ok=True)
ITERATIONS = 3   # 動作確認用に最小値

print(f"[Input]   {PDB}")
print(f"[Weights] {DATA_DIR}")
print(f"[Iter]    {ITERATIONS}")
print("Starting... (初回は XLA コンパイルで数分かかる場合があります)")

d = GradientDesign(datadir=DATA_DIR, output_path=OUTPUT)
d.design(
    target_file=PDB,
    iterations=ITERATIONS,
    recycles=0,
    loss={"FAPE": 1.0},
)

print("[Done] テスト完了")
