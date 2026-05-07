"""
af2seq gradient descent design pipeline

Usage examples:
  # 基本実行（二次構造ベース初期化）
  ~/.venvs/af2seq-arm64/bin/python design.py \
      --pdb input.pdb \
      --output results/

  # 固定残基あり（1-indexed、複数可）
  ~/.venvs/af2seq-arm64/bin/python design.py \
      --pdb input.pdb \
      --output results/ \
      --fix_pos 1 5 10 23

  # 複数トラジェクトリ
  ~/.venvs/af2seq-arm64/bin/python design.py \
      --pdb input.pdb \
      --output results/ \
      --trajectories 5 \
      --iterations 500

  # ランダムミューテーション率を変更（デフォルト10%）
  ~/.venvs/af2seq-arm64/bin/python design.py \
      --pdb input.pdb \
      --output results/ \
      --mut_rate 0.15
"""

import argparse
import os
import random
import resource

# --- メモリ上限: kernel panic 防止 ---
try:
    _limit = 20 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
except Exception:
    pass

# --- XLA 永続キャッシュ ---
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
_xla_cache = os.path.expanduser("~/.cache/jax_xla")
os.makedirs(_xla_cache, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _xla_cache)

from af2seq import GradientDesign
from af2seq.design.utils import generate_start_sequence

AA = list("ACDEFGHIKLMNPQRSTVWY")


def add_random_mutations(seq: str, rate: float, fix_pos: list) -> str:
    """rate の割合でランダムにアミノ酸を置換する（fix_pos は変更しない）。"""
    fixed = set(p - 1 for p in (fix_pos or []))  # 0-indexed に変換
    seq = list(seq)
    for i in range(len(seq)):
        if i not in fixed and random.random() < rate:
            seq[i] = random.choice([aa for aa in AA if aa != seq[i]])
    return "".join(seq)


def run_design(args, trajectory_id: int):
    print(f"\n=== Trajectory {trajectory_id + 1}/{args.trajectories} ===")

    # 1. 二次構造ベースで初期配列を生成
    start_seq = generate_start_sequence(args.pdb)
    print(f"[SS init]  {start_seq}")

    # 2. 10% ランダムミューテーションで多様性を付与
    start_seq = add_random_mutations(start_seq, args.mut_rate, args.fix_pos)
    print(f"[Mutated]  {start_seq}")

    if args.fix_pos:
        fixed_aas = "".join(start_seq[p - 1] for p in sorted(args.fix_pos))
        print(f"[Fixed]    positions={args.fix_pos} → AAs={fixed_aas}")

    # 3. 設計実行
    output_dir = os.path.join(args.output, f"traj_{trajectory_id:03d}")
    os.makedirs(output_dir, exist_ok=True)

    d = GradientDesign(datadir=args.data_dir, output_path=output_dir)
    d.design(
        target_file=args.pdb,
        start_seq=[start_seq],
        fix_pos=args.fix_pos if args.fix_pos else None,
        iterations=args.iterations,
        recycles=args.recycles,
        loss={"FAPE": 1.0},
    )

    best = d.best_result
    print(f"[Result]   {best['sequence']}")
    return best


def main():
    parser = argparse.ArgumentParser(description="af2seq gradient descent design")
    parser.add_argument("--pdb",        required=True,  help="ターゲットPDBファイル")
    parser.add_argument("--output",     required=True,  help="出力ディレクトリ")
    parser.add_argument("--data_dir",   default=os.path.expanduser("~/Library/Caches/colabfold"),
                        help="AlphaFoldの重みディレクトリ")
    parser.add_argument("--fix_pos",    type=int, nargs="+", default=None,
                        help="固定残基の位置（1-indexed、複数可）例: --fix_pos 1 5 10")
    parser.add_argument("--iterations", type=int, default=500, help="勾配降下ステップ数")
    parser.add_argument("--recycles",   type=int, default=0,   help="AF2リサイクル数")
    parser.add_argument("--trajectories", type=int, default=1, help="トラジェクトリ数（生成配列数）")
    parser.add_argument("--mut_rate",   type=float, default=0.1,
                        help="初期配列へのランダムミューテーション率（デフォルト: 0.1 = 10%%）")
    parser.add_argument("--seed",       type=int, default=42,  help="乱数シード")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    print(f"[PDB]          {args.pdb}")
    print(f"[Output]       {args.output}")
    print(f"[Iterations]   {args.iterations}")
    print(f"[Trajectories] {args.trajectories}")
    print(f"[Fix pos]      {args.fix_pos or 'none'}")
    print(f"[Mut rate]     {args.mut_rate * 100:.0f}%")

    results = []
    for i in range(args.trajectories):
        result = run_design(args, i)
        results.append({"trajectory": i, "sequence": result["sequence"]})

    # サマリー出力
    print("\n=== Summary ===")
    for r in results:
        print(f"  [{r['trajectory']:03d}] {r['sequence']}")


if __name__ == "__main__":
    main()
