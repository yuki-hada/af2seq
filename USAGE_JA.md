# AF2Seq 使い方ガイド（arm64 Mac 向け）

AlphaFoldの構造予測を損失関数として使い、**固定バックボーンに対する配列設計**を行うツールです。

---

## 環境のセットアップ

Python 仮想環境を作成してパッケージをインストールします。

```bash
python3 -m venv ~/.venvs/af2seq-arm64
~/.venvs/af2seq-arm64/bin/pip install -e .
~/.venvs/af2seq-arm64/bin/pip install "jax[cpu]" dm-haiku optax chex \
    tensorflow openmm pdbfixer pydssp torch
```

| 項目 | 値 |
|------|----|
| JAX | 0.10.0（CPU / arm64 native） |
| XLAキャッシュ | `~/.cache/jax_xla`（初回のみコンパイル、以降再利用） |
| AlphaFold重み | ColabFoldキャッシュを流用（`~/Library/Caches/colabfold`） |

---

## 1. 最小動作確認（テスト実行）

```bash
~/.venvs/af2seq-arm64/bin/python test_design.py
```

- 入力: `test_monomer.pdb`（リポジトリに同梱）
- 出力: `test_results/` ディレクトリ
- 初回は XLA コンパイルで数分かかる（2回目以降はキャッシュで高速）

---

## 2. コマンドライン（design.py）

論文の手順通りに**二次構造ベースの初期化 → 固定残基指定 → 勾配降下**をまとめたスクリプトです。

```bash
~/.venvs/af2seq-arm64/bin/python design.py \
    --pdb target.pdb \
    --output results/
```

### オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--pdb` | （必須） | 入力PDBファイル（単一鎖） |
| `--output` | （必須） | 出力ディレクトリ |
| `--data_dir` | `~/Library/Caches/colabfold` | AlphaFold重みのディレクトリ |
| `--fix_pos` | なし | 固定残基の位置（**1-indexed**、複数可） |
| `--iterations` | 500 | 勾配降下のステップ数 |
| `--recycles` | 0 | AF2のリサイクル数 |
| `--trajectories` | 1 | トラジェクトリ数（独立した設計の本数） |
| `--mut_rate` | 0.1 | 初期配列へのランダムミューテーション率（10%） |
| `--seed` | 42 | 乱数シード |

### 使用例

```bash
# 固定残基を指定（1番・5番・10番を固定）
~/.venvs/af2seq-arm64/bin/python design.py \
    --pdb target.pdb \
    --output results/ \
    --fix_pos 1 5 10

# 複数トラジェクトリで多様な配列を生成
~/.venvs/af2seq-arm64/bin/python design.py \
    --pdb target.pdb \
    --output results/ \
    --trajectories 10 \
    --iterations 500

# 複数PDBをループ処理（パイプライン）
for pdb in targets/*.pdb; do
    ~/.venvs/af2seq-arm64/bin/python design.py \
        --pdb "$pdb" \
        --output "results/$(basename "$pdb" .pdb)" \
        --fix_pos 1 5 10 \
        --iterations 500
done
```

### アルゴリズムの流れ

```
1. 初期化
   PDB → pydssp で二次構造を推定
   H(ヘリックス)→A / E(βシート)→V / -(ループ)→G
   + mut_rate のランダムミューテーションで多様性付与
   ※ fix_pos の残基は変更しない

2. 勾配降下（iterations ステップ）
   配列 → AF2 × 5モデル → 構造予測
   → FAPE損失を計算
   → 勾配をone-hot入力へ逆伝播（5×20×N）
   → 5モデルの平均勾配（20×N）でPSSMをADAM更新

3. 出力
   全ステップ中でpLDDTが最高だった配列を1本出力
   （trajectories=N なら独立したN本）
```

### 出力ファイル

各トラジェクトリの結果は `output/traj_000/` のように番号付きで保存されます。

```
results/
  traj_000/
    design_<name>_<id>_config.json      # 設計パラメータ
    design_<name>_<id>_extended.json    # 全ステップの損失トラッキング
    design_<name>_<id>_best.pdb         # 最良構造のPDB
```

### 実行時間の目安（CPU）

| 残基数 | 1ステップ | 500ステップ |
|--------|----------|------------|
| ~100残基 | ~85秒 | ~12時間 |

初回はXLAコンパイルが走るため数分追加でかかります。2回目以降は `~/.cache/jax_xla` のキャッシュで高速化されます。

---

## 3. Python API

### 勾配降下法（GradientDesign）

```python
import os
import jax

# メモリ節約・キャッシュ設定（スクリプト先頭に必ず書く）
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_xla"))

from af2seq import GradientDesign
from af2seq.design.utils import generate_start_sequence

# 二次構造ベースで初期配列を生成
start_seq = generate_start_sequence("target.pdb")
# H→A / E→V / -→G に変換した文字列が返る

d = GradientDesign(
    datadir=os.path.expanduser("~/Library/Caches/colabfold"),
    output_path="./results",
)

d.design(
    target_file="target.pdb",
    start_seq=[start_seq],
    fix_pos=[1, 5, 10],  # 1-indexed
    iterations=500,
    recycles=0,
    loss={"FAPE": 1.0},
)
```

### MCMC（MCMCDesign）

```python
from af2seq import MCMCDesign

mcmc = MCMCDesign(
    datadir=os.path.expanduser("~/Library/Caches/colabfold"),
    output_path="./results",
    random_seed=0,
    mcmc_muts=1,
)
mcmc.design(target_file="target.pdb", iterations=500)
```

---

## 4. 損失関数の種類

`loss` 辞書に複数を組み合わせられます（重みで調整）:

```python
loss={
    "FAPE": 1.0,      # バックボーン座標の誤差（メインの損失）
    "pLDDT": 0.5,     # 予測信頼度（高いほど良い）
    "SC_FAPE": 0.3,   # サイドチェーンの座標誤差
}
```

---

## トラブルシューティング

**初回が非常に遅い（数分）**  
→ XLA コンパイル中。`~/.cache/jax_xla` にキャッシュが作られ、2回目以降は速くなります。

**OOM / カーネルパニック**  
→ `XLA_PYTHON_CLIENT_PREALLOCATE=false` を必ず設定。  
→ `resource.setrlimit` でメモリ上限を設定（`test_design.py` 参照）。

**`pyrosetta` 警告が出る**  
→ 動作には不要なため無視して問題ありません。
