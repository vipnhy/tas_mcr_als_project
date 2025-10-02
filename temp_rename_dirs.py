import shutil
from pathlib import Path

ROOT = Path("analysis_runs/demo_vis_batch_20251002_102611/global_fit")

if not ROOT.exists():
    raise SystemExit("示例全局拟合目录不存在")

REPLACEMENTS = [
    ("<->", "_rev_"),
    ("->", "_to_"),
    (";", "__"),
]

def slugify(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace(" ", "_")
    sanitized = "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})
    sanitized = sanitized.strip("_-")
    return sanitized.lower() or "model"

def letters(n: int):
    return [chr(ord("A") + i) for i in range(max(n, 1))]

def sequential_name(n: int) -> str:
    seq_letters = letters(n)[:n]
    folder = "sequential_" + "_to_".join(seq_letters)
    return slugify(folder)

def parallel_name(n: int) -> str:
    seq_letters = letters(n)[:n]
    if n <= 1:
        base = "parallel"
    else:
        target = seq_letters[-1]
        sources = seq_letters[:-1]
        base = "parallel_" + "__".join(f"{src}_to_{target}" for src in sources)
    return slugify(base)

def mixed_direct_display(seq_letters):
    if len(seq_letters) <= 1:
        return seq_letters[0]
    if len(seq_letters) == 2:
        return f"{seq_letters[0]}->{seq_letters[1]}"
    seq_part = "->".join(seq_letters[:-1])
    return f"{seq_part}; {seq_letters[0]}->{seq_letters[-1]}"

def mixed_direct_name(n: int) -> str:
    seq_letters = letters(n)[:n]
    display = mixed_direct_display(seq_letters)
    base = f"mixed_direct_{slugify(display)}"
    return slugify(base)

def mixed_reversible_display(seq_letters):
    if len(seq_letters) <= 1:
        return seq_letters[0]
    parts = [f"{seq_letters[0]}<->{seq_letters[1]}"]
    for idx in range(1, len(seq_letters) - 1):
        parts.append(f"{seq_letters[idx]}->{seq_letters[idx+1]}")
    return "; ".join(parts)

def mixed_reversible_name(n: int) -> str:
    seq_letters = letters(n)[:n]
    display = mixed_reversible_display(seq_letters)
    base = f"mixed_reversible_{slugify(display)}"
    return slugify(base)

MODEL_NAME_FUNCS = {
    "gta_sequential": sequential_name,
    "gta_parallel": parallel_name,
    "gta_mixed_direct": mixed_direct_name,
    "gta_mixed_reversible": mixed_reversible_name,
}

def rename_models():
    for comp_dir in ROOT.glob("components_*"):
        try:
            n_components = int(comp_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        for run_dir in comp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for old_name, func in MODEL_NAME_FUNCS.items():
                src = run_dir / old_name
                if src.is_dir():
                    dst = run_dir / func(n_components)
                    if dst.exists():
                        print(f"跳过 {src} -> {dst} (目标已存在)")
                        continue
                    try:
                        src.rename(dst)
                        print(f"已重命名 {src.name} -> {dst.name}")
                    except PermissionError:
                        print(f"权限不足，尝试复制替代: {src} -> {dst}")
                        shutil.copytree(src, dst)
                        shutil.rmtree(src)
                    except OSError as exc:
                        print(f"重命名失败: {src} -> {dst}: {exc}")

if __name__ == "__main__":
    rename_models()
    print("目录重命名完成")
