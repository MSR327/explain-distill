import pytest

torch = pytest.importorskip("torch")


def test_fix_dtypes_makes_bf16_safe_for_fp16_amp():
    from distill.sft import fix_dtypes

    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(torch.bfloat16)
    for p in m[0].parameters():
        p.requires_grad = False
    census = fix_dtypes(m)
    assert census == {"bfloat16": 2, "bfloat16/train": 2}  # census reports what it found
    assert all(p.dtype == torch.float16 for p in m[0].parameters())  # frozen -> fp16
    assert all(p.dtype == torch.float32 for p in m[1].parameters())  # trainable -> fp32 (GradScaler needs this)
