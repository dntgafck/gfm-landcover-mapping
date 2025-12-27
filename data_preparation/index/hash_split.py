import hashlib


def get_stable_hash_float(key: str, seed: int) -> float:
    """
    Computes a deterministic float in [0, 1) using SHA256 of "seed:key".
    """
    input_str = f"{seed}:{key}"
    h = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
    # Use first 16 chars (8 bytes) to get a 64-bit int
    # 0xFFFFFFFFFFFFFFFF is the max value for 64-bit unsigned int
    u = int(h[:16], 16) / 0xFFFFFFFFFFFFFFFF
    return u


def assign_split(u: float, fractions: dict) -> str:
    """
    Assigns a split based on the random float and target fractions.
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    """
    train_frac = fractions["train"]
    val_frac = fractions["val"]

    if u < train_frac:
        return "train"
    elif u < (train_frac + val_frac):
        return "val"
    else:
        return "test"


def validate_fractions(fractions: dict, tolerance: float = 1e-6) -> None:
    """
    Ensures fractions sum to 1.0 and are non-negative.
    """
    if any(f < 0 for f in fractions.values()):
        raise ValueError("Fractions must be non-negative.")

    total = sum(fractions.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")
