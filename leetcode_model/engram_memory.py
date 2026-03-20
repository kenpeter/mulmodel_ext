"""
Engram memory for autoresearch loop.
Stores experiment results, recalls relevant past experiments.
"""

from engram import Memory

# Initialize once
_memory = Memory(path="/home/kenpeter/work/mulmodel_ext/leetcode_model/engram_data")


def store_result(cycle, pass_rate, compile_count, total, change_summary=""):
    """Store experiment result in Engram."""
    content = (
        f"Cycle {cycle}: pass={pass_rate:.1f}%, compile={compile_count}/{total}. "
        f"Change: {change_summary or 'none'}"
    )
    importance = 0.3 + (pass_rate / 100)  # higher pass = more important
    _memory.add(
        content=content,
        type="episodic",
        importance=min(importance, 1.0),
        source="autoresearch",
        tags=["training", "eval", f"cycle_{cycle}"],
        metadata={
            "cycle": cycle,
            "pass_rate": pass_rate,
            "compile_count": compile_count,
            "total": total,
        },
    )


def recall_relevant(query="what training changes improved results?", limit=5):
    """Recall relevant past experiments."""
    results = _memory.recall(query, limit=limit)
    if not results:
        return []
    return results


def get_patterns():
    """Get consolidated patterns from all experiments."""
    # Recall what worked
    worked = _memory.recall("what improved pass rate or compile rate?", limit=5)
    # Recall what failed
    failed = _memory.recall("what made results worse?", limit=5)
    return {"worked": worked, "failed": failed}


def consolidate():
    """Run memory consolidation (transfer to long-term)."""
    _memory.consolidate()


def stats():
    """Get memory stats."""
    return _memory.stats()
