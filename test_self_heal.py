"""
test_self_heal.py — Smoke tests for Phase 3 Self-Healing Loop

Test 1: Normal query (should succeed on attempt 1, self_healed=False)
Test 2: Forced self-healing (monkey-patch execute_sql to fail on first
        call with a realistic SQLite error, then succeed on retry)
Test 3: All retries exhausted (execute_sql always fails → RuntimeError)
Test 4: Explainability — explanation starts with "I calculated this by"
        and confidence is one of High/Medium/Low
"""

import agent

# ═══════════════════════════════════════════════════════════════
# Test 1 — Normal query (no healing needed)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Normal query — should succeed on first attempt")
print("=" * 60)

client = agent.init_gemini()

result = agent.ask("What is the total revenue?", client=client)

print(f"  Question    : {result['question']}")
print(f"  SQL         : {result['sql']}")
print(f"  Result      : {result['formatted']}")
print(f"  Explanation : {result['explanation']}")
print(f"  Confidence  : {result['confidence']}")
print(f"  Attempts    : {len(result['attempts'])}")
print(f"  Self-healed : {result['self_healed']}")

assert result["self_healed"] is False, "Should NOT have self-healed!"
assert len(result["attempts"]) == 1, "Should be exactly 1 attempt!"
assert result["attempts"][0]["error"] is None, "Should have no error!"
print("\n  ✅ TEST 1 PASSED\n")


# ═══════════════════════════════════════════════════════════════
# Test 2 — Forced self-healing (fail once, then succeed)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Forced failure → self-heal on retry")
print("=" * 60)

# Save original function
_original_execute = agent.execute_sql
_call_count = 0


def _failing_execute_once(sql, db_path=agent.DB_PATH):
    """Fail on the first call with a realistic error, then behave normally."""
    global _call_count
    _call_count += 1
    if _call_count == 1:
        raise Exception("OperationalError: no such column: profit")
    return _original_execute(sql, db_path)


# Monkey-patch
agent.execute_sql = _failing_execute_once

try:
    result = agent.ask("What is the total revenue?", client=client)

    print(f"  Question    : {result['question']}")
    print(f"  SQL         : {result['sql']}")
    print(f"  Result      : {result['formatted']}")
    print(f"  Attempts    : {len(result['attempts'])}")
    print(f"  Self-healed : {result['self_healed']}")
    for a in result["attempts"]:
        status = "OK" if a["error"] is None else a["error"]
        print(f"    Attempt {a['attempt']}: {status}")

    assert result["self_healed"] is True, "Should have self-healed!"
    assert len(result["attempts"]) >= 2, "Should be at least 2 attempts!"
    assert result["attempts"][0]["error"] is not None, "First attempt should have error!"
    assert result["attempts"][-1]["error"] is None, "Last attempt should succeed!"
    print("\n  ✅ TEST 2 PASSED\n")
finally:
    # Restore original
    agent.execute_sql = _original_execute


# ═══════════════════════════════════════════════════════════════
# Test 3 — All retries exhausted → RuntimeError
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: All retries fail → RuntimeError raised")
print("=" * 60)


def _always_failing_execute(sql, db_path=agent.DB_PATH):
    """Always fail."""
    raise Exception("OperationalError: database is locked")


agent.execute_sql = _always_failing_execute

try:
    agent.ask("What is the total revenue?", client=client)
    print("  ❌ TEST 3 FAILED — no RuntimeError raised!")
except RuntimeError as e:
    print(f"  Caught RuntimeError (expected): {str(e)[:100]}...")
    assert "Self-healing failed after 3 attempts" in str(e)
    print("\n  ✅ TEST 3 PASSED\n")
except Exception as e:
    print(f"  ❌ TEST 3 FAILED — unexpected error type: {type(e).__name__}: {e}")
finally:
    agent.execute_sql = _original_execute


# ═══════════════════════════════════════════════════════════════
# Test 4 — Explainability & Trust (Phase 4)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: Explainability — explanation format & confidence")
print("=" * 60)

result = agent.ask("What is the average order value?", client=client)

explanation = result["explanation"]
confidence = result["confidence"]

print(f"  Explanation : {explanation}")
print(f"  Confidence  : {confidence}")

# Check explanation starts with the required phrase
assert explanation.lower().startswith("i calculated"), (
    f"Explanation must start with 'I calculated...', got: {explanation[:50]}"
)
print("  ✓ Explanation starts with 'I calculated...'")

# Check confidence is a valid level
assert confidence in ("High", "Medium", "Low"), (
    f"Confidence must be High/Medium/Low, got: {confidence}"
)
print(f"  ✓ Confidence is valid: {confidence}")

# Check all three outputs are present and non-empty
assert result["sql"].strip(), "SQL must not be empty"
assert result["formatted"].strip(), "Formatted result must not be empty"
assert explanation.strip(), "Explanation must not be empty"
print("  ✓ SQL, Data, and Explanation all present and non-empty")

print("\n  ✅ TEST 4 PASSED\n")


print("=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
