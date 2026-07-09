import pytest

from app.caching.lru_cache import LRUTTLCache, cache_key


def test_get_on_empty_cache_returns_none_and_counts_a_miss():
    cache = LRUTTLCache[str, int](max_size=4)

    assert cache.get("missing") is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0


def test_set_then_get_returns_the_value_and_counts_a_hit():
    cache = LRUTTLCache[str, int](max_size=4)

    cache.set("a", 1)

    assert cache.get("a") == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


def test_evicts_the_least_recently_used_entry_when_full():
    cache = LRUTTLCache[str, int](max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)

    cache.set("c", 3)  # evicts "a" (oldest, never touched since insert)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert cache.stats.evictions == 1


def test_accessing_an_entry_marks_it_as_recently_used_protecting_it_from_eviction():
    cache = LRUTTLCache[str, int](max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # "a" is now more recently used than "b"

    cache.set("c", 3)  # should evict "b", not "a"

    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_entry_expires_after_its_ttl_using_an_injected_clock():
    fake_time = [0.0]
    cache = LRUTTLCache[str, int](max_size=4, clock=lambda: fake_time[0])
    cache.set("a", 1, ttl=10)

    fake_time[0] = 5.0
    assert cache.get("a") == 1  # not yet expired

    fake_time[0] = 10.0
    assert cache.get("a") is None  # expired at exactly ttl


def test_default_ttl_applies_when_no_per_call_ttl_given():
    fake_time = [0.0]
    cache = LRUTTLCache[str, int](max_size=4, default_ttl=5, clock=lambda: fake_time[0])
    cache.set("a", 1)

    fake_time[0] = 5.0
    assert cache.get("a") is None


def test_per_call_ttl_overrides_the_default():
    fake_time = [0.0]
    cache = LRUTTLCache[str, int](max_size=4, default_ttl=1, clock=lambda: fake_time[0])
    cache.set("a", 1, ttl=100)

    fake_time[0] = 5.0
    assert cache.get("a") == 1


def test_no_ttl_means_the_entry_never_expires():
    cache = LRUTTLCache[str, int](max_size=4)
    cache.set("a", 1)

    assert cache.get("a") == 1


def test_clear_empties_the_cache_but_keeps_cumulative_counters():
    cache = LRUTTLCache[str, int](max_size=4)
    cache.set("a", 1)
    cache.get("a")
    cache.get("missing")

    cache.clear()

    assert cache.get("a") is None
    assert cache.stats.size == 0
    assert cache.stats.hits == 1  # historical counters survive a clear
    assert cache.stats.misses == 2


def test_hit_ratio_is_zero_with_no_activity():
    cache = LRUTTLCache[str, int](max_size=4)

    assert cache.stats.hit_ratio == 0.0


def test_hit_ratio_computed_from_hits_and_misses():
    cache = LRUTTLCache[str, int](max_size=4)
    cache.set("a", 1)
    cache.get("a")
    cache.get("a")
    cache.get("missing")

    assert cache.stats.hit_ratio == pytest.approx(2 / 3, rel=1e-3)


def test_max_size_must_be_at_least_one():
    with pytest.raises(ValueError):
        LRUTTLCache[str, int](max_size=0)


def test_stats_size_reflects_current_entry_count():
    cache = LRUTTLCache[str, int](max_size=4)
    cache.set("a", 1)
    cache.set("b", 2)

    assert cache.stats.size == 2
    assert cache.stats.max_size == 4


def test_cache_key_joins_parts_into_a_stable_readable_string():
    assert cache_key("query text", 8) == "query text|8"
    assert cache_key("a", "b", "c") == cache_key("a", "b", "c")
