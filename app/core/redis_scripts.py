"""Redis Lua scripts shared by the code that keeps short-lived entries in sorted sets (sandbox coordination, research tasks).

A sorted set whose members are things that are alive, scored by the moment they lapse, with the time taken from Redis
itself rather than from whichever machine asks. Whoever owns a member renews it before it lapses; if the owner dies it
simply stops being there."""

# ZADD every member with score = (Redis clock now) + ttl. ARGV[1] = ttl ms, ARGV[2..] = members.
ADD_MEMBERS = """
local t = redis.call('TIME')
local lapse = t[1] * 1000 + math.floor(t[2] / 1000) + tonumber(ARGV[1])
for i = 2, #ARGV do redis.call('ZADD', KEYS[1], lapse, ARGV[i]) end
return #ARGV - 1
"""

# Drop what has lapsed, return what is left.
LIVE_MEMBERS = """
local t = redis.call('TIME')
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', t[1] * 1000 + math.floor(t[2] / 1000))
return redis.call('ZRANGE', KEYS[1], 0, -1)
"""
