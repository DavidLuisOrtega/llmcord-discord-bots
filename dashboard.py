import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp.web
import aiosqlite
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/llmcord.db")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))

BOT_NAMES = {
    "kevin": "Kevin",
    "saul": "Saul",
    "katherine": "Katherine",
    "damon": "Damon",
    "sarah": "Sarah",
}

STAT_TYPES = ["messages", "dms", "reactions", "proactive", "b2b", "memories"]

redis_client = None


async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return redis_client


async def get_sqlite():
    db_path = Path(SQLITE_PATH)
    if not db_path.exists():
        return None
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    return db


async def discover_bot_ids(r) -> dict[str, str]:
    bot_ids = {}
    keys = []
    cursor = "0"
    while True:
        cursor, batch = await r.scan(cursor=cursor, match="llmcord:bot:display_name:*", count=100)
        keys.extend(batch)
        if cursor == "0" or cursor == 0:
            break
    for key in keys:
        bot_id = key.split(":")[-1]
        name = await r.get(key) or bot_id
        bot_ids[bot_id] = name
    return bot_ids


async def collect_stats(r, bot_ids: dict[str, str], days: int = 7) -> dict:
    today = datetime.now(timezone.utc).date()
    stats = {}
    for bot_id, bot_name in bot_ids.items():
        bot_stats = {"name": bot_name, "id": bot_id, "daily": {}}
        for d in range(days):
            day = (today - timedelta(days=d)).isoformat()
            day_stats = {}
            for stat_type in STAT_TYPES:
                key = f"llmcord:stats:{stat_type}:{bot_id}:{day}"
                val = await r.get(key)
                day_stats[stat_type] = int(val) if val else 0
            bot_stats["daily"][day] = day_stats
        stats[bot_id] = bot_stats
    return stats


async def collect_vibes(r) -> dict:
    vibes = {}
    observed = await r.smembers("llmcord:observed_channels")
    for ch_id in observed:
        heuristic = await r.get(f"llmcord:channel:vibe:{ch_id}")
        enriched = await r.get(f"llmcord:channel:vibe_enriched:{ch_id}")
        if heuristic or enriched:
            vibes[ch_id] = {"heuristic": heuristic or "", "enriched": enriched or ""}
    return vibes


async def collect_memory_stats(db) -> dict:
    if db is None:
        return {"total_memories": 0, "total_moments": 0, "by_bot": {}}
    rows = await db.execute_fetchall("SELECT bot_id, COUNT(*) as cnt FROM memories GROUP BY bot_id")
    by_bot = {r[0]: r[1] for r in rows}
    total_mem = sum(by_bot.values())
    moment_rows = await db.execute_fetchall("SELECT COUNT(*) FROM shared_moments")
    total_moments = moment_rows[0][0] if moment_rows else 0
    await db.close()
    return {"total_memories": total_mem, "total_moments": total_moments, "by_bot": by_bot}


async def api_stats(request):
    r = await get_redis()
    bot_ids = await discover_bot_ids(r)
    stats = await collect_stats(r, bot_ids)
    vibes = await collect_vibes(r)
    db = await get_sqlite()
    mem_stats = await collect_memory_stats(db)
    return aiohttp.web.json_response({
        "stats": stats,
        "vibes": vibes,
        "memory": mem_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })


async def index(request):
    return aiohttp.web.Response(text=HTML_PAGE, content_type="text/html")


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llmcord Dashboard</title>
<style>
  :root {
    --bg: #1a1b26; --surface: #24283b; --border: #3b4261;
    --text: #c0caf5; --text-dim: #565f89; --accent: #7aa2f7;
    --green: #9ece6a; --orange: #e0af68; --red: #f7768e;
    --purple: #bb9af7; --cyan: #7dcfff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, sans-serif; padding: 24px; }
  h1 { font-size: 1.5rem; margin-bottom: 8px; color: var(--accent); }
  .subtitle { color: var(--text-dim); font-size: 0.85rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 1.05rem; margin-bottom: 12px; color: var(--accent); }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; color: var(--text-dim); font-weight: 500; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 6px 8px; border-bottom: 1px solid var(--border); }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .bar-container { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .bar-label { min-width: 80px; font-size: 0.8rem; color: var(--text-dim); }
  .bar { height: 20px; border-radius: 4px; min-width: 2px; transition: width 0.3s; }
  .bar.messages { background: var(--accent); }
  .bar.dms { background: var(--purple); }
  .bar.reactions { background: var(--green); }
  .bar.proactive { background: var(--orange); }
  .bar.b2b { background: var(--cyan); }
  .bar.memories { background: var(--red); }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; font-size: 0.8rem; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; }
  .vibe-tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; margin: 2px; }
  .vibe-hyped { background: #e0af6833; color: var(--orange); }
  .vibe-playful { background: #9ece6a33; color: var(--green); }
  .vibe-heated { background: #f7768e33; color: var(--red); }
  .vibe-sad { background: #bb9af733; color: var(--purple); }
  .vibe-deep { background: #7aa2f733; color: var(--accent); }
  .vibe-chill { background: #7dcfff33; color: var(--cyan); }
  .vibe-chaotic { background: #f7768e33; color: var(--red); }
  .big-num { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .stat-row { display: flex; gap: 32px; margin-top: 8px; }
  .stat-item { text-align: center; }
  .stat-item .label { font-size: 0.75rem; color: var(--text-dim); }
  #refresh-btn { background: var(--accent); color: var(--bg); border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  #refresh-btn:hover { opacity: 0.85; }
  .enriched { font-style: italic; color: var(--text-dim); font-size: 0.8rem; }
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center;">
  <div><h1>llmcord Dashboard</h1><div class="subtitle" id="timestamp">Loading...</div></div>
  <button id="refresh-btn" onclick="loadData()">Refresh</button>
</div>

<div class="grid" id="summary-cards"></div>

<div class="card" style="margin-bottom:24px;">
  <h2>Activity (Last 24h)</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div>Messages</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--purple)"></div>DMs</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--green)"></div>Reactions</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--orange)"></div>Proactive</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--cyan)"></div>B2B</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--red)"></div>Memories</div>
  </div>
  <div id="activity-bars"></div>
</div>

<div class="grid">
  <div class="card">
    <h2>7-Day Totals</h2>
    <table>
      <thead><tr><th>Bot</th><th class="num">Msgs</th><th class="num">DMs</th><th class="num">React</th><th class="num">Proact</th><th class="num">B2B</th><th class="num">Mem</th></tr></thead>
      <tbody id="weekly-table"></tbody>
    </table>
  </div>
  <div class="card">
    <h2>Channel Vibes</h2>
    <div id="vibes-section"><span class="enriched">No vibes detected yet</span></div>
  </div>
</div>

<div class="grid" style="margin-top:16px;">
  <div class="card">
    <h2>Memory &amp; Moments</h2>
    <div id="memory-section"></div>
  </div>
</div>

<script>
async function loadData() {
  try {
    const resp = await fetch('/api/stats');
    const data = await resp.json();
    render(data);
  } catch(e) {
    document.getElementById('timestamp').textContent = 'Error loading data: ' + e.message;
  }
}

function render(data) {
  const ts = new Date(data.generated_at);
  document.getElementById('timestamp').textContent = 'Last updated: ' + ts.toLocaleString();

  const stats = data.stats;
  const botIds = Object.keys(stats);
  const today = new Date().toISOString().split('T')[0];

  let summaryHtml = '';
  let totalMsgs24h = 0;
  botIds.forEach(id => {
    const bot = stats[id];
    const todayStats = bot.daily[today] || {};
    const msgs = todayStats.messages || 0;
    totalMsgs24h += msgs;
    const total24h = Object.values(todayStats).reduce((a,b) => a+b, 0);
    summaryHtml += `<div class="card"><h2>${bot.name}</h2><div class="big-num">${total24h}</div><div class="stat-row">
      <div class="stat-item"><div>${msgs}</div><div class="label">messages</div></div>
      <div class="stat-item"><div>${todayStats.dms||0}</div><div class="label">DMs</div></div>
      <div class="stat-item"><div>${todayStats.reactions||0}</div><div class="label">reactions</div></div>
      <div class="stat-item"><div>${todayStats.proactive||0}</div><div class="label">proactive</div></div>
    </div></div>`;
  });
  document.getElementById('summary-cards').innerHTML = summaryHtml;

  const maxVal = Math.max(1, ...botIds.flatMap(id => {
    const d = stats[id].daily[today] || {};
    return Object.values(d);
  }));
  let barsHtml = '';
  const types = ['messages','dms','reactions','proactive','b2b','memories'];
  botIds.forEach(id => {
    const bot = stats[id];
    const d = bot.daily[today] || {};
    barsHtml += `<div style="margin-bottom:12px;"><strong>${bot.name}</strong>`;
    types.forEach(t => {
      const val = d[t] || 0;
      const pct = Math.max(2, (val/maxVal)*100);
      barsHtml += `<div class="bar-container"><div class="bar-label">${t}</div><div class="bar ${t}" style="width:${pct}%"></div><span style="font-size:0.8rem">${val}</span></div>`;
    });
    barsHtml += '</div>';
  });
  document.getElementById('activity-bars').innerHTML = barsHtml;

  let weeklyHtml = '';
  botIds.forEach(id => {
    const bot = stats[id];
    const totals = {};
    types.forEach(t => totals[t] = 0);
    Object.values(bot.daily).forEach(day => {
      types.forEach(t => totals[t] += (day[t] || 0));
    });
    weeklyHtml += `<tr><td>${bot.name}</td>${types.map(t => `<td class="num">${totals[t]}</td>`).join('')}</tr>`;
  });
  document.getElementById('weekly-table').innerHTML = weeklyHtml;

  const vibes = data.vibes;
  const vibeChannels = Object.keys(vibes);
  if (vibeChannels.length > 0) {
    let vibeHtml = '';
    vibeChannels.forEach(ch => {
      const v = vibes[ch];
      const cls = 'vibe-' + (v.heuristic || 'chill');
      vibeHtml += `<div style="margin-bottom:8px;">Channel ${ch}: <span class="vibe-tag ${cls}">${v.heuristic||'unknown'}</span>`;
      if (v.enriched) vibeHtml += ` <span class="enriched">${v.enriched}</span>`;
      vibeHtml += '</div>';
    });
    document.getElementById('vibes-section').innerHTML = vibeHtml;
  }

  const mem = data.memory;
  let memHtml = `<div class="stat-row">
    <div class="stat-item"><div class="big-num">${mem.total_memories}</div><div class="label">Memories</div></div>
    <div class="stat-item"><div class="big-num">${mem.total_moments}</div><div class="label">Shared Moments</div></div>
  </div>`;
  if (Object.keys(mem.by_bot).length > 0) {
    memHtml += '<table style="margin-top:12px"><thead><tr><th>Bot</th><th class="num">Memories</th></tr></thead><tbody>';
    Object.entries(mem.by_bot).forEach(([bid, cnt]) => {
      const name = stats[bid] ? stats[bid].name : bid;
      memHtml += `<tr><td>${name}</td><td class="num">${cnt}</td></tr>`;
    });
    memHtml += '</tbody></table>';
  }
  document.getElementById('memory-section').innerHTML = memHtml;
}

loadData();
setInterval(loadData, 30000);
</script>
</body>
</html>"""


app = aiohttp.web.Application()
app.router.add_get("/", index)
app.router.add_get("/api/stats", api_stats)


if __name__ == "__main__":
    logging.info("Starting dashboard on port %d", DASHBOARD_PORT)
    aiohttp.web.run_app(app, port=DASHBOARD_PORT, print=lambda msg: logging.info(msg))
