import os

target = r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\frontend\templates\survey.html"

with open(target, 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Remove CSS
css_to_remove = """  /* Floating timer */
  .float-timer {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: var(--radius);
    padding: 0.75rem 1.25rem;
    z-index: 50;
    font-size: 0.8rem;
    color: var(--fg-muted);
    box-shadow: var(--glow-purple);
  }

  .float-timer .elapsed {
    font-family: var(--font-mono);
    color: var(--purple);
    font-weight: 700;
  }"""

# 2. Remove HTML
html_to_remove = """<!-- Floating timer -->
<div class="float-timer">
  ⏱ Time: <span class="elapsed" id="timerDisplay">0:00</span>
</div>"""

# 3. Remove JS
js_to_remove = """  // Timer
  let startTime = Date.now();
  setInterval(() => {
    let elapsed = Math.floor((Date.now() - startTime) / 1000);
    let m = Math.floor(elapsed / 60);
    let s = elapsed % 60;
    document.getElementById('timerDisplay').textContent = m + ':' + String(s).padStart(2, '0');
  }, 1000);"""

# Replace with unix newlines first, then try with windows newlines
for remove_str in [css_to_remove, html_to_remove, js_to_remove]:
    if remove_str in html:
        html = html.replace(remove_str, "")
    elif remove_str.replace('\n', '\r\n') in html:
        html = html.replace(remove_str.replace('\n', '\r\n'), "")

with open(target, 'w', encoding='utf-8') as f:
    f.write(html)

print("Timer removed from survey.html")
