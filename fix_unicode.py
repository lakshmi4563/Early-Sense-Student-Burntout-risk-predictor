import re
with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\ml\ml_pipeline.py", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('→', '->')
text = text.replace('Σ', 'Sigma')
text = text.replace('φ', 'phi')

with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\ml\ml_pipeline.py", 'w', encoding='utf-8') as f:
    f.write(text)

with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\app.py", 'r', encoding='utf-8') as f:
    app_text = f.read()

app_text = app_text.replace('→', '->')

with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\app.py", 'w', encoding='utf-8') as f:
    f.write(app_text)
