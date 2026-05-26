import re
with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\ml\ml_pipeline.py", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('✓', 'OK')

with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\ml\ml_pipeline.py", 'w', encoding='utf-8') as f:
    f.write(text)
