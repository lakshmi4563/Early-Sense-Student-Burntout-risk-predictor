import re
with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\run_tests.py", 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('✅', 'PASS')
text = text.replace('❌', 'FAIL')
text = text.replace('→', '->')

with open(r"e:\Onedrive\Desktop\Documents\6th Sem\MiniProject\Proj_Details\BurnoutGuard_MiniProject\run_tests.py", 'w', encoding='utf-8') as f:
    f.write(text)
