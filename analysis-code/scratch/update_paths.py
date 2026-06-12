import os

replacements = {
    '/storage/raj.ayush/s2s-forecast-data/fuxi/output': '/storage/raj.ayush/s2s-forecast-data/fuxi/output',
    '/storage/raj.ayush/s2s-forecast-data/fuxi': '/storage/raj.ayush/s2s-forecast-data/fuxi',
    '/storage/raj.ayush/s2s-forecast-data/spire': '/storage/raj.ayush/s2s-forecast-data/spire',
    '/storage/raj.ayush/s2s-forecast-data/ecmwf': '/storage/raj.ayush/s2s-forecast-data/ecmwf',
    '/storage/raj.ayush/s2s-forecast-data/ncep': '/storage/raj.ayush/s2s-forecast-data/ncep',
    '/storage/raj.ayush/s2s-forecast-data/era5': '/storage/raj.ayush/s2s-forecast-data/era5',
    '/storage/raj.ayush/s2s-forecast-data': '/storage/raj.ayush/s2s-forecast-data',
}

workspaces = [
    '/home/raj.ayush/s2s/s2s_anlysis',
    '/home/raj.ayush/.gemini/antigravity-cli/brain/3948ed00-797e-4f08-bcda-cf28d16936ed'
]

for ws in workspaces:
    for root, dirs, files in os.walk(ws):
        if '.git' in root or '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.py') or f.endswith('.md'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r') as file:
                        content = file.read()
                    new_content = content
                    for old, new in replacements.items():
                        new_content = new_content.replace(old, new)
                    if new_content != content:
                        with open(filepath, 'w') as file:
                            file.write(new_content)
                        print(f"Updated paths in: {filepath}")
                except Exception as e:
                    pass
print("Path update complete.")
