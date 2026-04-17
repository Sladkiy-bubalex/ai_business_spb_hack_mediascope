import os
import zipfile
print(os.listdir())
with zipfile.ZipFile('../mediahack.zip', 'r') as zip_ref:
    zip_ref.extractall('')