import gdown

url = "https://drive.google.com/file/d/1vlhZ5c7abUKF8xXERIw6m9Te8fW7ohw3/view?usp=sharing"

file_id = url.split("/")[-2]

# Prefix URL
prefix = "https://drive.google.com/uc?/export=download&id="
gdown.download(prefix+file_id, "kidney-ct-scan-image.zip")
