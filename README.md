
## Prepare environment
```
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

```


## Transcribe the audio
```
# the transcript is created at the same folder *.txt
python transcribe.py [AUDIO_FILE]

# start UI app
streamlit run app.py

```