# Test audio

Place short music clips (mp3 / wav / flac) in this folder and point
`AUDIO_PATH` in [`../infer.py`](../infer.py) or the Gradio app
[`../app.py`](../app.py) to them. Any file `torchaudio` can decode will work;
mono / stereo at any sample rate is fine because MOSS-Music resamples to
16 kHz internally.

Suggested local filenames:

* `example.wav` - a short music clip for quick smoke tests.
* `test_song.mp3` - a song snippet for caption / lyrics tests.
* `test_instrumental.mp3` - an instrumental clip for chord / key / tempo tests.

Included sample clips in this repository:

* `tonghua.mp3`
* `keximeiruguo - 02.mp3`
* `zhishaohaiyouni-intro+verse1.mp3`

Other local test files can also be placed in this folder as needed.
