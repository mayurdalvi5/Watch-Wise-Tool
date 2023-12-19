from flask import Flask, request, Response
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

app = Flask(__name__)

excluded_fillers = ['um', 'uh', 'like', 'youknow','a']

# Functions returns actual output i.e summary and hashtags
@app.get('/summary')
def summary_api():
    url = request.args.get('url', '')
    video_id = url.split('=')[1]
    transcript = get_transcript(video_id)
    summary = get_summary(transcript)
    hashtags = get_hashtags(summary)

    result = f"{summary}\n\n<b>Hashtags:</b>\n{', '.join(hashtags)}"
    return Response(result, content_type='text/html'), 200

# Funciton returns transcript of given url

def get_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([d['text'] for d in transcript_list])
    return transcript

# Function uses transcript to get summary of above transcript
def get_summary(transcript):
    summariser = pipeline('summarization', model='t5-small', framework='tf',max_length = 66)
    summary = ''
    max_length = 65

    for i in range(0, (len(transcript) // 1000) + 1):
        summary_text = summariser(transcript[i * 1000:(i + 1) * 1000], max_length=max_length)[0]['summary_text']
        summary = summary + summary_text + ' '
    return summary

# Genrate Hastages based on genrated summary
def get_hashtags(text, num_hashtags=10, min_length=4):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    excluded_fillers_lower = set(excluded_fillers + [word.lower() for word in excluded_fillers])
    filtered_words = [word.lower() for word in words if
                      len(word) > min_length and word.isalpha() and word.lower() not in stop_words and word.lower() not in excluded_fillers_lower]

    word_freq = Counter(filtered_words)

    top_hashtags = [f"#{word}" for word, _ in word_freq.most_common(num_hashtags)]
    return top_hashtags

if __name__ == '__main__':
    app.run()
