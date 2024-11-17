import os
from pytube import Playlist, YouTube
from moviepy.editor import AudioFileClip

def download_playlist_as_mp3(playlist_url, output_folder="downloads"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    playlist = Playlist(playlist_url)
    print(f"Found {len(playlist.video_urls)} videos in the playlist.")

    for video_url in playlist.video_urls:
        try:
            # Load the video
            yt = YouTube(video_url)
            title = yt.title if yt.title else "Unknown Title"
            print(f"Downloading: {title}")

            # Download the audio stream
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_file = audio_stream.download(output_path=output_folder)

            # Convert to MP3
            mp3_file = os.path.splitext(audio_file)[0] + ".mp3"
            with AudioFileClip(audio_file) as audio:
                audio.write_audiofile(mp3_file)

            os.remove(audio_file)
            print(f"Saved as MP3: {mp3_file}")

        except Exception as e:
            print(f"Error downloading {video_url}: {e}")

    print("All videos have been processed.")

if __name__ == "__main__":
    #download_playlist_as_mp3('https://www.youtube.com/playlist?list=PLKCcFRnEHHgWaIo6AlSfdisOHf6NR0LqF')
    yt = YouTube('https://www.youtube.com/watch?v=tn2-PUq1Z84')
    print(yt.title)
