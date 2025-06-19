# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import json
from google import genai
from google.genai import types
import mimetypes
import re
import struct
from dotenv import load_dotenv

load_dotenv()

def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # model = "gemini-2.5-pro"
    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""create me a 10 min podcast script about AI in finance and follow Structured output format 
                                     The interviewer will ask questions to guest seaker, and he will answer. 
                                     we have Jamie Dimon as our guest speaker, he is the CEO of JPMorgan Chase.
                                     The guest speaker should also use real life events to make the conversation more engaging. 
                                     """),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.ARRAY,
            items = genai.types.Schema(
                type = genai.types.Type.OBJECT,
                required = ["speaker", "text"],
                properties = {
                    "speaker": genai.types.Schema(
                        type = genai.types.Type.STRING,
                        enum = ["Speaker1", "Speaker2"],
                    ),
                    "text": genai.types.Schema(
                        type = genai.types.Type.STRING,
                    ),
                },
            ),
        ),
    )

    # for chunk in client.models.generate_content_stream(
    #     model=model,
    #     contents=contents,
    #     config=generate_content_config,
    # ):
    #     print(chunk.text, end="")

    full_json = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        full_json += chunk.text

    try:
        parsed_response = json.loads(full_json)
        plain_text_output = "Please read aloud the following in a podcast interview style:\n\n" + "\n".join(f"{item['speaker']}: {item['text']}" for item in parsed_response)
        # plain_text_output = "\n".join(f"{item['speaker']}: {item['text']}" for item in parsed_response)
        print(plain_text_output)

        # You can use the variable `plain_text_output` for further use
        return plain_text_output

    except json.JSONDecodeError as e:
        print("Failed to decode JSON response:", e)
        return None

if __name__ == "__main__":
    podcast_script = generate()
    # generate()





# Code to generate-speech from the generated script

def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate_audio():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # model = "gemini-2.5-pro-preview-tts"
    model = "gemini-2.5-flash-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=podcast_script),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Sulafat"
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Charon"
                            )
                        ),
                    ),
                ]
            ),
        ),
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            file_name = f"gemini_podcast_{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            print(chunk.text)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    generate_audio()


