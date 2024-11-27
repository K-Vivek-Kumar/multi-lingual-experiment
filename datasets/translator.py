from deep_translator import GoogleTranslator
import os


def chunk_text(text, max_length=4000):
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def translate_essay(input_file, output_dir):
    languages = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "zh-CN": "Chinese",
        "ja": "Japanese",
        "de": "German",
        "it": "Italian",
        "ru": "Russian",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
    }

    with open(input_file, "r", encoding="utf-8") as file:
        essay = file.read()

    text_chunks = chunk_text(essay, max_length=4000)

    for lang_code, lang_name in languages.items():
        try:
            translated_essay = ""

            for chunk in text_chunks:
                try:

                    translated_essay += GoogleTranslator(
                        source="auto", target=lang_code
                    ).translate(chunk)
                except Exception as chunk_error:
                    print(f"Error translating chunk in {lang_name}: {chunk_error}")

            output_file = f"{output_dir}/essay_{lang_name}.txt"
            with open(output_file, "w", encoding="utf-8") as out_file:
                out_file.write(translated_essay)

            print(f"Translated essay saved to {output_file}")
        except Exception as e:
            print(f"Failed to translate to {lang_name}: {e}")


if __name__ == "__main__":
    input_file = "input.txt"
    output_dir = "translations"

    os.makedirs(output_dir, exist_ok=True)

    translate_essay(input_file, output_dir)
