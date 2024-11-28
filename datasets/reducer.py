import re


def split_sentences(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as infile:

        content = infile.read()

        sentences = re.split(r"(?<=\.)\s+", content.strip())

    with open(output_file, "w", encoding="utf-8") as outfile:
        for sentence in sentences:
            outfile.write(sentence.strip() + "\n")


if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "responses.txt"

    split_sentences(input_file, output_file)
    print(f"Responses have been written to {output_file}")
