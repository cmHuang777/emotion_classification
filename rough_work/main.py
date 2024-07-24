import json
import spacy
import csv


# create an unique id for each conversation by concatenating filename and index in jsonl file
# given a json dialog object, returns the id in string
def dialog_to_id(dialog):
    return dialog["filename"] + dialog["index"]


nlp = spacy.load("en_core_web_sm")


# splits the text into a single paragraph by \n\n or paragraphs not exceeding token limit
def split_into_paragraphs(text, token_limit=100):
    paragraphs = text.split("\n\n")
    segments = []
    cur_segment = ""
    cur_len = 0

    for paragraph in paragraphs:
        k = len(nlp(paragraph))

        if k + cur_len <= token_limit:
            cur_segment += "\n\n"
            cur_segment += paragraph
            cur_len += k
        else:
            cur_len = k
            if cur_segment:
                segments.append(cur_segment.strip())
            cur_segment = paragraph

    if cur_segment:
        segments.append(cur_segment.strip())

    return segments


avg_conv_len = 6547 / 600
max_conv = 3000 / avg_conv_len
max_new_C_conv = max_conv / 6 + 1
max_new_D_conv = max_conv * 5 / 6 + 1
new_C_convo_size = 0
new_D_convo_size = 0
C_utterance_size = 0
D_utterance_size = 0
max_C_utterance_size = 3000 / 6
max_D_utterance_size = 3000 * 5 / 6

########################## FOR REFERENCES: ############################################
## new E will be the top 22 convo of qc_with_index which contains 305 utterances, this will be the new QC dataset
## C: 100 conv
## D: 500 conv


num_of_rows = 0
C_index = set()  # index of convo belonging to C
D_index = set()  # index of convo belonging to D
# get the index of convo in new E that belongs to each dataset
with open("qc_with_index.jsonl", "r", encoding="utf-8") as infile:
    for i, line in enumerate(infile):
        if i >= 22:  # only selects the top 22 convo for new E
            break
        dialogue = json.loads(line.strip())
        if not dialogue:
            continue
        if "singapore" in dialogue["filename"]:
            C_index.add(i)
        else:
            D_index.add(i)
print("Set C:", C_index)
print("Set D:", D_index)

# split into 2 sets, annotation and QC
# first count the number of utterances in new E that belongs to C and D respectively
# This uses the qc_with_token_limit_paragraph (matches the annotated version) and saves the top 22 convo into qc_top_22_convo.csv
with open("qc_with_token_limit_paragraph.csv", "r", encoding="utf-8") as infile, open(
    "qc_top_22_convo.csv", "w", newline="", encoding="utf-8"
) as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    cur_dialog_num = -1
    for row in reader:
        if row["speaker"] == "Dialogue22":  # new E only contains the top 22 convo
            break
        text = row["text"]
        writer.writerow(row)

        if not text:
            cur_dialog_num += 1
            if cur_dialog_num in C_index:
                new_C_convo_size += 1
            else:
                new_D_convo_size += 1
        else:
            if cur_dialog_num in C_index:
                C_utterance_size += 1
            else:
                D_utterance_size += 1

# print("Number of utterances in new E:", num_of_rows)
print("Number of convo in E belonging to C:", new_C_convo_size)
print("Number of convo in E belonging to D:", new_D_convo_size)
print("Number of utterances from C:", C_utterance_size)
print("Number of utterances from D:", D_utterance_size)


# now pick from shuffled_CandD_without_E until we reach max conv size for each group or max utterance size
# (top 22 of shuffled_CandD makes up new E, so the above file is obtained by deleting top 22 rows of shuffled_CandD)
cur_utterance_size = 305
MAX_UTERRANCE_SIZE = 3000

C_index = set()
D_index = set()
with open("shuffled_CandD_without_E.jsonl", "r", encoding="utf-8") as infile, open(
    "new_CandD_energy.csv", "w", newline="", encoding="utf-8"
) as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["speaker", "text", "emotion", "sentiment"])

    for i, line in enumerate(infile):
        if cur_utterance_size >= 3000 or (
            new_C_convo_size >= max_new_C_conv and new_D_convo_size >= max_new_D_conv
        ):
            break
        dialogue = json.loads(line.strip())
        if not dialogue:
            continue

        can_be_added = None
        if "singapore" in dialogue["filename"] and new_C_convo_size < max_new_C_conv:
            C_index.add(i)
            new_C_convo_size += 1
            can_be_added = "C"
        elif (
            "singapore" not in dialogue["filename"]
            and new_D_convo_size < max_D_utterance_size
        ):
            D_index.add(i)
            new_D_convo_size += 1
            can_be_added = "D"

        if not can_be_added:
            continue

        writer.writerow([f"Dialogue{i}", ""])
        speakers2id = {}
        nextId = 0
        for utterance in dialogue["conversation"]:
            user = utterance["speaker"]
            if user not in speakers2id:
                speakers2id[user] = nextId
                nextId += 1

            speaker = f"Speaker_{speakers2id[user]}"
            text = utterance["utterance"]
            segments = split_into_paragraphs(text)
            for segment in segments:
                cur_utterance_size += 1
                if can_be_added == "C":
                    C_utterance_size += 1
                else:
                    D_utterance_size += 1
                writer.writerow([speaker, segment, "", ""])

print("Number of utterances:", cur_utterance_size)
print("Number of convo belonging to C:", new_C_convo_size)
print("Number of convo belonging to D:", new_D_convo_size)
print("Number of utterances from C:", C_utterance_size)
print("Number of utterances from D:", D_utterance_size)
print("C_index:", sorted(list(C_index)))
print("D_index:", sorted(list(D_index)))
