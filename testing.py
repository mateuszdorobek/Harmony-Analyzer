from qualities import QUALITY_DICT, KEYS
import pandas as pd
import random
from tqdm.notebook import trange, tqdm


def generate_validation_file(file_name):
    test_list = []
    chord_types = list(QUALITY_DICT.keys())

    print('root_change test generation.')
    test_list.append(': root_change')
    for chord in chord_types:
        for note in range(len(KEYS)):
            for interval in range(1, len(KEYS)):
                for diff in range(1, len(KEYS)):
                    row = KEYS[note] + chord + " " + KEYS[(note + interval) % 12] + chord + " " + KEYS[
                        (diff + note) % 12] + chord + " " + KEYS[(diff + note + interval) % 12] + chord
                    test_list.append(row)

    print('chord_type_change test generation.')
    test_list.append(': chord_type_change')
    for chord_1 in chord_types:
        for chord_2 in chord_types:
            if chord_1 == chord_2:
                continue
            for note in range(len(KEYS)):
                for diff in range(1, len(KEYS)):
                    row = KEYS[note] + chord_1 + " " + KEYS[note] + chord_2 + " " + KEYS[
                        (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note) % 12] + chord_2
                    test_list.append(row)

    print('V-I_progression test generation.')
    test_list.append(': V-I_progression')
    chord_1 = '7'
    chord_2 = '^7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
            test_list.append(row)

    print('II-V_progression test generation.')
    test_list.append(': II-V_progression')
    chord_1 = '-7'
    chord_2 = '7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
            test_list.append(row)

    print('V/V-V_progression test generation.')
    test_list.append(': V/V-V_progression')
    chord = '7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord + " " + KEYS[(note + interval) % 12] + chord + " " + KEYS[
                (diff + note) % 12] + chord + " " + KEYS[(diff + note + interval) % 12] + chord
            test_list.append(row)

    print('less_common_progression test generation.')
    test_list.append(': less_common_progression')
    chord_types = chord_types[:10] + chord_types[13:]
    for chord_1 in chord_types:
        for chord_2 in chord_types:
            if chord_1 == chord_2:
                continue
            for note in range(len(KEYS)):
                for interval in range(1, len(KEYS)):
                    for diff in range(1, len(KEYS)):
                        if random.random() < 0.1:
                            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
                            test_list.append(row)
    print("Saving test file: " + file_name)
    pd.DataFrame(test_list).to_csv("data/validation/" + file_name, header=None, index=False)


if __name__ == "__main__":
    pass
    # generate_validation_file(file_name="test_chords.txt")