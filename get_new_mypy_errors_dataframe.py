# https://chat.openai.com/share/c25264c9-b2de-4286-b4a8-10a7399c26b0
# https://chat.openai.com/share/790c1b6d-60b3-4c15-b99e-0356d8e2034a

import argparse
import csv
import difflib
import logging


def read_mypy_output_dataframe(file_name: str) -> tuple[list[tuple[str]], list[str]]:
    rows_without_id_and_line_number: list[tuple[str]] = []
    line_numbers: list[str] = []

    with open(file_name, 'r', newline='') as fp:
        reader = csv.reader(fp)
        for row in reader:
            row_without_id_and_line_number = (row[1], *row[3:])
            line_number = row[2]
            
            rows_without_id_and_line_number.append(row_without_id_and_line_number)
            line_numbers.append(line_number)
            
    return rows_without_id_and_line_number, line_numbers


def find_indices_of_header_and_new_lines(
    original_rows_without_id_and_line_number: list[tuple[str]],
    new_rows_without_id_and_line_number: list[tuple[str]]
) -> list[int]:
    matcher = difflib.SequenceMatcher(
        None,
        original_rows_without_id_and_line_number,
        new_rows_without_id_and_line_number
    )
    opcodes = matcher.get_opcodes()
    # Include the CSV header that would otherwise be excluded
    indices_of_new_lines = [0]
    for (
        tag,
        start_index_in_first_list,
        end_index_in_first_list,
        start_index_in_second_list,
        end_index_in_second_list
    ) in opcodes:
        if tag == 'insert':
            indices_of_new_lines.extend(range(start_index_in_second_list, end_index_in_second_list))
    return indices_of_new_lines


if __name__ == '__main__':
    # Set up logging
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--before-type-weaving-mypy-output-dataframe', type=str, required=True)
    parser.add_argument('-a', '--after-type-weaving-mypy-output-dataframe', type=str, required=True)
    parser.add_argument('-n', '--new-mypy-errors-dataframe', type=str, required=True)
    args = parser.parse_args()

    before_type_weaving_mypy_output_dataframe: str = args.before_type_weaving_mypy_output_dataframe
    after_type_weaving_mypy_output_dataframe: str = args.after_type_weaving_mypy_output_dataframe
    new_mypy_errors_dataframe: str = args.new_mypy_errors_dataframe

    (
        mypy_output_dataframe_before_type_weaving_rows_without_id_and_line_number,
        mypy_output_dataframe_before_type_weaving_line_numbers
    ) = read_mypy_output_dataframe(before_type_weaving_mypy_output_dataframe)

    (
        mypy_output_dataframe_after_type_weaving_rows_without_id_and_line_number,
        mypy_output_dataframe_after_type_weaving_line_numbers
    ) = read_mypy_output_dataframe(after_type_weaving_mypy_output_dataframe)

    indices_of_header_and_new_lines = find_indices_of_header_and_new_lines(
        mypy_output_dataframe_before_type_weaving_rows_without_id_and_line_number,
        mypy_output_dataframe_after_type_weaving_rows_without_id_and_line_number
    )

    with open(new_mypy_errors_dataframe, 'w') as fp:
        writer = csv.writer(fp)
        for i in indices_of_header_and_new_lines:
            writer.writerow(
                (mypy_output_dataframe_after_type_weaving_line_numbers[i],) + mypy_output_dataframe_after_type_weaving_rows_without_id_and_line_number[i]
            )
