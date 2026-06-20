# Markoff Nickname Generator

A command-line nickname generator based on n-order Markov chains.

The generator learns character transition probabilities from a list of input names and produces new names that resemble the training data.

## Installation

Python 3.10+

Clone the repository:

```bash
git clone https://github.com/flacersko/markoff.git
cd markoff
```

No external dependencies are required.

## Input Format

Program expects a UTF-8 text file containing one name per line:

```text
Shadow
DarkKnight
NightWolf
Phantom
```

## Basic Usage

Generate a nickname:

```bash
python3 main.py --names-file names.txt
```

Generate multiple nicknames:

```bash
python3 main.py \
    --names-file names.txt \
    --count 10
```

## Examples

Generate names between 6 and 10 characters:

```bash
python3 main.py \
    --names-file names.txt \
    --min-length 6 \
    --max-length 10
```

Generate names starting with "s":

```bash
python3 main.py \
    --names-file names.txt \
    --start s
```

Generate names ending with "x":

```bash
python3 main.py \
    --names-file names.txt \
    --end x
```

Use beam search:

```bash
python3 main.py \
    --names-file names.txt \
    --beam 20
```

Generate CamelCase names:

```bash
python3 main.py \
    --names-file names.txt \
    --style-camel
```

## Parameters

| Parameter           | Description                 |
| ------------------- | --------------------------- |
| `--order`           | Markov chain order          |
| `--temp`            | Generation temperature      |
| `--beam`            | Beam search width           |
| `--count`           | Number of names to generate |
| `--min-length`      | Minimum length              |
| `--max-length`      | Maximum length              |
| `--length`          | Exact length                |
| `--start`           | Required starting character |
| `--end`             | Required ending character   |
| `--forbid-pattern`  | Regex pattern to reject     |
| `--forbid-letters`  | Forbidden characters        |
| `--style-camel`     | Capitalize first letter     |
| `--style-alternate` | Alternating letter case     |
| `--style-xx`        | Wrap name with xX / Xx      |
| `--add-digits`      | Append random digits        |
| `--prefix`          | Prefix string               |
| `--suffix`          | Suffix string               |

For full list, use:

```bash
python3 main.py
```

## How It Works

The generator builds an n-order Markov model from the provided dataset. During generation, the next character is selected according to the learned transition probabilities. Optional beam search can be used to explore multiple candidate sequences and improve output quality.

Additional filters may be applied to prevent excessive character repetition and repeated n-gram loops, but they don't work most of the times.

## License

MIT License

