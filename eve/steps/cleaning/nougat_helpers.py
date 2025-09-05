"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Script from here - https://github.com/facebookresearch/nougat/blob/main/nougat/postprocessing.py
"""


from typing import Union, List
import re
import os
from nltk.corpus import words
from multiprocessing import Pool
from functools import partial
from rapidfuzz.fuzz import ratio as ratio_perc


def ratio(*args, **kwargs):
    return ratio_perc(*args, **kwargs) / 100


reference_pattern = re.compile(r"^\* \[\d+\]", flags=re.M)


def markdown_compatible(s: str) -> str:
    """
    Make text compatible with Markdown formatting.

    This function makes various text formatting adjustments to make it compatible with Markdown.

    Args:
        s (str): The input text to be made Markdown-compatible.

    Returns:
        str: The Markdown-compatible text.
    """
    s = re.sub(
        r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$", r"\[\2 \\tag{\1}\]", s, flags=re.M
    )
    s = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$", r"\[\1 \\tag{\2}\]", s, flags=re.M
    )
    s = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\) (\\\[.+?\\\])$",
        r"\[\1 \\tag{\2}\] \3",
        s,
        flags=re.M,
    )
    s = s.replace(r"\. ", ". ")
    s = s.replace(r"\.}", ".}")
    s = s.replace(r"\. }", ". }")
    s = s.replace(r"\.\]", ".]")
    s = s.replace(r"\. ]", ". ]")
    s = re.sub(r"\\begin\{table\}\s*\\begin\{tabular\}(.*?)\\end\{tabular\}\s*\\end\{table\}", r"\n\\begin{table}\n\\begin{tabular}\1\\end{tabular}\n\\end{table}\n", s, flags=re.DOTALL)
    
    s = re.sub(r"([^\s])\$([^\$]*)\$", r"\1 $\2$", s)
    s = re.sub(r"\$([^\$]*)\$([^\s])", r"$\1$ \2", s)
    
    return s


def truncate_repetitions(generation: str, score_cutoff: float = 0.5, min_len: int = 30):
    """
    Truncate repetitions in the given generation.

    This function identifies and truncates repetitive content in the text.
    """
    try:
        sentences = generation.split(".")
        if len(sentences) < 3:
            return generation
        
        to_delete = set()
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sent_i = sentences[i].strip()
                sent_j = sentences[j].strip()
                
                if len(sent_i) < min_len or len(sent_j) < min_len:
                    continue
                    
                if ratio(sent_i, sent_j) > score_cutoff:
                    to_delete.add(j)
        
        new_sentences = [sent for i, sent in enumerate(sentences) if i not in to_delete]
        return ".".join(new_sentences)
    except Exception:
        return generation


def remove_numbers(lines: List[str]) -> List[str]:
    """Remove number patterns from lines."""
    clean_lines = []
    for line in lines:
        clean_line = re.sub(r'\[\d+\]', '', line)
        clean_line = re.sub(r'\d+\.', '', clean_line)
        clean_lines.append(clean_line.strip())
    return clean_lines


def get_slices(lines: List[str], clean_lines: List[str]) -> List[slice]:
    """Get slices of potentially hallucinated reference sections."""
    slices = []
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('## references'):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('##'):
                j += 1
            slices.append(slice(i, j))
    return slices


def remove_slice_from_lines(lines: List[str], clean_lines: List[str], sli: slice) -> str:
    """Remove slice from lines and return the removed text."""
    removed_text = '\n'.join(lines[sli])
    return removed_text


def remove_hallucinated_references(text: str) -> str:
    """
    Remove hallucinated or missing references from the text.

    This function identifies and removes references that are marked as missing or hallucinated
    from the input text.

    Args:
        text (str): The input text containing references.

    Returns:
        str: The text with hallucinated references removed.
    """
    lines = text.split("\n")
    if len(lines) == 0:
        return ""
    clean_lines = remove_numbers(lines)
    slices = get_slices(lines, clean_lines)
    to_delete = []
    for sli in slices:
        to_delete.append(remove_slice_from_lines(lines, clean_lines, sli))
    for to_delete in reversed(to_delete):
        text = text.replace(to_delete, "\n\n[MISSING_PAGE_POST]\n\n")
    text = re.sub(
        r"## References\n+\[MISSING_PAGE_POST(:\d+)?\]",
        "\n\n[MISSING_PAGE_POST\\1]",
        text,
    )
    return text


def postprocess_single(generation: str, markdown_fix: bool = True) -> str:
    """
    Postprocess a single generated text.

    Args:
        generation (str): The generated text to be postprocessed.
        markdown_fix (bool, optional): Whether to perform Markdown formatting fixes. Default is True.

    Returns:
        str: The postprocessed text.
    """
    generation = re.sub(
        r"(?:\n|^)#+ \d*\W? ?(.{100,})", r"\n\1", generation
    )
    generation = generation.strip()
    generation = generation.replace("\n* [leftmargin=*]\n", "\n")
    generation = re.sub(
        r"^#+ (?:\.?(?:\d|[ixv])+)*\s*(?:$|\n\s*)", "", generation, flags=re.M
    )
    lines = generation.split("\n")
    if (
        lines[-1].startswith("#")
        and lines[-1].lstrip("#").startswith(" ")
        and len(lines) > 1
    ):
        print("INFO: likely hallucinated title at the end of the page: " + lines[-1])
        generation = "\n".join(lines[:-1])
    generation = truncate_repetitions(generation)
    generation = remove_hallucinated_references(generation)
    generation = re.sub(
        r"^\* \[\d+\](\s?[A-W]\.+\s?){10,}.*$", "", generation, flags=re.M
    )
    generation = re.sub(r"^(\* \[\d+\])\[\](.*)$", r"\1\2", generation, flags=re.M)
    generation = re.sub(r"(^\w\n\n|\n\n\w$)", "", generation)
    generation = re.sub(
        r"([\s.,()])_([a-zA-Z0-9])__([a-zA-Z0-9]){1,3}_([\s.,:()])",
        r"\1\(\2_{\3}\)\4",
        generation,
    )
    generation = re.sub(
        r"([\s.,\d])_([a-zA-Z0-9])_([\s.,\d;])", r"\1\(\2\)\3", generation
    )
    generation = re.sub(
        r"(\nFootnote .*?:) (?:footnotetext|thanks):\W*(.*(?:\n\n|$))",
        r"\1 \2",
        generation,
    )
    generation = re.sub(r"\[FOOTNOTE:.+?\](.*?)\[ENDFOOTNOTE\]", "", generation)
    for match in reversed(
        list(
            re.finditer(
                r"(?:^)(-|\*)?(?!-|\*) ?((?:\d|[ixv])+ )?.+? (-|\*) (((?:\d|[ixv])+)\.(\d|[ixv]) )?.*(?:$)",
                generation,
                flags=re.I | re.M,
            )
        )
    ):
        start, stop = match.span()
        delim = match.group(3) + " "
        splits = match.group(0).split(delim)
        replacement = ""
        if match.group(1) is not None:
            splits = splits[1:]
            delim1 = match.group(1) + " "
        else:
            delim1 = ""
            continue
        pre, post = generation[:start], generation[stop:]
        for i, item in enumerate(splits):
            level = 0
            potential_numeral, _, rest = item.strip().partition(" ")
            if not rest:
                continue
            if re.match(
                r"^[\dixv]+((?:\.[\dixv])?)+$", potential_numeral, flags=re.I | re.M
            ):
                level = potential_numeral.count(".")

            replacement += (
                ("\n" if i > 0 else "")
                + ("\t" * level)
                + (delim if i > 0 or start == 0 else delim1)
                + item.strip()
            )
        if post == "":
            post = "\n"
        generation = pre + replacement + post

    if generation.endswith((".", "}")):
        generation += "\n\n"
    if re.match(r"[A-Z0-9,;:]$", generation):
        generation += " "
    elif generation.startswith(("#", "**", "\\begin")):
        generation = "\n\n" + generation
    elif generation.split("\n")[-1].startswith(("#", "Figure", "Table")):
        generation = generation + "\n\n"
    else:
        try:
            last_word = generation.split(" ")[-1]
            if last_word in words.words():
                generation += " "
        except LookupError:
            generation += " "
            import nltk

            nltk.download("words")
    for l in generation.split("\n"):
        if (
            l.count("\\begin{tabular}") > 15
            or l.count("\\multicolumn") > 60
            or l.count("&") > 400
        ):
            generation = generation.replace(l, "")
    generation = generation.replace(
        "\\begin{table} \\begin{tabular}", "\\begin{table}\n\\begin{tabular}"
    )
    generation = generation.replace(
        "\\end{tabular} \\end{table}", "\\end{tabular}\n\\end{table}"
    )
    generation = generation.replace("\\end{table} Tab", "\\end{table}\nTab")
    generation = re.sub(r"(^.+)\\begin{tab", r"\1\n\\begin{tab", generation, flags=re.M)

    generation = generation.replace(
        r"\begin{tabular}{l l}  & \\ \end{tabular}", ""
    ).replace("\\begin{tabular}{}\n\n\\end{tabular}", "")
    generation = generation.replace("\\begin{array}[]{", "\\begin{array}{")
    generation = re.sub(
        r"\\begin{tabular}{([clr ]){2,}}\s*[& ]*\s*(\\\\)? \\end{tabular}",
        "",
        generation,
    )
    generation = re.sub(r"(\*\*S\. A\. B\.\*\*\n+){2,}", "", generation)
    generation = re.sub(r"^#+( [\[\d\w])?$", "", generation, flags=re.M)
    generation = re.sub(r"^\.\s*$", "", generation, flags=re.M)
    generation = re.sub(r"\n{3,}", "\n\n", generation)
    if markdown_fix:
        return markdown_compatible(generation)
    else:
        return generation


def postprocess(
    generation: Union[str, List[str]], markdown_fix: bool = True
) -> Union[str, List[str]]:
    """
    Postprocess generated text or a list of generated texts.

    This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.

    Args:
        generation (Union[str, List[str]]): The generated text or a list of generated texts.
        markdown_fix (bool, optional): Whether to perform Markdown formatting fixes. Default is True.

    Returns:
        Union[str, List[str]]: The postprocessed text or list of postprocessed texts.
    """
    if type(generation) == list:
        if os.environ.get("NOUGAT_MULTIPROCESSING"):
            with Pool(int(os.environ.get("NOUGAT_MULTIPROCESSING"))) as p:
                return p.map(
                    partial(postprocess_single, markdown_fix=markdown_fix), generation
                )
        else:
            return [
                postprocess_single(s, markdown_fix=markdown_fix) for s in generation
            ]
    else:
        return postprocess_single(generation, markdown_fix=markdown_fix)
