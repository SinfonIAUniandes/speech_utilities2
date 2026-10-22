"""
Build a robot's LLM system prompt from the text files in 'data'.

Layout::

    data/<robot>_info.txt           persona, Spanish
    data/english_<robot>_info.txt   persona, English
    data/tools_info.txt             shared text, Spanish
    data/english_tools_info.txt     shared text, English

A persona pulls in shared text with a '{placeholder}', resolved in the
persona's own language: mixing languages makes small models answer
erratically. No ROS imports; the caller passes the directory in.
"""

import logging
from pathlib import Path
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = 'es'

#: Languages that take a file name prefix. Spanish is the unprefixed default,
#: which is the convention the inherited text files already use.
LANGUAGE_PREFIXES = {'es': '', 'en': 'english_'}

#: Placeholder name -> file stem in the data directory.
FRAGMENT_FILES = {'info_herramientas': 'tools_info'}

#: Appended last so the robot answers in one predictable language. Without
#: it a small model follows whichever language pulls hardest; see the wiki.
LANGUAGE_DIRECTIVES = {
    'es': 'Responde SIEMPRE en español, sin importar el idioma en que te escriban.',
    'en': 'ALWAYS answer in English, regardless of the language you are addressed in.',
}

#: Spoken when the model returns nothing. User-facing, so it stays in the
#: robot's language.
FALLBACK_ANSWERS = {
    'es': 'Lo siento, no pude generar una respuesta en este momento.',
    'en': 'Sorry, I could not produce an answer right now.',
}

_PERSONA_SUFFIX = '_info.txt'

_PLACEHOLDER_RE = re.compile(r'\{([A-Za-z_][A-Za-z0-9_]*)\}')


class PromptError(RuntimeError):
    """Raised when a usable system prompt cannot be produced."""


def normalise_language(language):
    """Return 'language' lowercased, or raise 'PromptError' if unsupported."""
    lang = (language or DEFAULT_LANGUAGE).strip().lower()
    if lang not in LANGUAGE_PREFIXES:
        known = ', '.join(sorted(LANGUAGE_PREFIXES))
        raise PromptError(f"Unsupported language '{language}'. Known languages: {known}")
    return lang


def fallback_answer(language=DEFAULT_LANGUAGE):
    """Return the sentence to speak when the model produces no answer."""
    return FALLBACK_ANSWERS.get(normalise_language(language), FALLBACK_ANSWERS[DEFAULT_LANGUAGE])


def persona_path(data_dir, robot_name, language=DEFAULT_LANGUAGE):
    """Return the persona file path for 'robot_name' in 'language'."""
    prefix = LANGUAGE_PREFIXES[normalise_language(language)]
    robot = (robot_name or '').strip().lower()
    return Path(data_dir) / f'{prefix}{robot}{_PERSONA_SUFFIX}'


def available_personas(data_dir):
    """
    Return the sorted '(robot, language)' pairs found in 'data_dir'.

    Shared fragments are skipped: they share the suffix but are not personas.
    """
    fragments = {
        f'{prefix}{stem}.txt'
        for stem in FRAGMENT_FILES.values()
        for prefix in LANGUAGE_PREFIXES.values()
    }
    english = LANGUAGE_PREFIXES['en']
    pairs: List[Tuple[str, str]] = []
    for path in sorted(Path(data_dir).glob(f'*{_PERSONA_SUFFIX}')):
        if path.name in fragments:
            continue
        stem = path.name[: -len(_PERSONA_SUFFIX)]
        if stem.startswith(english):
            pairs.append((stem[len(english):], 'en'))
        else:
            pairs.append((stem, 'es'))
    return sorted(pairs)


def load_fragment(data_dir, name, language=DEFAULT_LANGUAGE):
    """
    Return the text a '{name}' placeholder refers to, in 'language'.

    Falls back to the unprefixed file when no translation exists. Raises
    'PromptError' if neither is present.
    """
    stem = FRAGMENT_FILES.get(name, name)
    base = Path(data_dir)
    candidates = [
        base / f'{LANGUAGE_PREFIXES[normalise_language(language)]}{stem}.txt',
        base / f'{stem}.txt',
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding='utf-8').strip()
    looked = ' and '.join(str(path) for path in candidates)
    raise PromptError(
        f"Unknown prompt fragment '{{{name}}}' for '{language}': looked for {looked}"
    )


def render_system_prompt(
    data_dir,
    robot_name,
    language=DEFAULT_LANGUAGE,
    extra_context='',
    enforce_language=True,
):
    """
    Build the system prompt for 'robot_name'.

    Sections in order: the persona, then 'extra_context' for anything that
    changes between runs, then the language directive, which goes last because
    that is where it has the most influence.

    Raises 'PromptError' rather than returning an empty prompt: a robot with no
    persona answers as a generic assistant, which is hard to notice at runtime.
    """
    base = Path(data_dir)
    if not base.is_dir():
        raise PromptError(f'Data directory does not exist: {base}')

    lang = normalise_language(language)
    path = persona_path(base, robot_name, lang)
    if not path.is_file():
        known = ', '.join(f'{robot} ({code})' for robot, code in available_personas(base))
        raise PromptError(
            f"No persona for robot '{robot_name}' in language '{lang}' "
            f'(looked for {path}). Available: {known or "none found"}'
        )

    rendered = _PLACEHOLDER_RE.sub(
        lambda match: load_fragment(base, match.group(1), lang),
        path.read_text(encoding='utf-8'),
    ).strip()
    if not rendered:
        raise PromptError(f'Persona file is empty: {path}')

    sections = [rendered]
    if (extra_context or '').strip():
        sections.append(extra_context.strip())
    if enforce_language:
        sections.append(LANGUAGE_DIRECTIVES[lang])

    prompt = '\n\n'.join(sections)
    logger.info(
        "System prompt ready for '%s' (%s): %d characters from %s",
        robot_name, lang, len(prompt), path.name,
    )
    return prompt
