"""Build the public, static GitHub Pages edition of the blog series."""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name


SERIES_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = SERIES_ROOT / "site"
SITE_URL = "https://adidror005.github.io/sequence-models-for-prediction"

# The source filenames predate the addition of the PyTorch pipeline article.
# Public numbering deliberately treats that foundation as Part 2.
SERIES = [
    (1, "01-sequence-models-for-prediction.md", "Foundations"),
    (2, "pytorch-data-pipeline-and-training-loop.md", "Foundations"),
    (3, "02-linear-forecaster.md", "Models"),
    (4, "03-mlp-forecaster.md", "Models"),
    (5, "04-lstm-forecaster.md", "Models"),
    (6, "05-gru-forecaster.md", "Models"),
    (7, "06-cnn1d-forecaster.md", "Models"),
    (8, "07-tcn-forecaster.md", "Models"),
    (9, "08-transformer-forecaster.md", "Models"),
    (10, "09-patch-transformer-forecaster.md", "Models"),
    (11, "10-nbeats-style-forecaster.md", "Models"),
    (12, "11-choosing-a-sequence-model.md", "Synthesis"),
    (13, "12-electricity-results-and-interpretation.md", "Case study"),
    (14, "13-calendar-and-lagged-features.md", "Case study"),
    (15, "14-designing-a-trustworthy-experiment.md", "Practice"),
    (16, "15-finance-direction-case-study.md", "Case study"),
]


def highlight_code(code: str, language: str, _attrs: str) -> str:
    try:
        lexer = get_lexer_by_name(language) if language else TextLexer()
    except Exception:
        lexer = TextLexer()
    return highlight(code, lexer, HtmlFormatter(cssclass="highlight"))


MARKDOWN = (
    MarkdownIt(
        "commonmark",
        {
            "html": True,
            "linkify": True,
            "typographer": True,
            "highlight": highlight_code,
        },
    )
    .enable("table")
    .enable("strikethrough")
)


def output_name(source_name: str) -> str:
    return str(Path(source_name).with_suffix(".html"))


def convert_link(href: str) -> str:
    if href == "README.md":
        return "index.html"
    if href == "PUBLICATION_PLAN.md":
        return "index.html"
    if href.endswith(".md") and not href.startswith(("http://", "https://")):
        return href[:-3] + ".html"
    return href


def strip_editorial_metadata(markdown: str) -> str:
    markdown = re.sub(
        r"^\*\*Series:\*\*\s*.+\n?",
        "",
        markdown,
        flags=re.MULTILINE,
    )
    markdown = re.sub(
        r"^\*\*Suggested Medium tags:\*\*\s*.+\n?",
        "",
        markdown,
        flags=re.MULTILINE,
    )
    markdown = re.sub(
        r"^\*\*(?:Series|Companion) navigation:\*\*.*\n?",
        "",
        markdown,
        flags=re.MULTILINE,
    )
    return markdown


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def social_meta(
    title: str,
    description: str,
    canonical: str,
    image_url: str | None = None,
    page_type: str = "article",
) -> str:
    if image_url is None:
        og_source = SERIES_ROOT / "assets" / "og.png"
        if og_source.exists():
            image_url = f"{SITE_URL}/og.png"
    image_meta = ""
    if image_url:
        image_meta = f"""
<meta property="og:image" content="{html.escape(image_url)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{html.escape(image_url)}">"""
    return f"""
<meta property="og:type" content="{html.escape(page_type)}">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(description)}">
<meta property="og:url" content="{html.escape(canonical)}">
<meta name="twitter:title" content="{html.escape(title)}">
<meta name="twitter:description" content="{html.escape(description)}">
{image_meta}"""


def parse_article(part: int, source_name: str, section: str) -> dict[str, object]:
    source = SERIES_ROOT / source_name
    rendered = MARKDOWN.render(strip_editorial_metadata(source.read_text(encoding="utf-8")))
    soup = BeautifulSoup(rendered, "html.parser")

    title_node = soup.find("h1")
    title = title_node.get_text(" ", strip=True) if title_node else source.stem
    if title_node:
        title_node.decompose()

    description = "A practical guide to sequence models for real forecasting problems."
    first_paragraph = soup.find("p")
    if first_paragraph and first_paragraph.find("em"):
        description = first_paragraph.get_text(" ", strip=True)
        first_paragraph.decompose()

    used_ids: set[str] = set()
    toc: list[tuple[str, str]] = []
    for heading in soup.find_all("h2"):
        text = heading.get_text(" ", strip=True)
        base = slugify(text)
        identifier = base
        suffix = 2
        while identifier in used_ids:
            identifier = f"{base}-{suffix}"
            suffix += 1
        used_ids.add(identifier)
        heading["id"] = identifier
        toc.append((identifier, text))

    for anchor in soup.find_all("a", href=True):
        anchor["href"] = convert_link(anchor["href"])
        if anchor["href"].startswith(("http://", "https://")):
            anchor["rel"] = "noopener noreferrer"

    for table in soup.find_all("table"):
        wrapper = soup.new_tag("div")
        wrapper["class"] = "table-scroll"
        table.wrap(wrapper)

    first_image = soup.find("img", src=True)
    social_image = None
    if first_image:
        image_source = str(first_image["src"])
        social_image = (
            image_source
            if image_source.startswith(("http://", "https://"))
            else f"{SITE_URL}/{image_source.lstrip('/')}"
        )

    return {
        "part": part,
        "section": section,
        "source": source_name,
        "html": output_name(source_name),
        "title": title,
        "description": description,
        "body": str(soup),
        "toc": toc,
        "social_image": social_image,
    }


def article_page(
    article: dict[str, object],
    previous: dict[str, object] | None,
    following: dict[str, object] | None,
) -> str:
    title = str(article["title"])
    description = str(article["description"])
    canonical = f"{SITE_URL}/{article['html']}"
    toc_items = "".join(
        f'<li><a href="#{html.escape(identifier)}">{html.escape(label)}</a></li>'
        for identifier, label in article["toc"]
    )
    previous_link = (
        f'<a class="pager-card previous" href="{previous["html"]}"><span>Previous</span><strong>{html.escape(str(previous["title"]))}</strong></a>'
        if previous
        else f'<a class="pager-card previous" href="index.html"><span>Series home</span><strong>Explore all {len(SERIES)} parts</strong></a>'
    )
    following_link = (
        f'<a class="pager-card next" href="{following["html"]}"><span>Next</span><strong>{html.escape(str(following["title"]))}</strong></a>'
        if following
        else (
            '<a class="pager-card next" href="future-stock-prediction-roadmap.html"><span>Next application</span><strong>Broader stock-prediction roadmap</strong></a>'
            if int(article["part"]) == len(SERIES)
            else '<a class="pager-card next" href="index.html"><span>Finished</span><strong>Return to the series</strong></a>'
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Sequence Models for Prediction</title>
<meta name="description" content="{html.escape(description)}">
<link rel="canonical" href="{canonical}">
<link rel="stylesheet" href="styles.css">
{social_meta(title, description, canonical, article.get('social_image'))}
</head>
<body>
<header class="site-header">
  <a class="brand" href="index.html"><span class="brand-mark">S</span><span>Sequence Models<br><small>for Prediction</small></span></a>
  <nav aria-label="Primary navigation"><a href="index.html#series">All articles</a><a href="index.html#about">About</a><a class="github-link" href="https://github.com/adidror005/sequence-models-for-prediction">View source</a></nav>
</header>
<div class="reading-progress" style="--progress: {int(article['part']) / len(SERIES) * 100:.2f}%"></div>
<main>
  <header class="article-hero">
    <a class="back-link" href="index.html">← Series home</a>
    <p class="eyebrow">Part {article['part']} of {len(SERIES)} · {html.escape(str(article['section']))}</p>
    <h1>{html.escape(title)}</h1>
    <p class="dek">{html.escape(description)}</p>
  </header>
  <div class="reading-layout">
    <article class="article-content">{article['body']}</article>
    <aside class="toc" aria-label="On this page"><p>On this page</p><ol>{toc_items}</ol></aside>
  </div>
  <nav class="article-pager" aria-label="Article navigation">{previous_link}{following_link}</nav>
</main>
<footer class="site-footer"><p>Sequence Models for Prediction · Open code, reproducible experiments, honest baselines.</p><a href="index.html">Series index</a></footer>
</body>
</html>"""


def index_page(articles: list[dict[str, object]]) -> str:
    cards = []
    for article in articles:
        cards.append(
            f"""<article class="series-card">
  <div class="card-meta"><span>Part {article['part']:02d}</span><span>{html.escape(str(article['section']))}</span></div>
  <h2><a href="{article['html']}">{html.escape(str(article['title']))}</a></h2>
  <p>{html.escape(str(article['description']))}</p>
  <a class="read-link" href="{article['html']}">Read article <span aria-hidden="true">→</span></a>
</article>"""
        )
    title = "Sequence Models for Prediction"
    description = "A practical, code-complete series about how modern sequence models learn from ordered data—and when simpler baselines still win."
    canonical = f"{SITE_URL}/"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="canonical" href="{canonical}">
<link rel="stylesheet" href="styles.css">
{social_meta(title, description, canonical, f'{SITE_URL}/og.png', page_type='website')}
</head>
<body class="home">
<header class="site-header home-header">
  <a class="brand" href="index.html"><span class="brand-mark">S</span><span>Sequence Models<br><small>for Prediction</small></span></a>
  <nav aria-label="Primary navigation"><a href="#series">The series</a><a href="#about">About</a><a class="github-link" href="https://github.com/adidror005/sequence-models-for-prediction">View source</a></nav>
</header>
<main>
  <section class="home-hero">
    <div class="hero-copy">
      <p class="eyebrow">A practical forecasting series</p>
      <h1>Sequence models,<br><em>without the mystique.</em></h1>
      <p class="hero-dek">From linear windows and recurrent memory to convolutions, Transformers, and N-BEATS—built in PyTorch, tested on electricity demand and one-minute META direction, and judged against strong simple baselines.</p>
      <div class="hero-actions"><a class="button primary-button" href="{articles[0]['html']}">Start with Part 1</a><a class="button text-button" href="#series">Browse the series ↓</a></div>
    </div>
    <div class="sequence-visual" aria-hidden="true">
      <div class="visual-label">past window</div>
      <div class="signal signal-past"></div>
      <div class="model-node"><span>f</span><small>model</small></div>
      <div class="signal signal-future"></div>
      <div class="visual-label future-label">forecast</div>
    </div>
  </section>
  <section class="proof-strip" aria-label="Series highlights"><div><strong>{len(articles)}</strong><span>focused parts</span></div><div><strong>9</strong><span>model families</span></div><div><strong>2</strong><span>real-data case studies</span></div><div><strong>1</strong><span>honest CatBoost check</span></div></section>
  <section id="about" class="about-section">
    <p class="eyebrow">The central question</p>
    <h2>What should a forecasting model learn—and what should we hand it?</h2>
    <div class="about-grid"><p>This series first holds the input fixed and compares how different architectures process the same history. Only then does it ask whether calendar variables, explicit lags, differences, and rolling statistics add useful information.</p><p>The uncomfortable benchmark stays visible throughout: sophisticated neural networks often lose to gradient boosting on a carefully designed time-series table. Complexity has to earn its place.</p></div>
  </section>
  <section id="series" class="series-section">
    <div class="section-heading"><div><p class="eyebrow">Read in order or jump in</p><h2>The complete series</h2></div><p>Foundations first. Individual algorithms next. Electricity and finance evidence at the end.</p></div>
    <div class="series-grid">{''.join(cards)}</div>
  </section>
  <section class="coming-next">
    <div><p class="eyebrow">Beyond the first finance case</p><h2>Stock prediction—with stricter rules.</h2><p>Part 16 reports the saved one-minute META experiment. The broader roadmap adds multiple assets, walk-forward evaluation, realistic baselines, transaction assumptions, and economic tests.</p></div>
    <a class="button light-button" href="future-stock-prediction-roadmap.html">See what comes next →</a>
  </section>
</main>
<footer class="site-footer"><p>Sequence Models for Prediction · Open code, reproducible experiments, honest baselines.</p><a href="https://github.com/adidror005/sequence-models-for-prediction">Source on GitHub</a></footer>
</body>
</html>"""


def roadmap_page() -> str:
    source = SERIES_ROOT / "future-stock-prediction-roadmap.md"
    rendered = MARKDOWN.render(source.read_text(encoding="utf-8"))
    soup = BeautifulSoup(rendered, "html.parser")
    title_node = soup.find("h1")
    title = title_node.get_text(" ", strip=True) if title_node else "Stock Prediction Roadmap"
    if title_node:
        title_node.decompose()
    for anchor in soup.find_all("a", href=True):
        anchor["href"] = convert_link(anchor["href"])
    description = "A roadmap for extending the first META case study into a broader, cost-aware stock-prediction experiment."
    canonical = f"{SITE_URL}/future-stock-prediction-roadmap.html"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)} — Sequence Models for Prediction</title><meta name="description" content="{description}"><link rel="canonical" href="{canonical}"><link rel="stylesheet" href="styles.css">{social_meta(title, description, canonical)}</head><body><header class="site-header"><a class="brand" href="index.html"><span class="brand-mark">S</span><span>Sequence Models<br><small>for Prediction</small></span></a><nav aria-label="Primary navigation"><a href="index.html#series">All articles</a><a href="index.html#about">About</a><a class="github-link" href="https://github.com/adidror005/sequence-models-for-prediction">View source</a></nav></header><main><header class="article-hero"><a class="back-link" href="index.html">← Series home</a><p class="eyebrow">Future application · Research roadmap</p><h1>{html.escape(title)}</h1><p class="dek">{description}</p></header><div class="reading-layout roadmap-layout"><article class="article-content">{str(soup)}</article></div><nav class="article-pager"><a class="pager-card previous" href="15-finance-direction-case-study.html"><span>Previous</span><strong>One-minute META case study</strong></a><a class="pager-card next" href="index.html"><span>Series</span><strong>Return to all articles</strong></a></nav></main><footer class="site-footer"><p>Sequence Models for Prediction · Open code, reproducible experiments, honest baselines.</p><a href="index.html">Series index</a></footer></body></html>"""


STYLES = r"""
:root {
  --ink: #14251f;
  --muted: #5f6f68;
  --paper: #fbfaf5;
  --cream: #f0ecdf;
  --green: #174c3c;
  --green-2: #256c56;
  --lime: #ddec72;
  --rust: #d96c42;
  --line: #d9ded7;
  --white: #fffef9;
  color-scheme: light;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; color: var(--ink); background: var(--paper); font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
a { color: inherit; }
.site-header { min-height: 76px; padding: 14px max(24px, calc((100vw - 1200px) / 2)); display: flex; align-items: center; justify-content: space-between; gap: 24px; border-bottom: 1px solid var(--line); background: rgba(251,250,245,.96); }
.brand { display: inline-flex; gap: 10px; align-items: center; color: var(--ink); font-size: 15px; font-weight: 760; line-height: 1.05; text-decoration: none; letter-spacing: -.01em; }
.brand small { color: var(--muted); font-size: 11px; font-weight: 650; letter-spacing: .04em; text-transform: uppercase; }
.brand-mark { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 50%; color: var(--lime); background: var(--green); font-family: Georgia, serif; font-size: 22px; font-style: italic; }
.site-header nav { display: flex; gap: 28px; align-items: center; font-size: 14px; font-weight: 650; }
.site-header nav a { text-decoration: none; }
.site-header nav a:hover { color: var(--green-2); }
.github-link { padding: 9px 15px; border: 1px solid var(--ink); border-radius: 999px; }
.home-hero { min-height: 650px; padding: 88px max(24px, calc((100vw - 1200px) / 2)); display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(360px, .75fr); gap: 72px; align-items: center; background: var(--green); color: var(--white); overflow: hidden; }
.eyebrow { margin: 0 0 18px; color: var(--rust); font-size: 12px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.home-hero .eyebrow { color: var(--lime); }
.home-hero h1 { max-width: 800px; margin: 0; font-family: Georgia, Cambria, serif; font-size: clamp(58px, 7.3vw, 104px); font-weight: 500; line-height: .95; letter-spacing: -.055em; }
.home-hero h1 em { color: var(--lime); font-weight: 400; }
.hero-dek { max-width: 720px; margin: 30px 0 0; color: #e0e9e4; font-family: Georgia, Cambria, serif; font-size: 22px; line-height: 1.55; }
.hero-actions { margin-top: 38px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
.button { display: inline-flex; align-items: center; justify-content: center; min-height: 48px; padding: 0 22px; border-radius: 999px; font-size: 14px; font-weight: 760; text-decoration: none; }
.primary-button { color: var(--green); background: var(--lime); }
.text-button { color: var(--white); }
.sequence-visual { position: relative; min-height: 360px; }
.sequence-visual::before { content: ""; position: absolute; inset: 5% -34% auto auto; width: 410px; height: 410px; border: 1px solid #ffffff26; border-radius: 50%; }
.sequence-visual::after { content: ""; position: absolute; inset: auto auto -22% 10%; width: 240px; height: 240px; border: 1px solid #ddec7242; border-radius: 50%; }
.model-node { position: absolute; z-index: 2; left: 48%; top: 46%; width: 118px; height: 118px; display: grid; place-items: center; border-radius: 28px; color: var(--green); background: var(--lime); transform: translate(-50%,-50%) rotate(4deg); box-shadow: 0 28px 80px #071d1666; }
.model-node span { font-family: Georgia, serif; font-size: 56px; font-style: italic; line-height: .7; }
.model-node small { margin-top: -25px; font-size: 10px; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
.signal { position: absolute; top: 45%; height: 92px; border-top: 3px solid var(--lime); transform: translateY(-50%) skewY(-12deg); opacity: .9; }
.signal::before, .signal::after { content: ""; position: absolute; width: 40%; border-top: 3px solid var(--lime); transform: skewY(32deg); }
.signal::before { top: 30px; left: 16%; }.signal::after { top: -38px; right: 4%; }
.signal-past { left: -5%; width: 49%; }.signal-future { right: -11%; width: 39%; border-style: dashed; opacity: .58; }
.visual-label { position: absolute; top: 22%; left: 3%; color: #b7ccc3; font-size: 11px; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
.future-label { left: auto; right: 4%; }
.proof-strip { padding: 28px max(24px, calc((100vw - 1200px) / 2)); display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; color: var(--green); background: var(--lime); }
.proof-strip div { display: flex; align-items: baseline; gap: 10px; }.proof-strip strong { font-family: Georgia, serif; font-size: 34px; }.proof-strip span { font-size: 13px; font-weight: 720; }
.about-section, .series-section { padding: 110px max(24px, calc((100vw - 1200px) / 2)); }
.about-section h2, .section-heading h2, .coming-next h2 { max-width: 900px; margin: 0; font-family: Georgia, Cambria, serif; font-size: clamp(40px, 5vw, 68px); font-weight: 500; line-height: 1.06; letter-spacing: -.035em; }
.about-grid { max-width: 930px; margin: 46px 0 0 auto; display: grid; grid-template-columns: repeat(2, 1fr); gap: 54px; color: var(--muted); font-family: Georgia, serif; font-size: 20px; line-height: 1.65; }
.series-section { background: var(--cream); }
.section-heading { display: flex; justify-content: space-between; align-items: end; gap: 48px; margin-bottom: 54px; }
.section-heading > p { max-width: 390px; margin: 0; color: var(--muted); line-height: 1.6; }
.series-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
.series-card { min-height: 330px; padding: 28px; display: flex; flex-direction: column; border: 1px solid #d8d3c5; border-radius: 18px; background: var(--paper); transition: transform .2s ease, box-shadow .2s ease; }
.series-card:hover { transform: translateY(-4px); box-shadow: 0 20px 42px #16312815; }
.card-meta { display: flex; justify-content: space-between; gap: 12px; color: var(--green-2); font-size: 11px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.series-card h2 { margin: 34px 0 16px; font-family: Georgia, serif; font-size: 28px; font-weight: 500; line-height: 1.15; letter-spacing: -.02em; }
.series-card h2 a { text-decoration: none; }.series-card p { margin: 0 0 28px; color: var(--muted); line-height: 1.55; }
.read-link { margin-top: auto; color: var(--green); font-size: 13px; font-weight: 800; text-decoration: none; }.read-link span { color: var(--rust); }
.coming-next { padding: 90px max(24px, calc((100vw - 1200px) / 2)); display: flex; align-items: end; justify-content: space-between; gap: 60px; color: var(--white); background: var(--rust); }
.coming-next > div { max-width: 850px; }.coming-next .eyebrow { color: #ffe8df; }.coming-next p:last-child { max-width: 690px; font-size: 18px; line-height: 1.6; }
.light-button { flex: 0 0 auto; color: var(--rust); background: var(--white); }
.site-footer { padding: 30px max(24px, calc((100vw - 1200px) / 2)); display: flex; justify-content: space-between; gap: 24px; color: #bcd0c7; background: #0e2f25; font-size: 13px; }
.site-footer a { color: var(--lime); }
.reading-progress { height: 4px; background: linear-gradient(to right, var(--rust) 0 var(--progress), var(--cream) var(--progress)); }
.article-hero { padding: 92px max(24px, calc((100vw - 1000px) / 2)) 68px; background: var(--green); color: var(--white); }
.article-hero .back-link { display: inline-block; margin-bottom: 64px; color: #d3e0da; font-size: 13px; text-decoration: none; }
.article-hero .eyebrow { color: var(--lime); }.article-hero h1 { max-width: 940px; margin: 0; font-family: Georgia, serif; font-size: clamp(48px, 6vw, 78px); font-weight: 500; line-height: 1.03; letter-spacing: -.04em; }
.article-hero .dek { max-width: 790px; margin: 28px 0 0; color: #d8e4df; font-family: Georgia, serif; font-size: 23px; line-height: 1.5; }
.reading-layout { max-width: 1120px; margin: 0 auto; padding: 80px 24px 110px; display: grid; grid-template-columns: minmax(0, 760px) 250px; gap: 80px; align-items: start; }
.roadmap-layout { display: block; max-width: 808px; }
.article-content { font-family: Georgia, Cambria, "Times New Roman", serif; font-size: 20px; line-height: 1.72; min-width: 0; }
.article-content h2, .article-content h3 { scroll-margin-top: 24px; color: var(--ink); font-family: Inter, ui-sans-serif, sans-serif; letter-spacing: -.025em; }
.article-content h2 { margin: 64px 0 18px; font-size: 34px; line-height: 1.18; }.article-content h3 { margin: 42px 0 14px; font-size: 24px; line-height: 1.25; }
.article-content p { margin: 0 0 25px; }.article-content ul, .article-content ol { margin: 0 0 26px; padding-left: 28px; }.article-content li { margin: 8px 0; }
.article-content a { color: var(--green-2); text-underline-offset: 3px; }.article-content blockquote { margin: 38px 0; padding: 4px 0 4px 25px; border-left: 4px solid var(--rust); color: #30443c; font-size: 25px; font-style: italic; }
.article-content img { display: block; max-width: 100%; height: auto; margin: 42px auto 14px; border-radius: 8px; }.article-content figure { margin: 42px 0; }.article-content figcaption { color: var(--muted); font-family: Inter, sans-serif; font-size: 13px; text-align: center; }
.article-content pre, .highlight { max-width: 100%; overflow-x: auto; margin: 30px 0; padding: 22px; border-radius: 10px; background: #eff1ed; font-size: 14px; line-height: 1.55; }
.article-content code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }.article-content :not(pre) > code { padding: 2px 5px; border-radius: 4px; background: #e9ece7; font-size: .82em; }
.table-scroll { max-width: 100%; overflow-x: auto; margin: 34px 0; border: 1px solid var(--line); border-radius: 10px; }.article-content table { width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 14px; }.article-content th, .article-content td { padding: 13px 15px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }.article-content th { color: white; background: var(--green); }.article-content tr:last-child td { border-bottom: 0; }
.toc { position: sticky; top: 28px; max-height: calc(100vh - 56px); overflow-y: auto; padding: 20px 0 20px 22px; border-left: 1px solid var(--line); font-size: 12px; }.toc > p { margin: 0 0 15px; color: var(--ink); font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }.toc ol { margin: 0; padding: 0; list-style: none; }.toc li { margin: 0 0 10px; }.toc a { color: var(--muted); line-height: 1.35; text-decoration: none; }.toc a:hover { color: var(--rust); }
.article-pager { max-width: 1120px; margin: 0 auto 110px; padding: 0 24px; display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }.pager-card { min-height: 150px; padding: 24px; display: flex; flex-direction: column; justify-content: space-between; border: 1px solid var(--line); border-radius: 16px; text-decoration: none; }.pager-card span { color: var(--rust); font-size: 11px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }.pager-card strong { max-width: 430px; font-family: Georgia, serif; font-size: 23px; font-weight: 500; line-height: 1.2; }.pager-card.next { text-align: right; align-items: end; }.pager-card:hover { border-color: var(--green-2); background: var(--cream); }
""" + HtmlFormatter(cssclass="highlight").get_style_defs(".highlight") + r"""
@media (max-width: 900px) {
  .home-hero { grid-template-columns: 1fr; padding-top: 70px; }.sequence-visual { min-height: 270px; }.proof-strip { grid-template-columns: 1fr 1fr; }.series-grid { grid-template-columns: 1fr 1fr; }.reading-layout { grid-template-columns: 1fr; }.toc { display: none; }
}
@media (max-width: 640px) {
  .site-header { align-items: flex-start; }.site-header nav { gap: 14px; font-size: 12px; }.site-header nav a:not(.github-link) { display: none; }.github-link { padding: 8px 11px; }
  .home-hero { min-height: auto; padding-top: 64px; gap: 32px; }.home-hero h1 { font-size: 54px; }.hero-dek { font-size: 19px; }.sequence-visual { min-height: 230px; }.proof-strip { grid-template-columns: 1fr 1fr; gap: 18px; }.proof-strip div { display: block; }.proof-strip span { display: block; }
  .about-section, .series-section { padding-top: 74px; padding-bottom: 74px; }.about-grid, .series-grid { grid-template-columns: 1fr; }.about-grid { gap: 10px; margin-top: 30px; }.section-heading, .coming-next { align-items: flex-start; flex-direction: column; }.series-card { min-height: 280px; }.coming-next { padding-top: 72px; padding-bottom: 72px; }
  .article-hero { padding-top: 60px; }.article-hero .back-link { margin-bottom: 44px; }.article-hero h1 { font-size: 46px; }.article-hero .dek { font-size: 20px; }.reading-layout { padding-top: 54px; }.article-content { font-size: 18px; }.article-content h2 { font-size: 29px; }.article-content pre, .highlight { padding: 16px; font-size: 12px; }.article-pager { grid-template-columns: 1fr; }.site-footer { flex-direction: column; }
}
"""


def build(limit: int | None = None) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SERIES_ROOT / "assets", OUTPUT_ROOT / "assets", dirs_exist_ok=True)
    shutil.copytree(SERIES_ROOT / "code", OUTPUT_ROOT / "code", dirs_exist_ok=True)
    og_source = SERIES_ROOT / "assets" / "og.png"
    if og_source.exists():
        shutil.copy2(og_source, OUTPUT_ROOT / "og.png")

    selected = SERIES[:limit] if limit else SERIES
    articles = [parse_article(*entry) for entry in selected]
    for index, article in enumerate(articles):
        previous = articles[index - 1] if index else None
        following = articles[index + 1] if index + 1 < len(articles) else None
        (OUTPUT_ROOT / str(article["html"])).write_text(
            article_page(article, previous, following), encoding="utf-8"
        )

    (OUTPUT_ROOT / "index.html").write_text(index_page(articles), encoding="utf-8")
    (OUTPUT_ROOT / "future-stock-prediction-roadmap.html").write_text(
        roadmap_page(), encoding="utf-8"
    )
    (OUTPUT_ROOT / "styles.css").write_text(STYLES, encoding="utf-8")
    (OUTPUT_ROOT / ".nojekyll").write_text("", encoding="utf-8")
    (OUTPUT_ROOT / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n",
        encoding="utf-8",
    )
    sitemap_pages = ["", *(str(article["html"]) for article in articles), "future-stock-prediction-roadmap.html"]
    sitemap = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">" + "".join(
        f"<url><loc>{SITE_URL}/{page}</loc></url>" for page in sitemap_pages
    ) + "</urlset>\n"
    (OUTPUT_ROOT / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    print(f"Built {len(articles)} public articles in {OUTPUT_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build(args.limit)


if __name__ == "__main__":
    main()
