"""Build copy-friendly HTML previews and table images for Medium publishing."""

from __future__ import annotations

import html
import json
import re
import shutil
import textwrap
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name


SERIES_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = SERIES_ROOT / "medium_exports"
TABLE_ROOT = OUTPUT_ROOT / "table_images"
ASSET_ROOT = OUTPUT_ROOT / "assets"
CODE_ROOT = OUTPUT_ROOT / "code"

BODY_FONT_PATHS = [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Helvetica.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]
BOLD_FONT_PATHS = [
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def load_font(paths: list[Path], size: int) -> ImageFont.FreeTypeFont:
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


BODY_FONT = load_font(BODY_FONT_PATHS, 25)
BOLD_FONT = load_font(BOLD_FONT_PATHS, 25)
SMALL_FONT = load_font(BODY_FONT_PATHS, 20)


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


def wrap_cell(value: str, width_pixels: int) -> list[str]:
    characters = max(8, int((width_pixels - 28) / 13))
    lines: list[str] = []
    for paragraph in value.splitlines() or [""]:
        wrapped = textwrap.wrap(
            paragraph,
            width=characters,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def render_table_image(
    headers: list[str],
    rows: list[list[str]],
    destination: Path,
) -> None:
    column_count = max(len(headers), *(len(row) for row in rows))
    normalized_headers = headers + [""] * (column_count - len(headers))
    normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]
    all_rows = [normalized_headers, *normalized_rows]

    widths = []
    for column in range(column_count):
        maximum = max(len(row[column]) for row in all_rows)
        widths.append(max(150, min(520, 55 + maximum * 13)))

    max_width = 1900
    total_width = sum(widths)
    if total_width > max_width:
        scale = max_width / total_width
        widths = [max(120, int(width * scale)) for width in widths]

    wrapped_rows = []
    row_heights = []
    for row in all_rows:
        wrapped = [wrap_cell(value, widths[index]) for index, value in enumerate(row)]
        wrapped_rows.append(wrapped)
        row_heights.append(max(58, max(len(lines) for lines in wrapped) * 31 + 22))

    margin = 32
    image = Image.new(
        "RGB",
        (sum(widths) + margin * 2, sum(row_heights) + margin * 2),
        "white",
    )
    draw = ImageDraw.Draw(image)
    y = margin

    for row_index, (wrapped, row_height) in enumerate(zip(wrapped_rows, row_heights)):
        x = margin
        background = "#17324D" if row_index == 0 else (
            "#F3F6F8" if row_index % 2 == 0 else "#FFFFFF"
        )
        foreground = "#FFFFFF" if row_index == 0 else "#202124"
        font = BOLD_FONT if row_index == 0 else BODY_FONT

        for column, lines in enumerate(wrapped):
            width = widths[column]
            draw.rectangle(
                (x, y, x + width, y + row_height),
                fill=background,
                outline="#CBD3DA",
                width=2,
            )
            draw.multiline_text(
                (x + 14, y + 11),
                "\n".join(lines),
                font=font,
                fill=foreground,
                spacing=5,
            )
            x += width
        y += row_height

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def convert_local_link(href: str) -> tuple[str, bool]:
    if href == "README.md":
        return "index.html", True
    if href == "PUBLICATION_PLAN.md":
        return "publication-plan.html", True
    if href.endswith(".md") and not href.startswith(("http://", "https://")):
        return href[:-3] + ".html", True
    return href, False


def process_article(source: Path, part: int) -> dict[str, object]:
    source_markdown = source.read_text(encoding="utf-8")
    tag_match = re.search(
        r"^\*\*Suggested Medium tags:\*\*\s*(.+)$",
        source_markdown,
        flags=re.MULTILINE,
    )
    suggested_tags = tag_match.group(1).strip() if tag_match else ""
    source_markdown = re.sub(
        r"^\*\*Suggested Medium tags:\*\*\s*.+\n?",
        "",
        source_markdown,
        flags=re.MULTILINE,
    )
    rendered = MARKDOWN.render(source_markdown)
    soup = BeautifulSoup(rendered, "html.parser")
    title_node = soup.find("h1")
    title = title_node.get_text(" ", strip=True) if title_node else source.stem
    local_links = 0
    for anchor in soup.find_all("a", href=True):
        converted, is_local = convert_local_link(anchor["href"])
        anchor["href"] = converted
        if is_local:
            anchor["class"] = [*anchor.get("class", []), "needs-live-url"]
            local_links += 1

    table_files: list[str] = []
    for table_index, table in enumerate(list(soup.find_all("table")), start=1):
        header_cells = table.find_all("th")
        headers = [cell.get_text(" ", strip=True) for cell in header_cells]
        body_rows = []
        tbody = table.find("tbody")
        row_nodes = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
        for row in row_nodes:
            body_rows.append(
                [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            )

        filename = f"{source.stem}-table-{table_index}.png"
        render_table_image(headers, body_rows, TABLE_ROOT / filename)
        table_files.append(f"table_images/{filename}")

        figure = soup.new_tag("figure")
        image = soup.new_tag("img")
        image["src"] = f"table_images/{filename}"
        image["alt"] = "Table with columns: " + ", ".join(headers)
        figure.append(image)
        caption = soup.new_tag("figcaption")
        caption.string = f"Table {table_index}. " + " · ".join(headers)
        figure.append(caption)
        table.replace_with(figure)

    normal_images = [image.get("src", "") for image in soup.find_all("img")]
    code_blocks = len(soup.find_all("pre"))
    article_html = str(soup)
    output_name = source.with_suffix(".html").name
    output = OUTPUT_ROOT / output_name
    output.write_text(
        page_template(
            title=title,
            article_html=article_html,
            source_name=source.name,
            suggested_tags=suggested_tags,
            code_blocks=code_blocks,
            table_files=table_files,
            image_files=normal_images,
            local_links=local_links,
        ),
        encoding="utf-8",
    )

    return {
        "part": part,
        "title": title,
        "source": source.name,
        "html": output_name,
        "suggested_tags": suggested_tags,
        "code_blocks": code_blocks,
        "table_images": table_files,
        "images": normal_images,
        "local_links_to_replace": local_links,
    }


def page_template(
    title: str,
    article_html: str,
    source_name: str,
    suggested_tags: str,
    code_blocks: int,
    table_files: list[str],
    image_files: list[str],
    local_links: int,
) -> str:
    asset_items = "".join(
        f'<li><a href="{html.escape(path)}" target="_blank">{html.escape(path)}</a></li>'
        for path in image_files
    ) or "<li>No images in this article.</li>"
    warning = (
        f"Replace {local_links} local series link(s) with live Medium URLs before publishing."
        if local_links
        else "No local series URLs were found."
    )
    tag_note = html.escape(suggested_tags) if suggested_tags else "No suggested tags found."
    formatter_css = HtmlFormatter(cssclass="highlight").get_style_defs(".highlight")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Medium export</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; color: #242424; background: #f5f5f2; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.toolbar {{ position: sticky; top: 0; z-index: 5; padding: 14px 20px; color: #fff; background: #17324d; box-shadow: 0 2px 8px #0002; }}
.toolbar-inner {{ max-width: 1080px; margin: auto; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
.toolbar strong {{ margin-right: auto; }}
button, .toolbar a {{ border: 1px solid #ffffff77; border-radius: 999px; padding: 8px 14px; color: white; background: transparent; font-weight: 600; text-decoration: none; cursor: pointer; }}
button.primary {{ color: #17324d; background: white; }}
.copy-status {{ width: 100%; color: #d9e6f2; font-size: 13px; }}
.instructions {{ max-width: 920px; margin: 28px auto 0; padding: 18px 24px; border-radius: 12px; background: #fff7d6; line-height: 1.5; }}
.instructions h2 {{ margin-top: 0; font-size: 20px; }}
.instructions code {{ background: #ffffff99; padding: 2px 5px; }}
.article-shell {{ max-width: 820px; margin: 28px auto 80px; padding: 56px 64px; background: white; box-shadow: 0 1px 8px #00000012; }}
.article-content {{ font-family: Georgia, Cambria, "Times New Roman", serif; font-size: 21px; line-height: 1.62; }}
.article-content h1 {{ margin: 0 0 12px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 46px; line-height: 1.12; letter-spacing: -0.025em; }}
.article-content h2 {{ margin: 50px 0 14px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 30px; line-height: 1.22; }}
.article-content h3 {{ margin: 34px 0 10px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 24px; line-height: 1.3; }}
.article-content p {{ margin: 0 0 24px; }}
.article-content ul, .article-content ol {{ margin: 0 0 24px; padding-left: 30px; }}
.article-content li {{ margin: 7px 0; }}
.article-content blockquote {{ margin: 32px 0; padding: 4px 0 4px 24px; border-left: 4px solid #242424; font-size: 25px; font-style: italic; }}
.article-content a {{ color: inherit; text-decoration-thickness: 1px; text-underline-offset: 3px; }}
.article-content a.needs-live-url {{ text-decoration-color: #d93025; text-decoration-style: wavy; }}
.article-content img {{ display: block; max-width: 100%; height: auto; margin: 36px auto 10px; }}
.article-content figure {{ margin: 38px 0; }}
.article-content figcaption {{ color: #6b6b6b; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; text-align: center; }}
.article-content pre, .highlight {{ overflow-x: auto; margin: 28px 0; padding: 20px; border-radius: 6px; background: #f5f5f5; font-size: 15px; line-height: 1.5; }}
.article-content code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }}
.article-content :not(pre) > code {{ padding: 2px 5px; border-radius: 3px; background: #f2f2f2; font-size: 0.82em; }}
.article-content hr {{ margin: 48px 0; border: 0; text-align: center; }}
.article-content hr::after {{ content: "···"; letter-spacing: 14px; }}
{formatter_css}
@media (max-width: 760px) {{
  .article-shell {{ margin: 0; padding: 36px 22px 60px; box-shadow: none; }}
  .article-content {{ font-size: 19px; }}
  .article-content h1 {{ font-size: 36px; }}
  .instructions {{ margin: 16px; }}
}}
</style>
</head>
<body>
<div class="toolbar">
  <div class="toolbar-inner">
    <strong>Medium export · {html.escape(source_name)}</strong>
    <button class="primary" onclick="copyRichArticle()">Copy rich article</button>
    <button onclick="selectArticle()">Select article</button>
    <a href="index.html">Export index</a>
    <span id="copy-status" class="copy-status" aria-live="polite"></span>
  </div>
</div>
<aside class="instructions">
  <h2>Transfer notes</h2>
  <p><strong>{html.escape(warning)}</strong></p>
  <p><strong>Suggested Medium tags:</strong> {tag_note}</p>
  <p>This preview contains {code_blocks} code block(s) and {len(table_files)} rendered table image(s). After pasting, recreate code using Medium’s native code-block control and upload every image from the export package.</p>
  <details><summary>Images required by this article</summary><ul>{asset_items}</ul></details>
</aside>
<main class="article-shell">
  <article class="article-content">{article_html}</article>
</main>
<script>
async function copyRichArticle() {{
  const source = document.querySelector('.article-content');
  const clone = source.cloneNode(true);
  const status = document.getElementById('copy-status');
  clone.querySelectorAll('img').forEach((img) => img.src = new URL(img.getAttribute('src'), location.href).href);
  clone.querySelectorAll('a').forEach((a) => a.href = new URL(a.getAttribute('href'), location.href).href);
  const rich = new Blob([clone.innerHTML], {{type: 'text/html'}});
  const plain = new Blob([source.innerText], {{type: 'text/plain'}});
  try {{
    await navigator.clipboard.write([new ClipboardItem({{'text/html': rich, 'text/plain': plain}})]);
    status.textContent = 'Copied as rich text. Paste it into a new Medium draft.';
  }} catch (error) {{
    selectArticle();
    const copied = document.execCommand('copy');
    status.textContent = copied
      ? 'Copied using the compatibility fallback. Paste it into Medium.'
      : 'Clipboard access was blocked. The article is selected; press Command/Ctrl+C.';
  }}
}}
function selectArticle() {{
  const range = document.createRange();
  range.selectNodeContents(document.querySelector('.article-content'));
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
  document.getElementById('copy-status').textContent = 'Article selected. Press Command/Ctrl+C to copy it.';
}}
</script>
</body>
</html>
"""


def build_index(manifest: list[dict[str, object]]) -> None:
    cards = []
    for item in manifest:
        part_label = f"PART {item['part']}"
        cards.append(
            f"""<article class="card">
<div class="part">{part_label}</div>
<h2><a href="{html.escape(str(item['html']))}">{html.escape(str(item['title']))}</a></h2>
<p>{item['code_blocks']} code block(s) · {len(item['table_images'])} table image(s) · {len(item['images'])} total image reference(s)</p>
</article>"""
        )

    OUTPUT_ROOT.joinpath("index.html").write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Medium export package</title><style>
body {{ max-width: 1050px; margin: 48px auto; padding: 0 24px; color: #242424; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f2; }}
h1 {{ font-size: 46px; margin-bottom: 10px; }}
.intro {{ max-width: 760px; font-size: 19px; line-height: 1.55; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 18px; margin-top: 36px; }}
.card {{ padding: 24px; border-radius: 12px; background: white; box-shadow: 0 1px 7px #0001; }}
.part {{ color: #557087; font-size: 12px; font-weight: 800; letter-spacing: .12em; }}
.card h2 {{ font-size: 22px; line-height: 1.25; }}
a {{ color: #17324d; }}
.card p {{ color: #666; font-size: 14px; }}
</style></head><body><h1>Sequence Models for Prediction</h1><p class="intro">Copy-friendly Medium previews for all {len(manifest)} articles. Open an article, use <strong>Copy rich article</strong>, paste into Medium, then recreate native code blocks, upload the listed images, and replace local series links with live URLs.</p><div class="grid">{''.join(cards)}</div></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SERIES_ROOT / "assets", ASSET_ROOT, dirs_exist_ok=True)
    shutil.copytree(SERIES_ROOT / "code", CODE_ROOT, dirs_exist_ok=True)

    numbered = sorted(SERIES_ROOT.glob("[0-9][0-9]-*.md"))
    companion = SERIES_ROOT / "pytorch-data-pipeline-and-training-loop.md"
    sources = [numbered[0], companion, *numbered[1:]]
    manifest = [
        process_article(source, part)
        for part, source in enumerate(sources, start=1)
    ]
    build_index(manifest)
    OUTPUT_ROOT.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Built {len(manifest)} Medium exports in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
