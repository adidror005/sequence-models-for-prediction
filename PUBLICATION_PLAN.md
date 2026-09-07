# Publication Plan: Sequence Models for Prediction

## Recommended cadence

Publish **two articles per week**, approximately three or four days apart. Do not release all 16 on consecutive days.

A consistent Tuesday/Friday or Monday/Thursday rhythm gives each article time to circulate before the next one reaches the same followers. It also leaves enough time to answer comments, share a second promotional angle, and use early reader questions to improve later drafts.

Medium supports scheduling in the writer’s local timezone. Choose one publication time for the first four posts, then use story statistics—especially presentations, views, reads, and follower conversions—to decide whether the time should change. Avoid changing topic, day, and time simultaneously, because then the source of any performance change becomes impossible to identify.

## Eight-week release sequence

| Week | First release | Second release | Editorial purpose |
|---:|---|---|---|
| 1 | Part 1: Practical map | Part 2: PyTorch pipeline | Establish the series and its shared leak-free implementation. |
| 2 | Part 3: Linear | Part 4: MLP | Move from a transparent baseline to fixed-window nonlinearity. |
| 3 | Part 5: LSTM | Part 6: GRU | Compare two approaches to gated recurrent memory. |
| 4 | Part 7: CNN1D | Part 8: TCN | Move from local filters to designed long-range receptive fields. |
| 5 | Part 9: Transformer | Part 10: Patch Transformer | Contrast time-step attention with temporal tokens. |
| 6 | Part 11: N-BEATS-style | Part 12: Choosing a model | Finish the algorithms and synthesize the design choices. |
| 7 | Part 13: Electricity comparison | Part 14: Calendar and lag features | Move from controlled model comparison to input representation. |
| 8 | Part 15: Trustworthy experiments | Part 16: Local vs. global stock models | Move from reusable protocol to per-symbol datasets and pooled financial learning. |

If one post receives unusually strong discussion, delaying the next release by one or two days is reasonable. Preserve the order; the conceptual progression matters more than hitting an exact date.

The [PyTorch data-pipeline article](pytorch-data-pipeline-and-training-loop.md) is Part 2. It gives readers the shared Dataset, DataLoader, chronological splitting, training-loop, and evaluation machinery before the architecture-specific articles begin.

## The launch pattern for every article

### Before publication

- Read the article as a standalone post; many readers will enter in the middle of the series.
- Confirm that the title promises one clear idea rather than merely naming a part number.
- Select a strong preview image and verify its mobile crop.
- Use no more than five precise Medium topics.
- Check equations, code blocks, image captions, and alt text after pasting into Medium.
- Run the article’s complete implementation and shape assertion in the target PyTorch environment; local syntax validation alone does not replace an execution test.
- Keep the full model class in the article, then link to the shared training pipeline instead of repeating the same loader and early-stopping code nine times.
- Replace local draft links with public Medium URLs.
- Confirm that every result says whether it is preliminary, single-seed, or not yet measured.
- Keep the global CatBoost reality check in Part 1 and the full treatment in Part 12. Do not add the same warning to every algorithm article, and do not imply that the electricity notebook tested gradient boosting.

### Publication day

- Publish or submit at the planned time.
- Share one concise post explaining the reader’s problem and the article’s main payoff.
- Respond to substantive comments while the article is fresh.
- Record the live URL in the registry below.

### Two or three days later

- Share a second angle rather than repeating the original announcement: one diagram, counterintuitive conclusion, equation, or implementation mistake.
- Link to the article from the previously published installment.
- Add the new live URL to the public series index in Part 1.
- Record presentations, views, reads, read ratio, and followers gained. Compare patterns only after several posts; one article is too noisy to set the strategy.

## Cross-linking workflow

Every draft contains local previous/index/next navigation. Those links are useful inside this folder but will not work on Medium.

For each release:

1. Replace **Series index** with the public URL for Part 1.
2. Replace **Previous** with the prior article’s live URL.
3. Leave the next article as an unlinked teaser while it is unpublished.
4. After the next article goes live, edit the previous article and add its URL.
5. Update Part 1 so it remains the public table of contents.

This creates entry points from search and Medium recommendations without requiring readers to discover the series in order.

## Best candidates for publication submissions

Prioritize these articles when approaching established data-science, machine-learning, or software publications:

1. **Part 1** — broadest entry point and public series index.
2. **Part 9** — Transformer interest can introduce new readers to the series.
3. **Part 12** — practical model-selection framing, including the CatBoost benchmark.
4. **Part 13** — original electricity comparison and strongest chart-led story.
5. **Part 14** — counterintuitive feature-engineering result and methodological lesson.
6. **Part 16** — finance interest, a compressed leaderboard, and a careful distinction between AUC and profit.

Publication editors control their own queues. Coordinate timing with them instead of promising readers an exact release date that an external editor may change.

## Promotion angles

Avoid announcing every post as “Part N is live.” Lead with the claim or question.

| Article group | Example hook |
|---|---|
| Foundations | “A time-series model cannot be evaluated fairly until the forecast origin is explicit.” |
| Linear and MLP | “Before reaching for attention, find out how much of the future is already a weighted combination of the past.” |
| LSTM and GRU | “Gates do not create infinite memory; they create a trainable information bottleneck.” |
| CNN and TCN | “Your input may contain 168 hours while your network can access only 31.” |
| Transformers | “Global access is not the same as useful inductive bias.” |
| Model selection | “The uncomfortable benchmark: CatBoost may beat the architecture you spent a week tuning.” |
| Electricity results | “Nine architectures received the same raw week. Seven finished within roughly 0.009 kW RMSE.” |
| Feature comparison | “A causal lag can still make an ablation unfair by quietly doubling historical reach.” |
| Experimental design | “A feature changes data, representation, and information path at the same time.” |
| Finance case study | “Should every stock get its own sequence model—or should one shared model learn across the market using a ticker embedding?” |

## Live URL registry

Fill this table as articles are published. It is the source of truth for updating Medium navigation.

| Part | Status | Live Medium URL | Publication | Release date |
|---:|---|---|---|---|
| 1 | Draft |  |  |  |
| 2 | Draft |  |  |  |
| 3 | Draft |  |  |  |
| 4 | Draft |  |  |  |
| 5 | Draft |  |  |  |
| 6 | Draft |  |  |  |
| 7 | Draft |  |  |  |
| 8 | Draft |  |  |  |
| 9 | Draft |  |  |  |
| 10 | Draft |  |  |  |
| 11 | Draft |  |  |  |
| 12 | Draft |  |  |  |
| 13 | Draft |  |  |  |
| 14 | Draft |  |  |  |
| 15 | Draft |  |  |  |
| 16 | Draft |  |  |  |

## Platform references

- [What happens when a story is published on Medium](https://help.medium.com/hc/en-us/articles/360018677974-What-happens-to-your-story-when-you-publish-on-Medium)
- [How to schedule a Medium story](https://help.medium.com/hc/en-us/articles/216650227-Schedule-to-publish)
- [How to submit a story to a Medium publication](https://help.medium.com/hc/en-us/articles/213904978-How-to-submit-a-story-to-a-publication)
